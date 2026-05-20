"""Combined benchmark: sparse attention + SSL pretraining vs baseline.

2x2 factorial design:
   Attention: Dense (standard SDPA) vs Sparse (Gather-based O(Nk))
   Init:      Random vs SSL-pretrained

Measures wall-clock time to reach target validation loss.
Shows total speedup from both improvements combined.
"""

import csv, logging, sys, yaml, time, os, math
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Path setup ---
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "benchmark_attn"))
sys.path.insert(0, str(ROOT / "benchmark_ssl"))

from model_parts import GatherSparseAttention
from track_encoder import AssociationEncoder, AgentCentricNormalization, FeedForward
from ssl_pipeline import load_experiment_frames, features_from_frame

# ============================================================
# Data pipeline
# ============================================================

def create_tracking_pairs(frames, max_pairs=80):
    """Create src→tgt pairs from adjacent frames with real assoc labels."""
    pairs = []
    for i in range(0, len(frames) - 1, 2):
        try:
            _, _, _, mask_src, img_src = frames[i]
            _, _, _, mask_tgt, img_tgt = frames[i + 1]
            from tifffile import imread
            ms, mt = imread(mask_src), imread(mask_tgt)
            def load_img(p):
                img = imread(p).astype(np.float32)
                p1, p998 = np.percentile(img, (1, 99.8))
                return np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
            is_ = load_img(img_src); it_ = load_img(img_tgt)
            rs = features_from_frame(ms, is_); rt = features_from_frame(mt, it_)
            if rs is None or rt is None: continue
            cs, ls, fs_d = rs; ct, lt, ft_d = rt
            fs = np.concatenate(list(fs_d.values()), axis=-1).astype(np.float32)
            ft = np.concatenate(list(ft_d.values()), axis=-1).astype(np.float32)
            n_src, n_tgt = len(ls), len(lt)
            if n_src == 0 or n_tgt == 0: continue
            assoc = np.zeros((n_src, n_tgt), dtype=np.float32)
            lt_map = {int(l): j for j, l in enumerate(lt)}
            for i_s, l_s in enumerate(ls):
                j = lt_map.get(int(l_s))
                if j is not None: assoc[i_s, j] = 1.0
            pairs.append({
                "cs": torch.from_numpy(cs).float(), "ct": torch.from_numpy(ct).float(),
                "fs": torch.from_numpy(fs).float(), "ft": torch.from_numpy(ft).float(),
                "a": torch.from_numpy(assoc).float(),
                "ns": n_src, "nt": n_tgt,
            })
            if len(pairs) >= max_pairs: break
        except Exception as e:
            continue
    return pairs


def collate_pairs(batch):
    batch = [b for b in batch if b["ns"] > 0 and b["nt"] > 0]
    if not batch: return None
    mx_s = max(b["ns"] for b in batch); mx_t = max(b["nt"] for b in batch)
    B = len(batch)
    cs = torch.zeros(B, mx_s, 2); ct = torch.zeros(B, mx_t, 2)
    fs = torch.zeros(B, mx_s, 7); ft = torch.zeros(B, mx_t, 7)
    a = torch.zeros(B, mx_s, mx_t)
    ps = torch.ones(B, mx_s, dtype=torch.bool)
    pt = torch.ones(B, mx_t, dtype=torch.bool)
    for i, b in enumerate(batch):
        cs[i, :b["ns"]] = b["cs"]; fs[i, :b["ns"]] = b["fs"]; ps[i, :b["ns"]] = False
        ct[i, :b["nt"]] = b["ct"]; ft[i, :b["nt"]] = b["ft"]; pt[i, :b["nt"]] = False
        a[i, :b["ns"], :b["nt"]] = b["a"]
    return {"cs": cs, "ct": ct, "fs": fs, "ft": ft, "a": a, "ps": ps, "pt": pt}


# ============================================================
# Encoder variants
# ============================================================

class DenseEncoder(nn.Module):
    """Standard TransformerEncoder (dense O(N²) attention)."""
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        return self.norm(self.encoder(x, src_key_padding_mask=mask))


class SparseEncoder(nn.Module):
    """Encoder using GatherSparseAttention (O(Nk) attention)."""
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, knn_k=16):
        super().__init__()
        self.knn_k = knn_k
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GatherSparseAttention(d_model, nhead, knn_k, dropout=dropout, mode="none")
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True, norm_first=True)
            # Replace the self-attention with GatherSparseAttention
            # We'll handle this manually in forward()
            self.layers.append(nn.ModuleDict({
                "attn": attn,
                "linear1": layer.linear1, "linear2": layer.linear2,
                "norm1": layer.norm1, "norm2": layer.norm2,
                "dropout": layer.dropout, "dropout1": layer.dropout1, "dropout2": layer.dropout2,
            }))
        self.norm = nn.LayerNorm(d_model)

    def compute_knn(self, coords):
        """Compute KNN indices from coordinates."""
        B, N, D = coords.shape
        # Use only the last two dims for 2D coords
        yx = coords[..., -2:]
        dist = torch.cdist(yx, yx)
        _, idx = torch.topk(dist, k=min(self.knn_k, N), dim=-1, largest=False)
        # Pad if N < knn_k (shouldn't happen with real data)
        if idx.shape[-1] < self.knn_k:
            pad = idx[:, :, -1:].expand(-1, -1, self.knn_k - idx.shape[-1])
            idx = torch.cat([idx, pad], dim=-1)
        return idx

    def forward(self, x, coords=None, mask=None):
        if coords is None:
            # Fallback to dense via the norm
            return self.norm(x)
        knn_idx = self.compute_knn(coords)
        for layer in self.layers:
            # Self-attention with GatherSparseAttention
            attn_out = layer["attn"](x, x, x, knn_idx, coords)
            x = layer["norm1"](x + layer["dropout1"](attn_out))
            # FFN
            ff = layer["linear2"](layer["dropout"](F.gelu(layer["linear1"](x))))
            x = layer["norm2"](x + layer["dropout2"](ff))
        return self.norm(x)


class UnifiedModel(nn.Module):
    """Model with interchangeable encoder: dense or sparse."""

    def __init__(self, encoder, feat_dim=7, coord_dim=2, d_model=128, use_ssl_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.pair_norm = AgentCentricNormalization(ndim=coord_dim)
        self.pair_proj = nn.Linear(coord_dim * 2 + 1, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )
        self.use_ssl_encoder = use_ssl_encoder

    def forward(self, cs, fs, ct, ft, ps=None, pt=None):
        f_src = self.feat_proj(fs)
        f_tgt = self.feat_proj(ft)
        c_src = self.coord_proj(cs)
        c_tgt = self.coord_proj(ct)

        pair = self.pair_norm(cs, ct)
        pair_enc = self.pair_proj(pair)
        ctx_src = pair_enc.mean(dim=2)
        ctx_tgt = pair_enc.mean(dim=1)

        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))

        both = torch.cat([src_feat, tgt_feat], dim=1)
        pad_both = torch.cat([ps, pt], dim=1) if ps is not None else None

        coords_both = torch.cat([cs, ct], dim=1)
        if isinstance(self.encoder, SparseEncoder):
            encoded = self.encoder(both, coords=coords_both, mask=pad_both)
        else:
            encoded = self.encoder(both, mask=pad_both)

        N_src = cs.shape[1]
        src_enc = encoded[:, :N_src]
        tgt_enc = encoded[:, N_src:]

        B, N_s, D = src_enc.shape; N_t = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, N_t, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, N_s, -1, -1)
        logits = self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)
        return logits


def init_from_ssl(model, ssl_state_dict):
    """Transfer SSL-pretrained encoder weights."""
    own = model.state_dict()
    for k in ["feat_proj.weight", "feat_proj.bias", "coord_proj.weight", "coord_proj.bias",
              "pair_proj.weight", "pair_proj.bias",
              "fusion.0.weight", "fusion.0.bias", "fusion.2.weight", "fusion.2.bias"]:
        if k in ssl_state_dict:
            own[k] = ssl_state_dict[k]
    # Transfer transformer encoder weights if compatible
    ssl_enc = {k: v for k, v in ssl_state_dict.items() if k.startswith("encoder.")}
    own_enc = {k: v for k, v in own.items() if k.startswith("encoder.")}
    for k in own_enc:
        ssl_k = k
        if ssl_k in ssl_enc and own_enc[k].shape == ssl_enc[ssl_k].shape:
            own_enc[k] = ssl_enc[ssl_k]
    own.update(own_enc)
    model.load_state_dict(own, strict=False)


# ============================================================
# Main benchmark
# ============================================================

def run(config_path=None, n_epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    frames = load_experiment_frames(str(ROOT / "data/vanvliet"), conditions=["rpsM", "recA", "pheA"])
    np.random.seed(42); np.random.shuffle(frames)
    n_val = max(1, int(len(frames) * 0.1))
    train_f, val_f = frames[n_val:], frames[:n_val]
    train_pairs = create_tracking_pairs(train_f)
    val_pairs = create_tracking_pairs(val_f)
    logger.info(f"Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")

    # SSL pretrained weights
    ssl_ckpt_path = ROOT / "benchmark_ssl" / "runs" / "ssl_v1" / "best_model.pt"
    ssl_state = None
    if ssl_ckpt_path.exists():
        ssl_state = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        logger.info("Loaded SSL checkpoint")

    # Model configs: 2x2 factorial + K ablation
    D = 128; H = 4; L = 4

    configs = [
        ("dense+rand",   lambda: UnifiedModel(DenseEncoder(D, H, L), d_model=D), False),
        ("dense+ssl",    lambda: UnifiedModel(DenseEncoder(D, H, L), d_model=D), True),
        ("sparseK4+rand", lambda: UnifiedModel(SparseEncoder(D, H, L, knn_k=4), d_model=D), False),
        ("sparseK4+ssl",  lambda: UnifiedModel(SparseEncoder(D, H, L, knn_k=4), d_model=D), True),
        ("sparseK16+rand", lambda: UnifiedModel(SparseEncoder(D, H, L, knn_k=16), d_model=D), False),
        ("sparseK16+ssl",  lambda: UnifiedModel(SparseEncoder(D, H, L, knn_k=16), d_model=D), True),
    ]

    outdir = ROOT / "benchmark_combined" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "combined_benchmark.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"])

    all_results = {}
    for variant_name, model_fn, use_ssl in configs:
        logger.info(f"\n{'='*60}\n  {variant_name}\n{'='*60}")
        model = model_fn().to(device)
        if use_ssl and ssl_state is not None:
            init_from_ssl(model, ssl_state)
            logger.info(f"  SSL init applied")
        logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        pos_w = torch.tensor(10.0, device=device)

        results = []
        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            tl, ta = [], []
            for i in range(0, len(train_pairs), 4):
                batch = collate_pairs(train_pairs[i:i+4])
                if batch is None: continue
                cs, ct = batch["cs"].to(device), batch["ct"].to(device)
                fs, ft = batch["fs"].to(device), batch["ft"].to(device)
                a = batch["a"].to(device)
                ps, pt = batch["ps"].to(device), batch["pt"].to(device)
                opt.zero_grad()
                logits = model(cs, fs, ct, ft, ps, pt)
                loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_w)
                valid = ~(ps[:, :, None] | pt[:, None, :])
                loss = (loss * valid.float()).sum() / valid.float().sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(loss.item())
                acc = ((logits.detach() > 0).float() == a).float()
                ta.append((acc * valid.float()).sum().item() / valid.float().sum().item())

            # Val
            model.eval()
            vl, va = [], []
            with torch.no_grad():
                for i in range(0, len(val_pairs), 4):
                    batch = collate_pairs(val_pairs[i:i+4])
                    if batch is None: continue
                    cs, ct = batch["cs"].to(device), batch["ct"].to(device)
                    fs, ft = batch["fs"].to(device), batch["ft"].to(device)
                    a = batch["a"].to(device)
                    ps, pt = batch["ps"].to(device), batch["pt"].to(device)
                    logits = model(cs, fs, ct, ft, ps, pt)
                    loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_w)
                    valid = ~(ps[:, :, None] | pt[:, None, :])
                    loss = (loss * valid.float()).sum() / valid.float().sum()
                    vl.append(loss.item())
                    acc = ((logits > 0).float() == a).float()
                    va.append((acc * valid.float()).sum().item() / valid.float().sum().item())

            et = time.perf_counter() - t0
            tloss, tacc = float(np.mean(tl)), float(np.mean([v.item() if torch.is_tensor(v) else v for v in ta]))
            vloss, vacc = float(np.mean(vl)), float(np.mean([v.item() if torch.is_tensor(v) else v for v in va]))
            results.append({"e": epoch, "tl": tloss, "ta": tacc, "vl": vloss, "va": vacc, "t": et})

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([variant_name, epoch, f"{tloss:.6f}", f"{tacc:.4f}", f"{vloss:.6f}", f"{vacc:.4f}", f"{et:.2f}"])

            logger.info(f"  Epoch {epoch:2d}: tl={tloss:.4f} vl={vloss:.4f} acc={vacc:.4f} [{et:.1f}s]")

        all_results[variant_name] = results

    logger.info(f"\n{'='*60}\nResults: {csv_path}")
    return all_results


if __name__ == "__main__":
    run()
