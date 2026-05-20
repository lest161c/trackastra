"""SSL reinvestigation on REAL vanvliet data.

Fixes:
1. Real tracking pairs (adjacent frames) — harder task, shows differences
2. Proper QKV weight split for sparse encoder SSL transfer
3. Feature-projection-only SSL transfer (feat_proj, coord_proj, pair_proj, fusion)
4. Tests K=4,8,16,32 × SSL/random × 10/50/100% data

Goal: find ideal K + ideal pretraining setup.
"""

import csv, logging, sys, time, gc, math
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "benchmark_attn"))
sys.path.insert(0, str(ROOT / "benchmark_ssl"))

from model_parts import GatherSparseAttention
from track_encoder import AssociationEncoder, AgentCentricNormalization
from ssl_pipeline import load_experiment_frames, features_from_frame


# ============================================================
# Model — same as FullDownstreamModel but with proper SSL init
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SparseTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256,
                 dropout=0.1, knn_k=16):
        super().__init__()
        self.knn_k = knn_k
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GatherSparseAttention(d_model, nhead, knn_k, dropout=dropout, mode="none")
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            self.layers.append(nn.ModuleDict({
                "attn": attn,
                "linear1": layer.linear1, "linear2": layer.linear2,
                "norm1": layer.norm1, "norm2": layer.norm2,
                "dropout": layer.dropout,
                "dropout1": layer.dropout1, "dropout2": layer.dropout2,
            }))
        self.norm = nn.LayerNorm(d_model)

    def compute_knn(self, coords):
        B, N, _ = coords.shape
        yx = coords[..., -2:]
        dist = torch.cdist(yx, yx)
        k = min(self.knn_k, N)
        _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        if idx.shape[-1] < self.knn_k:
            pad = idx[:, :, -1:].expand(-1, -1, self.knn_k - idx.shape[-1])
            idx = torch.cat([idx, pad], dim=-1)
        return idx

    def forward(self, x, mask=None, coords=None):
        x = self.pos_enc(x)
        if coords is None:
            return x
        knn_idx = self.compute_knn(coords)
        for layer in self.layers:
            attn_out = layer["attn"](x, x, x, knn_idx, coords)
            x = layer["norm1"](x + layer["dropout1"](attn_out))
            ff = layer["linear2"](layer["dropout"](F.gelu(layer["linear1"](x))))
            x = layer["norm2"](x + layer["dropout2"](ff))
        return self.norm(x)


class RealDataModel(nn.Module):
    """Full model for real data — matches SSL's feature encoding + custom encoder + head."""
    def __init__(self, feat_dim=7, coord_dim=2, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=256, dropout=0.1, sparse_k=None):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.pair_norm = AgentCentricNormalization(ndim=coord_dim)
        self.pair_proj = nn.Linear(coord_dim * 2 + 1, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        if sparse_k is not None and sparse_k > 0:
            self.encoder = SparseTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout, knn_k=sparse_k)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers)
            self.norm = nn.LayerNorm(d_model)
        self.sparse_k = sparse_k
        self.head = nn.Linear(d_model * 2, 1)

    def _dense_encode(self, x, mask):
        x = self.pos_enc(x)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.norm(x)

    def encode(self, cs, fs, ct, ft, ps, pt, coords_both):
        f_src, f_tgt = self.feat_proj(fs), self.feat_proj(ft)
        c_src, c_tgt = self.coord_proj(cs), self.coord_proj(ct)
        pair = self.pair_norm(cs, ct)
        pair_enc = self.pair_proj(pair)
        ctx_src, ctx_tgt = pair_enc.mean(dim=2), pair_enc.mean(dim=1)
        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))
        both = torch.cat([src_feat, tgt_feat], dim=1)
        pad_both = torch.cat([ps, pt], dim=1) if ps is not None else None

        if self.sparse_k is not None and self.sparse_k > 0:
            encoded = self.encoder(both, mask=pad_both, coords=coords_both)
        else:
            encoded = self._dense_encode(both, pad_both)

        N_s = cs.shape[1]
        return encoded[:, :N_s], encoded[:, N_s:]

    def forward(self, cs, fs, ct, ft, ps=None, pt=None):
        coords_both = torch.cat([cs, ct], dim=1)
        src_enc, tgt_enc = self.encode(cs, fs, ct, ft, ps, pt, coords_both)
        B, N1, D = src_enc.shape
        N2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, N2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, N1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


def init_from_ssl(model, ssl_state):
    """Transfer SSL weights with QKV split for sparse encoder."""
    own = model.state_dict()
    mapped = 0

    # Direct matches: feat_proj, coord_proj, pair_proj, fusion
    for prefix in ["feat_proj", "coord_proj", "pair_proj", "fusion"]:
        for suffix in ["weight", "bias"]:
            k = f"{prefix}.{suffix}"
            if k in ssl_state and k in own:
                own[k] = ssl_state[k].clone()
                mapped += 1

    # Encoder: dense uses identical keys, sparse needs QKV split
    if model.sparse_k is not None and model.sparse_k > 0:
        # Sparse: encoder.layers.{i}.attn.q_pro, .k_pro, .v_pro weights
        # SSL: encoder.encoder.layers.{i}.self_attn.in_proj_weight (fused QKV)
        for layer_i in range(4):
            if f"encoder.layers.{layer_i}.attn.q_pro.weight" not in own:
                continue
            # Fused weight (384, 128) → split into three (128, 128)
            fused_w = ssl_state.get(f"encoder.encoder.layers.{layer_i}.self_attn.in_proj_weight")
            fused_b = ssl_state.get(f"encoder.encoder.layers.{layer_i}.self_attn.in_proj_bias")
            if fused_w is not None:
                D = fused_w.shape[0] // 3
                for j, proj in enumerate(["q", "k", "v"]):
                    own[f"encoder.layers.{layer_i}.attn.{proj}_pro.weight"] = fused_w[j*D:(j+1)*D].clone()
                    if fused_b is not None:
                        own[f"encoder.layers.{layer_i}.attn.{proj}_pro.bias"] = fused_b[j*D:(j+1)*D].clone()
                    mapped += 2

            # Output projection
            for s in ["weight", "bias"]:
                k = f"encoder.layers.{layer_i}.attn.proj.{s}"
                ssl_k = f"encoder.encoder.layers.{layer_i}.self_attn.out_proj.{s}"
                if ssl_k in ssl_state and k in own:
                    own[k] = ssl_state[ssl_k].clone()
                    mapped += 1

            # FFN layers
            for comp in ["linear1", "linear2"]:
                for s in ["weight", "bias"]:
                    k = f"encoder.layers.{layer_i}.{comp}.{s}"
                    ssl_k = f"encoder.encoder.layers.{layer_i}.{comp}.{s}"
                    if ssl_k in ssl_state and k in own:
                        own[k] = ssl_state[ssl_k].clone()
                        mapped += 1

            # Norm layers
            for norm_i in ["norm1", "norm2"]:
                for s in ["weight", "bias"]:
                    k = f"encoder.layers.{layer_i}.{norm_i}.{s}"
                    ssl_k = f"encoder.encoder.layers.{layer_i}.{norm_i}.{s}"
                    if ssl_k in ssl_state and k in own:
                        own[k] = ssl_state[ssl_k].clone()
                        mapped += 1

        # Encoder norm
        for s in ["weight", "bias"]:
            k = f"encoder.norm.{s}"
            ssl_k = f"encoder.norm.{s}"
            if ssl_k in ssl_state and k in own:
                own[k] = ssl_state[ssl_k].clone()
                mapped += 1
    else:
        # Dense: keys match exactly
        for k in own:
            if k in ssl_state and own[k].shape == ssl_state[k].shape:
                own[k] = ssl_state[k].clone()
                mapped += 1

    model.load_state_dict(own, strict=False)
    logger.info(f"  SSL transfer: {mapped}/{len(own)} keys mapped")
    return mapped


# ============================================================
# Real data loading
# ============================================================

def load_real_pairs(frames, max_pairs=200):
    """Create src→tgt pairs from adjacent frames with real assoc labels."""
    pairs = []
    for i in range(0, len(frames) - 1, 2):
        try:
            _, _, _, ms_path, is_path = frames[i]
            _, _, _, mt_path, it_path = frames[i + 1]
            from tifffile import imread
            ms, mt = imread(ms_path), imread(mt_path)
            def _load(p):
                img = imread(p).astype(np.float32)
                p1, p998 = np.percentile(img, (1, 99.8))
                return np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
            rs = features_from_frame(ms, _load(is_path))
            rt = features_from_frame(mt, _load(it_path))
            if rs is None or rt is None:
                continue
            cs, ls, fs_d = rs
            ct, lt, ft_d = rt
            fs = np.concatenate(list(fs_d.values()), axis=-1).astype(np.float32)
            ft = np.concatenate(list(ft_d.values()), axis=-1).astype(np.float32)
            if len(ls) == 0 or len(lt) == 0:
                continue
            a = np.zeros((len(ls), len(lt)), dtype=np.float32)
            lt_map = {int(l): j for j, l in enumerate(lt)}
            for i_s, l_s in enumerate(ls):
                j = lt_map.get(int(l_s))
                if j is not None:
                    a[i_s, j] = 1.0
            pairs.append({
                "cs": torch.from_numpy(cs).float(), "ct": torch.from_numpy(ct).float(),
                "fs": torch.from_numpy(fs).float(), "ft": torch.from_numpy(ft).float(),
                "a": torch.from_numpy(a).float(),
                "ns": len(ls), "nt": len(lt),
            })
            if len(pairs) >= max_pairs:
                break
        except Exception:
            continue
    return pairs


# ============================================================
# Training
# ============================================================

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    SSL_PATH = ROOT / "benchmark_ssl" / "runs" / "ssl_v1" / "best_model.pt"
    ssl_state = torch.load(SSL_PATH, map_location="cpu", weights_only=False)["model_state_dict"]
    logger.info(f"SSL checkpoint: {len(ssl_state)} keys")

    # Load real data
    frames = load_experiment_frames(str(ROOT / "data/vanvliet"), conditions=["rpsM", "recA", "pheA"])
    np.random.seed(42)
    np.random.shuffle(frames)
    all_pairs = load_real_pairs(frames, max_pairs=300)
    val_pairs = all_pairs[:40]
    train_pool = all_pairs[40:]
    logger.info(f"Real pairs: {len(train_pool)} train + {len(val_pairs)} val")

    outdir = ROOT / "benchmark_combined" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ssl_realdata.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "init", "frac", "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "time_s"])

    K_VALS = [0, 4, 8, 16, 32]
    FRACTIONS = [(1.0, "100%"), (0.5, "50%"), (0.1, "10%")]
    N_EPOCHS = 10

    for K in K_VALS:
        for frac, flabel in FRACTIONS:
            n = max(1, int(len(train_pool) * frac))
            train_pairs = train_pool[:n]

            for init_name, ssl_enabled in [("rand", False), ("ssl", True)]:
                tag = f"{'dense' if K==0 else f'K={K}'}+{init_name}@{flabel}"
                try:
                    model = RealDataModel(sparse_k=K if K > 0 else None).to(device)
                    params = sum(p.numel() for p in model.parameters())
                    if ssl_enabled:
                        n_map = init_from_ssl(model, ssl_state)

                    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
                    pw = torch.tensor(10.0, device=device)

                    for epoch in range(1, N_EPOCHS + 1):
                        t0 = time.perf_counter()
                        model.train()
                        tls, tas = [], []
                        for b in train_pairs:
                            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                            opt.zero_grad()
                            # Add batch dim (B=1)
                            logits = model(b["cs"].unsqueeze(0), b["fs"].unsqueeze(0),
                                           b["ct"].unsqueeze(0), b["ft"].unsqueeze(0))
                            loss = F.binary_cross_entropy_with_logits(logits, b["a"].unsqueeze(0), pos_weight=pw)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                            tls.append(loss.item())
                            with torch.no_grad():
                                acc = ((logits.detach() > 0).float() == b["a"].unsqueeze(0)).float().mean().item()
                                tas.append(acc)

                        model.eval()
                        vls, vas = [], []
                        with torch.no_grad():
                            for b in val_pairs:
                                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                                cs, fs, ct, ft = b["cs"].unsqueeze(0), b["fs"].unsqueeze(0), b["ct"].unsqueeze(0), b["ft"].unsqueeze(0)
                                logits = model(cs, fs, ct, ft)
                                loss = F.binary_cross_entropy_with_logits(logits, b["a"].unsqueeze(0), pos_weight=pw)
                                vls.append(loss.item())
                                acc = ((logits > 0).float() == b["a"].unsqueeze(0)).float().mean().item()
                                vas.append(acc)

                        et = time.perf_counter() - t0
                        tl, ta = float(np.mean(tls)), float(np.mean(tas))
                        vl, va = float(np.mean(vls)), float(np.mean(vas))

                        with open(csv_path, "a", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([K, init_name, flabel, epoch,
                                        f"{tl:.6f}", f"{vl:.6f}", f"{ta:.4f}", f"{va:.4f}", f"{et:.2f}"])

                        if epoch == 1 or epoch % 5 == 0:
                            logger.info(f"  {tag} Ep{epoch:2d}: tl={tl:.4f} vl={vl:.4f} [{et:.1f}s]")

                except RuntimeError as e:
                    logger.info(f"  {tag}: FAILED ({e})")
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

    logger.info(f"\nDone. Results: {csv_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL VAL LOSS SUMMARY (epoch 10)")
    logger.info("="*60)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        finals = {}
        for r in reader:
            if int(r["epoch"]) == N_EPOCHS:
                key = (r["K"], r["init"], r["frac"])
                finals[key] = (float(r["val_loss"]), float(r["val_acc"]))

    # Print as table
    logger.info(f"{'K':>4} {'Init':>6} {'Frac':>6} {'Val Loss':>10} {'Val Acc':>10}")
    logger.info("-" * 40)
    for (k, init, frac), (vl, va) in sorted(finals.items()):
        logger.info(f"{k:>4} {init:>6} {frac:>6} {vl:>10.4f} {va:>10.4f}")

    # Best per category
    for flabel in ["10%", "50%", "100%"]:
        subset = {k: v for k, v in finals.items() if k[2] == flabel}
        if not subset:
            continue
        best_key = min(subset, key=lambda x: subset[x][0])
        logger.info(f"Best at {flabel}: K={best_key[0]} {best_key[1]} (vl={subset[best_key][0]:.4f})")


if __name__ == "__main__":
    run()
