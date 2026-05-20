"""SSL reinvestigation: proper architecture match, K sweep, label fractions.

Root cause of SSL failure: simplified model dropped pair_proj, fusion, pos_enc
layers. Only 50/100 SSL keys transferred.

This experiment uses the FULL AssociationEncoder architecture for downstream,
enabling perfect SSL weight transfer. Tests:
- K: 0 (dense), 4, 8, 16, 32
- Init: random vs SSL
- Data fraction: 10%, 50%, 100%
"""

import csv, logging, sys, time, gc, math
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

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
from track_encoder import AssociationEncoder, AgentCentricNormalization, TransformerEncoder


# ============================================================
# Full downstream model (matches SSL architecture)
# ============================================================

class FullDownstreamModel(nn.Module):
    """Matches SSL AssociationEncoder architecture for proper weight transfer."""

    def __init__(self, feat_dim=7, coord_dim=2, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=256, dropout=0.1, sparse_k=None):
        super().__init__()

        # Same input projection as SSL encoder
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.pair_norm = AgentCentricNormalization(ndim=coord_dim)
        self.pair_proj = nn.Linear(coord_dim * 2 + 1, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Encoder: dense or sparse
        if sparse_k is not None and sparse_k > 0:
            self.encoder = SparseTransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, knn_k=sparse_k
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            # Add norm to match SSL's TransformerEncoder wrapper
            self.encoder = SSLTransformerEncoderWrapper(self.encoder, d_model)

        # Simple pairwise head
        self.head = nn.Linear(d_model * 2, 1)

    def encode(self, cs, fs, ct, ft, ps=None, pt=None):
        """Shared encode step — returns src_enc, tgt_enc."""
        f_src = self.feat_proj(fs)
        f_tgt = self.feat_proj(ft)
        c_src = self.coord_proj(cs)
        c_tgt = self.coord_proj(ct)

        pair = self.pair_norm(cs, ct)  # (B, N_src, N_tgt, 2*ndim+1)
        pair_enc = self.pair_proj(pair)  # (B, N_src, N_tgt, D)
        ctx_src = pair_enc.mean(dim=2)  # (B, N_src, D)
        ctx_tgt = pair_enc.mean(dim=1)  # (B, N_tgt, D)

        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))

        both = torch.cat([src_feat, tgt_feat], dim=1)
        pad_both = torch.cat([ps, pt], dim=1) if ps is not None else None

        encoded = self.encoder(both, mask=pad_both)
        N_s = cs.shape[1]
        return encoded[:, :N_s], encoded[:, N_s:]

    def forward(self, cs, fs, ct, ft, ps=None, pt=None):
        src_enc, tgt_enc = self.encode(cs, fs, ct, ft, ps, pt)
        B, N1, D = src_enc.shape
        N2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, N2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, N1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


class SSLTransformerEncoderWrapper(nn.Module):
    """Wraps TransformerEncoder + pos_enc + norm to match SSL API."""
    def __init__(self, encoder, d_model):
        super().__init__()
        self.encoder = encoder
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


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
    """Encoder with GatherSparseAttention."""

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
        B, N, D = coords.shape
        yx = coords[..., -2:]
        dist = torch.cdist(yx, yx)
        _, idx = torch.topk(dist, k=min(self.knn_k, N), dim=-1, largest=False)
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


def init_from_ssl(model, ssl_state):
    """Transfer ALL matching SSL weights (full architecture match)."""
    own = model.state_dict()
    mapped = 0
    for k in own:
        # Try exact match first
        if k in ssl_state and own[k].shape == ssl_state[k].shape:
            own[k] = ssl_state[k].clone()
            mapped += 1
        # Try decoder → head mapping
        elif k == "head.weight" and "decoder.assoc_head.weight" in ssl_state:
            w = ssl_state["decoder.assoc_head.weight"]  # (1, 256)
            own[k] = w.clone()
            mapped += 1
        elif k == "head.bias" and "decoder.assoc_head.bias" in ssl_state:
            own[k] = ssl_state["decoder.assoc_head.bias"].clone()
            mapped += 1
        # Map encoder.encoder → encoder for the wrapper vs SSL naming
        elif k.startswith("encoder.encoder."):
            ssl_k = k.replace("encoder.encoder.", "encoder.", 1)
            if ssl_k in ssl_state and own[k].shape == ssl_state[ssl_k].shape:
                own[k] = ssl_state[ssl_k].clone()
                mapped += 1

    model.load_state_dict(own, strict=False)
    logger.info(f"SSL weight transfer: {mapped}/{len(own)} keys mapped, {len(own)-mapped} random")
    return mapped


# ============================================================
# Data
# ============================================================

def make_synthetic_data(N, B=2, n_pairs=200):
    """Generate N synthetic src→tgt pairs with controlled cells."""
    pairs = []
    for _ in range(n_pairs):
        cs = torch.randn(B, N, 2) * 100
        ct = cs + torch.randn(B, N, 2) * 3
        fs = torch.randn(B, N, 7)
        ft = torch.randn(B, N, 7)
        a = torch.eye(N).unsqueeze(0).expand(B, -1, -1).float()
        ps = torch.zeros(B, N, dtype=torch.bool)
        pt = torch.zeros(B, N, dtype=torch.bool)
        pairs.append({"cs": cs, "ct": ct, "fs": fs, "ft": ft,
                       "a": a, "ps": ps, "pt": pt})
    return pairs


def train_model(model, train_pairs, val_pairs, n_epochs=15, lr=3e-4, device="cuda"):
    """Train model, return loss history."""
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pw = torch.tensor(10.0, device=device)
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tls, tas = [], []
        for b in train_pairs:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            opt.zero_grad()
            logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
            loss = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(loss.item())
            with torch.no_grad():
                pred = (logits.detach() > 0).float()
                tas.append((pred == b["a"]).float().mean().item())

        model.eval()
        vls, vas = [], []
        with torch.no_grad():
            for b in val_pairs:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                loss = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pw)
                vls.append(loss.item())
                pred = (logits > 0).float()
                vas.append((pred == b["a"]).float().mean().item())

        et = time.perf_counter() - t0
        rec = {"epoch": epoch, "tl": float(np.mean(tls)), "ta": float(np.mean(tas)),
               "vl": float(np.mean(vls)), "va": float(np.mean(vas)), "t": et}
        history.append(rec)
        if epoch == 1 or epoch % 5 == 0:
            logger.info(f"  Ep {epoch:2d}: tl={rec['tl']:.4f} vl={rec['vl']:.4f} [{et*1000:.0f}ms]")
    return history


# ============================================================
# Main
# ============================================================

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    SSL_PATH = ROOT / "benchmark_ssl" / "runs" / "ssl_v1" / "best_model.pt"
    ssl_state = None
    if SSL_PATH.exists():
        ssl_state = torch.load(SSL_PATH, map_location="cpu", weights_only=False)["model_state_dict"]
        logger.info(f"SSL checkpoint loaded ({len(ssl_state)} keys)")

    outdir = ROOT / "benchmark_combined" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ssl_reinvestigation.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sparse_k", "init", "data_frac", "epoch",
                     "train_loss", "train_acc", "val_loss", "val_acc", "time_s"])

    # Parameters
    N = 128  # small enough for full model with pair_norm
    TOTAL_PAIRS = 200
    VAL_PAIRS = 40
    F = 1  # data fraction multiplier: 1, 0.5, 0.1
    K_VALS = [0, 4, 8, 16, 32]  # 0 = dense
    N_EPOCHS = 10

    # Generate full dataset
    all_pairs = make_synthetic_data(N, B=2, n_pairs=TOTAL_PAIRS + VAL_PAIRS)
    val_pairs = all_pairs[:VAL_PAIRS]
    train_pool = all_pairs[VAL_PAIRS:]

    for sparse_k in K_VALS:
        desc = "dense" if sparse_k == 0 else f"K={sparse_k}"
        for data_frac, frac_label in [(1.0, "100%"), (0.5, "50%"), (0.1, "10%")]:
            n_train = max(1, int(len(train_pool) * data_frac))
            train_pairs = train_pool[:n_train]

            for init_name, use_ssl in [("rand", False), ("ssl", True)]:
                tag = f"{desc}+{init_name}@{frac_label}"
                try:
                    model = FullDownstreamModel(sparse_k=sparse_k if sparse_k > 0 else None).to(device)
                    n_params = sum(p.numel() for p in model.parameters())
                    if use_ssl and ssl_state is not None:
                        n_mapped = init_from_ssl(model, ssl_state)

                    history = train_model(model, train_pairs, val_pairs, N_EPOCHS, device=device)

                    for rec in history:
                        with open(csv_path, "a", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([sparse_k, init_name, frac_label, rec["epoch"],
                                        f"{rec['tl']:.6f}", f"{rec['ta']:.4f}",
                                        f"{rec['vl']:.6f}", f"{rec['va']:.4f}",
                                        f"{rec['t']:.3f}"])

                    logger.info(f"{tag}: final vl={history[-1]['vl']:.4f} {n_params:,} params")

                except RuntimeError as e:
                    logger.info(f"{tag}: FAILED ({e})")
                    continue
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

    logger.info(f"\nDone. Results: {csv_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        final = {}
        for r in reader:
            if int(r["epoch"]) == N_EPOCHS:
                key = (int(r["sparse_k"]), r["init"], r["data_frac"])
                final[key] = float(r["val_loss"])
    for (k, init, frac), vl in sorted(final.items()):
        logger.info(f"  K={k:>2} {init:>4} {frac:>4}: vl={vl:.4f}")


if __name__ == "__main__":
    run()
