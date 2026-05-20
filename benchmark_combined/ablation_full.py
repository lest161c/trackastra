"""Full ablation: all attention × K × L × N × init combos.

Phase 1 — Speed scaling (1 epoch, all N): measure time + memory
Phase 2 — Convergence (15 epochs, N=2048): measure val loss over time
"""

import csv, logging, sys, yaml, time, os, math, gc
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


# ============================================================
# Model variants
# ============================================================

class DenseEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None, coords=None):
        return self.norm(self.encoder(x, src_key_padding_mask=mask))


class SparseEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, knn_k=16):
        super().__init__()
        self.knn_k = knn_k
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GatherSparseAttention(d_model, nhead, knn_k, dropout=dropout, mode="none")
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation="gelu", batch_first=True, norm_first=True)
            self.layers.append(nn.ModuleDict({
                "attn": attn,
                "linear1": layer.linear1, "linear2": layer.linear2,
                "norm1": layer.norm1, "norm2": layer.norm2,
                "dropout": layer.dropout, "dropout1": layer.dropout1, "dropout2": layer.dropout2,
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
        if coords is None:
            return self.norm(x)
        knn_idx = self.compute_knn(coords)
        for layer in self.layers:
            attn_out = layer["attn"](x, x, x, knn_idx, coords)
            x = layer["norm1"](x + layer["dropout1"](attn_out))
            ff = layer["linear2"](layer["dropout"](F.gelu(layer["linear1"](x))))
            x = layer["norm2"](x + layer["dropout2"](ff))
        return self.norm(x)


class AblationModel(nn.Module):
    """Simplified model: no O(N²) pairwise norm. Just proj + encoder + linear head."""

    def __init__(self, encoder, feat_dim=7, coord_dim=2, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(feat_dim + coord_dim, d_model)
        self.head = nn.Linear(d_model * 2, 1)
        self.coord_dim = coord_dim

    def forward(self, cs, fs, ct, ft, ps=None, pt=None):
        # Concatenate features + coords, project
        src_in = torch.cat([fs, cs], dim=-1)
        tgt_in = torch.cat([ft, ct], dim=-1)
        src = self.proj(src_in)
        tgt = self.proj(tgt_in)

        both = torch.cat([src, tgt], dim=1)
        pad_both = torch.cat([ps, pt], dim=1) if ps is not None else None
        coords_both = torch.cat([cs, ct], dim=1)
        encoded = self.encoder(both, mask=pad_both,
                               coords=coords_both if isinstance(self.encoder, SparseEncoder) else None)

        N_s = cs.shape[1]
        src_enc, tgt_enc = encoded[:, :N_s], encoded[:, N_s:]
        B, N1, D = src_enc.shape; N2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, N2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, N1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


def init_from_ssl(model, ssl_state):
    """Transfer SSL weights to simplified model."""
    own = model.state_dict()
    # Build proj from feat_proj + coord_proj weights
    if "feat_proj.weight" in ssl_state and "coord_proj.weight" in ssl_state:
        w_f = ssl_state["feat_proj.weight"]  # (D, feat_dim)
        w_c = ssl_state["coord_proj.weight"]  # (D, coord_dim)
        own["proj.weight"] = torch.cat([w_f, w_c], dim=1)
    if "feat_proj.bias" in ssl_state:
        own["proj.bias"] = ssl_state["feat_proj.bias"]

    # Transfer transformer encoder weights
    ssl_enc = {k: v for k, v in ssl_state.items() if k.startswith("encoder.")}
    own_enc = {k: v for k, v in own.items() if k.startswith("encoder.")}
    for k in own_enc:
        sk = k
        if sk in ssl_enc and own_enc[k].shape == ssl_enc[sk].shape:
            own_enc[k] = ssl_enc[sk]
    own.update(own_enc)
    model.load_state_dict(own, strict=False)


# ============================================================
# Synthetic data generator
# ============================================================

def make_synthetic_batch(N, B=2, feat_dim=7, coord_dim=2):
    """Generate synthetic src→tgt pair with controlled N cells."""
    cs = torch.randn(B, N, coord_dim) * 100
    ct = cs + torch.randn(B, N, coord_dim) * 3  # small movement
    fs = torch.randn(B, N, feat_dim)
    ft = torch.randn(B, N, feat_dim)
    # Identity association: i→i for all N
    a = torch.eye(N).unsqueeze(0).expand(B, -1, -1).float()
    ps = torch.zeros(B, N, dtype=torch.bool)
    pt = torch.zeros(B, N, dtype=torch.bool)
    return {"cs": cs, "ct": ct, "fs": fs, "ft": ft, "a": a, "ps": ps, "pt": pt}


def measure_timing(model, batch, device, warmup=3, repeat=10):
    """Measure forward+backward time and peak memory."""
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    opt = AdamW(model.parameters(), lr=3e-4)
    pos_w = torch.tensor(10.0, device=device)

    for _ in range(warmup):
        logits = model(batch["cs"], batch["fs"], batch["ct"], batch["ft"], batch["ps"], batch["pt"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["a"], pos_weight=pos_w)
        loss.backward(); opt.step(); opt.zero_grad()

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    for _ in range(repeat):
        logits = model(batch["cs"], batch["fs"], batch["ct"], batch["ft"], batch["ps"], batch["pt"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["a"], pos_weight=pos_w)
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / repeat
    mem = (torch.cuda.max_memory_allocated() - mem_before) / (1024**2)
    return t, mem


# ============================================================
# Main
# ============================================================

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    ssl_ckpt = ROOT / "benchmark_ssl" / "runs" / "ssl_v1" / "best_model.pt"
    ssl_state = None
    if ssl_ckpt.exists():
        ssl_state = torch.load(ssl_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        logger.info("SSL checkpoint loaded")

    outdir = ROOT / "benchmark_combined" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ablation_full.csv"

    D, H = 128, 4

    # Phase 1 — Speed scaling: all N × attn × L (1 epoch each, random init)
    N_vals = [128, 256, 512, 1024]
    L_vals = [1, 4]
    attn_configs = [
        ("dense",  lambda L: DenseEncoder(D, H, L)),
        ("sparseK4",  lambda L: SparseEncoder(D, H, L, knn_k=4)),
        ("sparseK16", lambda L: SparseEncoder(D, H, L, knn_k=16)),
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "attn", "L", "K", "N", "init", "epoch",
                     "time_s", "mem_mb", "train_loss", "val_loss", "train_acc", "val_acc", "status"])

    # Phase 1
    logger.info("\n" + "="*60 + "\nPHASE 1: Speed scaling\n" + "="*60)
    for attn_name, attn_fn in attn_configs:
        for L in L_vals:
            for N in N_vals:
                K = 0 if attn_name == "dense" else (4 if "K4" in attn_name else 16)
                try:
                    model = AblationModel(attn_fn(L), d_model=D).to(device)
                    batch = make_synthetic_batch(N, B=2)
                    t, mem = measure_timing(model, batch, device)

                    # Quick loss measurement (1 epoch)
                    opt = AdamW(model.parameters(), lr=3e-4)
                    pos_w = torch.tensor(10.0, device=device)
                    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                    loss = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pos_w)
                    loss.backward(); opt.step()
                    with torch.no_grad():
                        logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                        tl = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pos_w).item()
                        pred = (logits > 0).float()
                        ta = (pred == b["a"]).float().mean().item()

                    status = "ok"
                    logger.info(f"  {attn_name} L={L} N={N:<5}: {t*1000:.1f}ms/step  {mem:.0f}MB  loss={tl:.4f}")
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                        t, mem, tl, ta = -1, -1, -1, -1
                        status = "oom"
                        logger.info(f"  {attn_name} L={L} N={N:<5}: OOM")
                    else:
                        t, mem, tl, ta = -1, -1, -1, -1
                        status = f"err: {e}"

                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["speed", attn_name, L, K, N, "rand", 0,
                                f"{t:.6f}" if t >= 0 else "",
                                f"{mem:.1f}" if mem >= 0 else "",
                                f"{tl:.6f}" if tl >= 0 else "",
                                "", f"{ta:.4f}" if ta >= 0 else "", "", status])

                # Clean up
                del model; gc.collect(); torch.cuda.empty_cache()

    # Phase 2 — Convergence at N=512 (highest OOM-safe for dense on 4GB)
    conv_N = 512
    logger.info(f"\n" + "="*60 + f"\nPHASE 2: Convergence at N={conv_N}\n" + "="*60)

    # Determine which attn variants are OOM-safe
    for L in L_vals:
        for attn_name, attn_fn in attn_configs:
            for init_name, use_ssl in [("rand", False), ("ssl", True)]:
                K = 0 if attn_name == "dense" else (4 if "K4" in attn_name else 16)
                tag = f"{attn_name}+{init_name} L={L}"

                try:
                    # Quick OOM check
                    model = AblationModel(attn_fn(L), d_model=D).to(device)
                    batch = make_synthetic_batch(conv_N, B=2)
                    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                    loss = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=torch.tensor(10.0, device=device))
                    loss.backward()
                    del model; gc.collect(); torch.cuda.empty_cache()
                except RuntimeError:
                    logger.info(f"  {tag}: OOM at N={conv_N}, skipping convergence")
                    continue

                model = AblationModel(attn_fn(L), d_model=D).to(device)
                if use_ssl and ssl_state is not None:
                    init_from_ssl(model, ssl_state)

                opt = AdamW(model.parameters(), lr=3e-4); pos_w = torch.tensor(10.0, device=device)
                params = sum(p.numel() for p in model.parameters())
                logger.info(f"  {tag}: {params:,} params")

                for epoch in range(1, 16):
                    try:
                        t0 = time.perf_counter()

                        # Train
                        model.train()
                        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                        opt.zero_grad()
                        logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                        loss = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pos_w)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        tl = loss.item()
                        with torch.no_grad():
                            pred = (logits.detach() > 0).float()
                            ta = (pred == b["a"]).float().mean().item()

                        # Val (same as train since synthetic — just measure)
                        model.eval()
                        with torch.no_grad():
                            logits = model(b["cs"], b["fs"], b["ct"], b["ft"], b["ps"], b["pt"])
                            vl = F.binary_cross_entropy_with_logits(logits, b["a"], pos_weight=pos_w).item()
                            pred = (logits > 0).float()
                            va = (pred == b["a"]).float().mean().item()

                        et = time.perf_counter() - t0
                        status = "ok"
                    except RuntimeError as e:
                        tl, ta, vl, va, et = -1, -1, -1, -1, -1
                        status = "oom"

                    with open(csv_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["converge", attn_name, L, K, conv_N, init_name, epoch,
                                    f"{et:.6f}" if et >= 0 else "",
                                    "",
                                    f"{tl:.6f}" if tl >= 0 else "",
                                    f"{vl:.6f}" if vl >= 0 else "",
                                    f"{ta:.4f}" if ta >= 0 else "",
                                    f"{va:.4f}" if va >= 0 else "",
                                    status])

                    if epoch == 1 or epoch % 5 == 0:
                        logger.info(f"    Ep {epoch:2d}: tl={tl:.4f} vl={vl:.4f} [{et*1000:.0f}ms]" if status == "ok"
                                    else f"    Ep {epoch:2d}: {status}")

                    if status == "oom":
                        break

                del model; gc.collect(); torch.cuda.empty_cache()

    logger.info(f"\nDone. Results: {csv_path}")
    
    # Summary
    logger.info("\n" + "="*60 + "\nKEY SPEEDUP NUMBERS\n" + "="*60)
    speed_df = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["phase"] == "speed" and r["status"] == "ok":
                key = (r["attn"], int(r["L"]), int(r["N"]))
                if key not in speed_df or float(r["time_s"]) < speed_df[key][0]:
                    speed_df[key] = (float(r["time_s"]), float(r["mem_mb"]) if r["mem_mb"] else 0)

    for L in L_vals:
        for N in [512, 1024]:
            dense_key = ("dense", L, N)
            if dense_key not in speed_df:
                continue
            dt, dm = speed_df[dense_key]
            for k_name in ["sparseK4", "sparseK16"]:
                sk = (k_name, L, N)
                if sk not in speed_df: continue
                st, sm = speed_df[sk]
                speedup = dt / st if st > 0 else 0
                mem_save = dm / sm if sm > 0 else 0
                logger.info(f"  {k_name} L={L} N={N}: {speedup:.1f}× speed, {mem_save:.0f}× mem vs dense")


if __name__ == "__main__":
    run()
