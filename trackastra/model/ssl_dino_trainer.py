"""DINO-based contrastive SSL pretraining for Trackastra.

Uses frozen DINOv2 visual backbone + learned projection head + TrackingTransformer
encoder, trained with NT-Xent contrastive loss on geometrically distorted single frames.

Reference:
    ASCENT (Han & Lu 2025): image patches → DINO → transformer encoder → NT-Xent
    Cell-DINO (Moutakanni et al. 2025): DINOv2 SSL for cell microscopy

Usage:
    python -m trackastra.model.ssl_dino_trainer --config configs/ssl_dino_pretrain.yaml
"""

import os, sys, logging, time, yaml, argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trackastra.model import TrackingTransformer
from trackastra.data.ssl_distortions import DistortionPipeline
from trackastra.model.dino_encoder import extract_patches
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ssl_dino_trainer")


# ─── NT-Xent loss (identical to ssl_trainer.py) ──────────────────────────────


def nt_xent_loss(z1, z2, pm1, pm2, temperature=0.05):
    B, N_max, D = z1.shape
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    total_loss = 0.0
    n_frames = 0
    for b in range(B):
        valid = ~pm1[b] & ~pm2[b]
        n = valid.sum().item()
        if n < 2:
            continue
        e1, e2 = z1[b][valid], z2[b][valid]
        features = torch.cat([e1, e2], dim=0)
        sim = torch.matmul(features, features.T) / temperature
        sim = sim - torch.eye(2 * n, device=sim.device) * 1e9
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(features.device)
        total_loss += F.cross_entropy(sim, labels)
        n_frames += 1
    if n_frames == 0:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    return total_loss / n_frames


def embedding_consistency(z1, z2, pm1, pm2):
    valid = ~pm1 & ~pm2
    if valid.sum() == 0:
        return 0.0
    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    return (z1_n * z2_n).sum(dim=-1)[valid].mean().item()


def inter_cell_similarity(z1, pm1):
    valid = ~pm1
    total_sim = 0.0
    n = 0
    for b in range(z1.shape[0]):
        v = valid[b]
        nv = v.sum().item()
        if nv < 2:
            continue
        z = F.normalize(z1[b][v], dim=-1)
        sim = z @ z.T
        mask = 1 - torch.eye(nv, device=sim.device)
        total_sim += (sim * mask).sum().item() / (nv * (nv - 1))
        n += 1
    return total_sim / n if n else 0.0


# ─── data ─────────────────────────────────────────────────────────────────────


def scan_frames(data_root, conditions):
    frames = []
    data_root = Path(data_root)
    for cond in conditions:
        cond_path = data_root / cond
        if not cond_path.is_dir():
            continue
        for exp_path in sorted(cond_path.iterdir()):
            if not exp_path.is_dir():
                continue
            tra_dir = exp_path / "TRA"
            img_dir = exp_path / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            for mpath in sorted(tra_dir.glob("man_track*.tif")):
                stem = mpath.stem.replace("man_track", "")
                try:
                    fidx = int(stem)
                except ValueError:
                    continue
                ipath = img_dir / f"t{fidx:06d}.tif"
                if ipath.exists():
                    frames.append((str(mpath), str(ipath)))
    logger.info(f"Scanned {len(frames)} frames from {len(conditions)} conditions")
    return frames


def load_frame(mask_path, img_path, ndim=2):
    """Load a single frame, extract centroids + labels + image."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

    from skimage.measure import regionprops_table
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) == 0:
        return None

    coords = np.stack(
        [props["centroid-0"], props["centroid-1"]], axis=-1
    ).astype(np.float32)
    labels = props["label"].astype(np.int32)
    return coords, labels, img


class SSLDinoDataset:
    """Dataset: single frame → distortion → patches → DINO → NT-Xent.

    For each frame:
      1. Load image + detect cells (centroids + labels)
      2. Apply DistortionPipeline → two augmented coordinate views
      3. Extract image patches at warped coordinates
      4. Return coords + patches + padding masks for both views
    """

    def __init__(self, frames, distortion_pipeline, ndim=2):
        self.frames = frames
        self.dist = distortion_pipeline
        self.ndim = ndim

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        mask_path, img_path = self.frames[idx]
        result = load_frame(mask_path, img_path, self.ndim)
        if result is None:
            return None

        coords, labels, img = result

        # Apply distortion pipeline to get two views
        # The pipeline expects (coords, features_dict, labels) and returns
        # (c1, f1, l1, c2, f2, l2). We pass dummy features since we use patches.
        dummy_feats = OrderedDict({"dummy": np.zeros((len(labels), 1), dtype=np.float32)})
        c1, _, l1, c2, _, l2 = self.dist(coords.copy(), dummy_feats, labels.copy())

        # Sort by label for positive-pair alignment
        idx1 = np.argsort(l1)
        idx2 = np.argsort(l2)
        c1, l1 = c1[idx1], l1[idx1]
        c2, l2 = c2[idx2], l2[idx2]

        # Extract patches at warped coordinates
        patches1 = extract_patches(img, c1)
        patches2 = extract_patches(img, c2)

        # Time coordinate = 0 for both views (same frame)
        t1 = np.zeros((len(l1), 1), dtype=np.float32)
        t2 = np.zeros((len(l2), 1), dtype=np.float32)
        c1_t = np.concatenate([t1, c1], axis=-1).astype(np.float32)
        c2_t = np.concatenate([t2, c2], axis=-1).astype(np.float32)

        return {
            "coords1": torch.from_numpy(c1_t).float(),
            "coords2": torch.from_numpy(c2_t).float(),
            "patches1": torch.from_numpy(patches1).float().unsqueeze(1),  # (N, 1, 64, 64)
            "patches2": torch.from_numpy(patches2).float().unsqueeze(1),
            "n1": len(l1),
            "n2": len(l2),
        }


def collate_ssl_dino(batch):
    """Collate two-view batches with padding."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_n = max(max(b["n1"], b["n2"]) for b in batch)
    B = len(batch)
    ndim = 3  # time + y + x

    c1 = torch.zeros(B, max_n, ndim)
    c2 = torch.zeros(B, max_n, ndim)
    p1 = torch.zeros(B, max_n, 1, 64, 64)
    p2 = torch.zeros(B, max_n, 1, 64, 64)
    pm1 = torch.ones(B, max_n, dtype=torch.bool)
    pm2 = torch.ones(B, max_n, dtype=torch.bool)

    for i, b in enumerate(batch):
        n1, n2 = b["n1"], b["n2"]
        if n1 > 0:
            c1[i, :n1] = b["coords1"]
            p1[i, :n1] = b["patches1"]
            pm1[i, :n1] = False
        if n2 > 0:
            c2[i, :n2] = b["coords2"]
            p2[i, :n2] = b["patches2"]
            pm2[i, :n2] = False

    return {
        "coords1": c1, "coords2": c2,
        "patches1": p1, "patches2": p2,
        "pm1": pm1, "pm2": pm2,
    }


# ─── training loop ────────────────────────────────────────────────────────────


def train_ssl(cfg, model, device):
    ssl_cfg = cfg["ssl"]
    epochs = ssl_cfg["epochs"]
    lr = ssl_cfg["lr"]
    temperature = ssl_cfg.get("temperature", 0.05)
    batch_size = ssl_cfg["batch_size"]
    val_split = ssl_cfg.get("val_split", 0.1)

    data_root = cfg["data_root"]
    conditions = cfg["conditions"]
    all_frames = scan_frames(data_root, conditions)

    np.random.seed(cfg.get("seed", 42))
    np.random.shuffle(all_frames)
    n_val = max(1, int(len(all_frames) * val_split))
    val_frames, train_frames = all_frames[:n_val], all_frames[n_val:]

    # Distortion pipeline (same as ASCENT §3.1)
    dist_cfg = {
        "distortions": cfg.get("distortions",
            ["affine", "elastic", "jitter", "dropout", "photometric", "feature_noise"]),
        "affine": cfg.get("affine", {}),
        "elastic": cfg.get("elastic", {}),
        "jitter": cfg.get("jitter", {}),
        "dropout": cfg.get("dropout", {}),
        "photometric": cfg.get("photometric", {}),
        "feature_noise": cfg.get("feature_noise", {}),
        "seed": cfg.get("seed", 42),
    }
    dist = DistortionPipeline.from_config(dist_cfg)

    train_ds = SSLDinoDataset(train_frames, dist, ndim=cfg.get("ndim", 2))
    val_ds = SSLDinoDataset(val_frames, dist, ndim=cfg.get("ndim", 2))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ssl_dino
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ssl_dino
    )

    opt = AdamW(model.parameters(), lr=lr, weight_decay=ssl_cfg.get("weight_decay", 0.01))
    best_val = float("inf")

    run_name = cfg.get("name", "ssl_dino_pretrain")
    wandb.init(project="trackastra-ssl-dino", name=run_name, config=cfg)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_losses, train_cons = [], []

        for batch in train_loader:
            if batch is None:
                continue
            c1 = batch["coords1"].to(device)
            c2 = batch["coords2"].to(device)
            p1 = batch["patches1"].to(device)
            p2 = batch["patches2"].to(device)
            pm1 = batch["pm1"].to(device)
            pm2 = batch["pm2"].to(device)

            opt.zero_grad()
            z1 = model.encode(c1, patches=p1, padding_mask=pm1)
            z2 = model.encode(c2, patches=p2, padding_mask=pm2)
            loss = nt_xent_loss(z1, z2, pm1, pm2, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())
            train_cons.append(embedding_consistency(z1.detach(), z2.detach(), pm1, pm2))

        model.eval()
        val_losses, val_cons = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                c1 = batch["coords1"].to(device)
                c2 = batch["coords2"].to(device)
                p1 = batch["patches1"].to(device)
                p2 = batch["patches2"].to(device)
                pm1 = batch["pm1"].to(device)
                pm2 = batch["pm2"].to(device)
                z1 = model.encode(c1, patches=p1, padding_mask=pm1)
                z2 = model.encode(c2, patches=p2, padding_mask=pm2)
                loss = nt_xent_loss(z1, z2, pm1, pm2, temperature)
                val_losses.append(loss.item())
                val_cons.append(embedding_consistency(z1, z2, pm1, pm2))

        dt = time.perf_counter() - t0
        tl, tc = np.mean(train_losses), np.mean(train_cons)
        vl, vc = np.mean(val_losses) if val_losses else 0, np.mean(val_cons) if val_cons else 0

        # Collapse monitoring
        test_batch = next(iter(val_loader))
        with torch.no_grad():
            zt = model.encode(
                test_batch["coords1"].to(device),
                patches=test_batch["patches1"].to(device),
                padding_mask=test_batch["pm1"].to(device),
            )
            ics = inter_cell_similarity(zt, test_batch["pm1"].to(device))

        logger.info(
            f"DINO-SSL Epoch {epoch:>3}: train_loss={tl:.4f} train_cons={tc:.4f} "
            f"val_loss={vl:.4f} val_cons={vc:.4f} inter_sim={ics:.4f} [{dt:.0f}s]"
        )
        wandb.log({
            "epoch": epoch,
            "train/loss": tl,
            "train/consistency": tc,
            "val/loss": vl,
            "val/consistency": vc,
            "monitor/inter_cell_sim": ics,
            "time_per_epoch_s": dt,
        })

        if ics > 0.8:
            logger.warning(f"⚠ COLLAPSE WARNING: inter_sim={ics:.4f} > 0.8")

        if vl < best_val:
            best_val = vl

    outdir = Path(cfg.get("outdir", "runs/ssl_dino_pretrain"))
    model.save(outdir)
    wandb.log({"best_val_loss": best_val})
    wandb.finish()
    logger.info(f"Saved to {outdir}")
    logger.info(f"SSL DINO pretraining done. Best val_loss={best_val:.4f}")
    return best_val


def build_cfg(args):
    return {
        "data_root": args.data_root,
        "conditions": [c.strip() for c in args.conditions.split(",")],
        "ndim": args.ndim,
        "seed": args.seed,
        "outdir": args.outdir,
        "ssl": {
            "epochs": args.ssl_epochs,
            "lr": args.ssl_lr,
            "temperature": args.ssl_temperature,
            "batch_size": args.ssl_batch_size,
        },
        "distortions": ["affine", "elastic", "jitter", "dropout", "photometric", "feature_noise"],
        "affine": {"degrees": 15, "scale": [0.85, 1.15], "shear": [0.1, 0.1]},
        "elastic": {"alpha": [10, 50], "sigma": [5, 15]},
        "jitter": {"std": [2, 8], "p_cell_jitter": 0.8},
        "dropout": {"p_drop": [0.05, 0.2]},
        "photometric": {"scale": [0.5, 2.0], "shift": [-0.1, 0.1]},
        "feature_noise": {"std": [0.02, 0.15]},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str,
                   default="/data/cat/ws/mawe985g-data/data/celltracking/vanvliet")
    p.add_argument("--conditions", type=str, default="rpsM,recA,pheA,metA,cib,trpL")
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--knn_neighbors", type=int, default=16)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--pos_embed_per_dim", type=int, default=32)
    p.add_argument("--feat_embed_per_dim", type=int, default=8)
    p.add_argument("--ssl_epochs", type=int, default=20)
    p.add_argument("--ssl_lr", type=float, default=3e-4)
    p.add_argument("--ssl_batch_size", type=int, default=4)
    p.add_argument("--ssl_temperature", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="runs/ssl_dino_pretrain")
    p.add_argument("--max_tokens", type=int, default=4096)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = build_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = TrackingTransformer(
        coord_dim=cfg["ndim"],
        feat_dim=0,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        pos_embed_per_dim=args.pos_embed_per_dim,
        feat_embed_per_dim=args.feat_embed_per_dim,
        window=args.window,
        use_dino=True,
    ).to(device)

    logger.info(
        f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
        f"use_dino=True"
    )

    train_ssl(cfg, model, device)
