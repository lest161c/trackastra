"""SSL pretraining dataset for Trackastra.

Loads single frames, applies geometric distortions to create synthetic
frame pairs with identity association labels.

Integrates with Trackastra's existing WRAugmentationPipeline and WRFeatures.
"""

import logging
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tifffile import imread

from trackastra.data.wrfeat import (
    WRFeatures,
    WRAugmentationPipeline,
    WRRandomAffine,
    WRRandomFlip,
    WRRandomOffset,
    WRRandomMovement,
    WRRandomBrightness,
    _PROPERTIES,
)

logger = logging.getLogger(__name__)


class SSLPretrainDataset(Dataset):
    """SSL pretraining: single frame → distorted frame, identity association.

    For each frame, extracts WRFeatures, applies geometric distortion via
    Trackastra's existing WRAugmentationPipeline, and generates identity
    association labels. No real tracking labels needed.
    """

    def __init__(
        self,
        root: str,
        ndim: int = 2,
        features: str = "regionprops2",
        window_size: int = 1,
        conditions: list[str] | None = None,
    ):
        self.root = Path(root)
        self.ndim = ndim
        self.features = features
        self.window_size = window_size

        # Scan for frames (same as CTCData discovery)
        self.frames = []
        if conditions is None:
            conditions = sorted(d.name for d in self.root.iterdir()
                                if d.is_dir() and not d.name.startswith("."))
        for cond in conditions:
            cond_path = self.root / cond
            if not cond_path.is_dir():
                continue
            for exp_path in sorted(cond_path.iterdir()):
                if not exp_path.is_dir():
                    continue
                img_dir = exp_path / "img"
                if not img_dir.exists():
                    continue
                masks = sorted(exp_path.glob("TRA/man_track*.tif"))
                for mask_path in masks:
                    stem = mask_path.stem.replace("man_track", "")
                    try:
                        frame_idx = int(stem)
                    except ValueError:
                        continue
                    img_path = img_dir / f"t{frame_idx:06d}.tif"
                    if img_path.exists():
                        self.frames.append((cond, exp_path.name, frame_idx,
                                            str(mask_path), str(img_path)))
        logger.info(f"SSLPretrainDataset: {len(self.frames)} frames")

        # SSL distortion pipeline (uses existing wrfeat augmentations)
        self.augment = WRAugmentationPipeline([
            WRRandomFlip(p=0.3),
            WRRandomAffine(degrees=15, scale=(0.85, 1.15), shear=(0.1, 0.1), p=0.5),
            WRRandomOffset(offset=(-3, 3), p=0.5),
            WRRandomMovement(offset=(-5, 5), p=0.3),
            WRRandomBrightness(scale=(0.5, 2.0), shift=(-0.1, 0.1), p=0.3),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        _, _, frame_idx, mask_path, img_path = self.frames[idx]

        # Load mask + image
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        # Extract WRFeatures (single frame → no time dimension)
        mask_4d = mask[np.newaxis, ...]   # (1, H, W)
        img_4d = img[np.newaxis, ...]     # (1, H, W)
        feats = WRFeatures.from_mask_img(mask_4d, img_4d, properties=self.features,
                                         t_start=frame_idx)

        if len(feats) == 0:
            # Return empty sample
            return {
                "coords": torch.zeros(0, 1 + self.ndim),
                "features": torch.zeros(0, 0),
                "assoc_matrix": torch.zeros(0, 0),
                "timepoints": torch.zeros(0, dtype=torch.long),
                "labels": torch.zeros(0, dtype=torch.long),
                "padding_mask": torch.ones(0, dtype=torch.bool),
            }

        # Source: original features
        src_coords = feats.coords.copy()
        src_feats = OrderedDict((k, v.copy()) for k, v in feats.features.items())
        src_labels = feats.labels.copy()
        src_time = feats.timepoints.copy()

        # Target: apply distortion to get synthetic "next frame"
        feats_t = self.augment(feats)
        tgt_coords = feats_t.coords
        tgt_feats = feats_t.features
        tgt_labels = feats_t.labels
        tgt_time = feats_t.timepoints + 1  # next frame so delta_cutoff mask allows dt=1

        # Identity association: match by label
        n_src, n_tgt = len(src_labels), len(tgt_labels)
        assoc = np.zeros((n_src, n_tgt), dtype=np.float32)
        label_to_tgt = {int(l): j for j, l in enumerate(tgt_labels)}
        for i, lbl in enumerate(src_labels):
            j = label_to_tgt.get(int(lbl))
            if j is not None:
                assoc[i, j] = 1.0

        # Stack features
        feat_array = np.concatenate(list(src_feats.values()), axis=-1).astype(np.float32)
        feat_array_t = np.concatenate(list(tgt_feats.values()), axis=-1).astype(np.float32)

        # Combine src + tgt into single window (as Trackastra expects)
        # Coords: (N, 1+ndim) = (time, y, x)
        coords = np.zeros((n_src + n_tgt, 1 + self.ndim), dtype=np.float32)
        coords[:n_src, 0] = src_time
        coords[:n_src, 1:] = src_coords
        coords[n_src:, 0] = tgt_time
        coords[n_src:, 1:] = tgt_coords

        features_all = np.concatenate([feat_array, feat_array_t], axis=0).astype(np.float32)
        labels_all = np.concatenate([src_labels, tgt_labels], axis=0)
        timepoints = np.concatenate([src_time, tgt_time], axis=0).astype(np.int32)

        # Full association matrix (N_src+N_tgt × N_src+N_tgt)
        full_assoc = np.zeros((n_src + n_tgt, n_src + n_tgt), dtype=np.float32)
        full_assoc[:n_src, n_src:] = assoc  # src→tgt cross quadrant

        return {
            "coords": torch.from_numpy(coords).float(),
            "features": torch.from_numpy(features_all).float(),
            "assoc_matrix": torch.from_numpy(full_assoc).float(),
            "timepoints": torch.from_numpy(timepoints).long(),
            "labels": torch.from_numpy(labels_all).long(),
            "padding_mask": torch.zeros(n_src + n_tgt, dtype=torch.bool),
        }


def collate_ssl(batch):
    """Collate SSL batch with padding, matching Trackastra's collate_sequence_padding."""
    max_n = max(b["coords"].shape[0] for b in batch)
    ndim = batch[0]["coords"].shape[-1]
    feat_dim = 0
    for b in batch:
        if b["features"].numel() > 0:
            feat_dim = b["features"].shape[-1]
            break
    B = len(batch)

    coords = torch.zeros(B, max_n, ndim)
    features = torch.zeros(B, max_n, max(feat_dim, 1))
    assoc = torch.zeros(B, max_n, max_n)
    timepoints = -torch.ones(B, max_n, dtype=torch.long)
    labels = torch.zeros(B, max_n, dtype=torch.long)
    padding_mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["coords"].shape[0]
        if n == 0:
            continue
        coords[i, :n] = b["coords"]
        features[i, :n] = b["features"]
        assoc[i, :n, :n] = b["assoc_matrix"]
        timepoints[i, :n] = b["timepoints"]
        labels[i, :n] = b["labels"]
        padding_mask[i, :n] = False

    return {
        "coords": coords,
        "features": features,
        "assoc_matrix": assoc,
        "timepoints": timepoints,
        "labels": labels,
        "padding_mask": padding_mask,
    }
