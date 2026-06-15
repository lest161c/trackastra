"""
DINOv2 encoder for cell patch features.

Provides frozen DINOv2 visual backbone + learned projection head.
Used by TrackingTransformer when use_dino=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── patch extraction ──────────────────────────────────────────────────────────

PATCH_SIZE = 64
DINO_INPUT_SIZE = 224
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_patches(
    img: np.ndarray,
    centroids: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Extract square patches around centroids from a 2D image.

    Args:
        img: (H, W) float32, normalized to [0, 1].
        centroids: (N, 2) float32, (y, x) coordinates.

    Returns:
        (N, patch_size, patch_size) float32 array.
    """
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        pt = max(0, -y1)
        pb = max(0, y2 - h)
        pl = max(0, -x1)
        pr = max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(
                crop,
                ((0, max(0, patch_size - crop.shape[0])), (0, max(0, patch_size - crop.shape[1]))),
                mode="reflect",
            )[:patch_size, :patch_size]
        patches.append(crop)
    return np.stack(patches) if patches else np.zeros((0, patch_size, patch_size), dtype=np.float32)


# ─── DINO backbone (singleton) ────────────────────────────────────────────────


class DINOBackbone(nn.Module):
    """Frozen DINOv2 ViT-S/14 encoder.

    Lazily loaded on first forward pass (to avoid hub download at import time).
    """

    _model: nn.Module | None = None

    @classmethod
    def load(cls) -> nn.Module:
        if cls._model is None:
            cls._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            cls._model.eval()
            for p in cls._model.parameters():
                p.requires_grad = False
        return cls._model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 CLS token embeddings.

        Args:
            x: (N, 3, 224, 224) float32, ImageNet-normalized.

        Returns:
            (N, 384) float32 embeddings.
        """
        model = self.load()
        return model(x)


# ─── DINO projection head ─────────────────────────────────────────────────────


class DINOProjection(nn.Module):
    """Frozen DINO backbone + learned MLP projection to d_model.

    Maps: (B, N, 1, P, P) cell patches → (B, N, d_model) embeddings.
    Handles normalization, resize, DINO forward, and projection internally.
    """

    def __init__(self, d_model: int = 320, hidden: int = 256, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.backbone = DINOBackbone()
        self.proj = nn.Sequential(
            nn.Linear(384, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Process cell patches through DINO + projection.

        Args:
            patches: (B, N, 1, P, P) float32, per-patch normalized to [0, 1].

        Returns:
            (B, N, d_model) float32 embeddings.
        """
        B, N, C, P, _ = patches.shape
        # Flatten batch+cell dims: (B*N, 1, P, P)
        x = patches.view(B * N, C, P, P)
        # Resize to DINO input size
        x = F.interpolate(x, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
        # Convert to 3 channels
        x = x.expand(-1, 3, -1, -1)
        # ImageNet normalization
        mean = DINO_MEAN.to(x.device)
        std = DINO_STD.to(x.device)
        x = (x - mean) / std
        # DINO forward (no grad, frozen)
        with torch.no_grad():
            feats = self.backbone(x)  # (B*N, 384)
        # Project
        feats = self.proj(feats)  # (B*N, d_model)
        # Reshape back
        return feats.view(B, N, -1)
