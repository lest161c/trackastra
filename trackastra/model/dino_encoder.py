"""
DINOv2 / DINOv3 encoder for cell patch features.

Provides frozen DINO visual backbones + learned projection head.
Used by TrackingTransformer when use_dino=True.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# ─── patch extraction ──────────────────────────────────────────────────────────

PATCH_SIZE = 64
DINO_INPUT_SIZE = 224
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Model specs for DINOv2 and DINOv3
DINO_SPECS = {
    "v2": {
        "repo": "facebookresearch/dinov2",
        "model": "dinov2_vits14",
        "dim": 384,
    },
    "v3": {
        "repo": "facebookresearch/dinov3",
        "model": "dinov3_vits16",
        "dim": 384,
    },
}


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
    """Frozen DINOv2/DINOv3 ViT encoder with lazy loading.

    Args:
        version: "v2" for DINOv2 (vits14, 384D) or "v3" for DINOv3 (vits16, 384D).
    """

    _models: dict = {}

    def __init__(self, version: str = "v2"):
        super().__init__()
        if version not in DINO_SPECS:
            raise ValueError(f"Unknown DINO version '{version}'. Choose from {list(DINO_SPECS)}")
        self._version = version
        self._spec = DINO_SPECS[version]

    @classmethod
    def _load_spec(cls, spec: dict) -> nn.Module:
        key = spec["model"]
        if key not in cls._models:
            cls._models[key] = torch.hub.load(spec["repo"], spec["model"])
            cls._models[key].eval()
            for p in cls._models[key].parameters():
                p.requires_grad = False
        return cls._models[key]

    @property
    def dim(self) -> int:
        return self._spec["dim"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINO CLS token embeddings.

        Args:
            x: (N, 3, 224, 224) float32, ImageNet-normalized on any device.

        Returns:
            (N, embed_dim) float32 embeddings.
        """
        model = self._load_spec(self._spec)
        if next(model.parameters()).device != x.device:
            model = model.to(x.device)
        return model(x)


# ─── DINO projection head ─────────────────────────────────────────────────────


class DINOProjection(nn.Module):
    """Frozen DINO backbone + learned MLP projection to d_model.

    Maps: (B, N, 1, P, P) cell patches → (B, N, d_model) embeddings.
    Handles normalization, resize, DINO forward, and projection internally.

    Args:
        d_model: Output embedding dimension for the transformer.
        hidden: Hidden dimension for the MLP projection head.
        patch_size: Square patch size to crop around centroids.
        version: "v2" (DINOv2 vits14, 384D) or "v3" (DINOv3 vits16, 384D).
    """

    def __init__(
        self,
        d_model: int = 320,
        hidden: int = 256,
        patch_size: int = PATCH_SIZE,
        version: str = "v2",
    ):
        super().__init__()
        self.patch_size = patch_size
        self._version = version
        self.backbone = DINOBackbone(version=version)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.dim, hidden),
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
        x = patches.view(B * N, C, P, P)
        x = F.interpolate(x, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        mean = DINO_MEAN.to(x.device)
        std = DINO_STD.to(x.device)
        x = (x - mean) / std
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.proj(feats)
        return feats.view(B, N, -1)
