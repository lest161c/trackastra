"""ScaledCNN encoder for cell patch feature extraction.

Provides the ScaledCNN architecture used in NT-Xent pretraining
and a factory function to load a frozen checkpoint.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PATCH_SIZE = 64
OUT_DIM = 128


class ScaledCNN(nn.Module):
    """ConvNet for 64×64 grayscale cell patches. Output: 128-dim embedding.

    Args:
        scale: 'small' (~20K params), 'medium' (~100K params), 'large' (~1.47M params)
        out_dim: embedding dimension (default 128)
    """
    def __init__(self, scale='large', out_dim=OUT_DIM):
        super().__init__()
        if scale == 'small':
            ch = [8, 16, 32]           # 3 conv layers
        elif scale == 'medium':
            ch = [16, 32, 64]          # 3 conv layers
        else:  # large
            ch = [32, 64, 128, 256]    # 4 conv layers

        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        # After pooling: large: 64 → 32 → 16 → 8 → 4
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, patches):
        """patches: (N, 1, 64, 64) → embeddings: (N, out_dim)"""
        return self.fc(self.conv(patches))


def load_cnn_checkpoint(
    checkpoint_path: str,
    scale: str = 'large',
    map_location=None,
) -> nn.Module:
    """Load a frozen ScaledCNN from an NT-Xent pretrained checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        scale: CNN scale ('small', 'medium', 'large').
        map_location: torch device map (default: 'cpu').

    Returns:
        ScaledCNN in eval mode with frozen parameters.
    """
    if map_location is None:
        map_location = 'cpu'

    cnn = ScaledCNN(scale=scale, out_dim=OUT_DIM)
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    # Checkpoint may be a full dict with 'model_state_dict' key
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    cnn.load_state_dict(state_dict)
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in cnn.parameters())
    logger.info(
        f"Loaded frozen ScaledCNN ({scale}) from {checkpoint_path}: "
        f"{n_params:,} params, output dim {OUT_DIM}"
    )
    return cnn
