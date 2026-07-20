"""Transformer class."""

import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch

# from torch_geometric.nn import GATv2Conv
import yaml
from torch import nn

# NoPositionalEncoding,
from trackastra.utils import blockwise_causal_norm

from .model_parts import (
    FeedForward,
    CachedDistAttention,
    GatherSparseAttention,
    PositionalEncoding,
)
from .cnn_encoder import ScaledCNN, load_cnn_checkpoint
from .dino_encoder import DINOProjection

logger = logging.getLogger(__name__)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        cutoff_spatial: int = 256,
        window: int = 16,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
        attention_layer: nn.Module = None,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        if attention_layer is not None:
            self.attn = attention_layer
        else:
            self.attn = RelativePositionalAttention(
                coord_dim,
                d_model,
                num_heads,
                cutoff_spatial=cutoff_spatial,
                n_spatial=positional_bias_n_spatial,
                cutoff_temporal=window,
                n_temporal=window,
                dropout=dropout,
                mode=positional_bias,
                attn_dist_mode=attn_dist_mode,
            )
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        knn_indices: torch.Tensor = None,
        dist_2d: torch.Tensor = None,
    ):
        x = self.norm1(x)

        # setting coords to None disables positional bias
        a = self.attn(
            x,
            x,
            x,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
            dist_2d=dist_2d,
            knn_indices=knn_indices,
        )

        x = x + a
        x = x + self.mlp(self.norm2(x))

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
        attention_layer: nn.Module = None,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        if attention_layer is not None:
            self.attn = attention_layer
        else:
            self.attn = RelativePositionalAttention(
                coord_dim,
                d_model,
                num_heads,
                cutoff_spatial=cutoff_spatial,
                n_spatial=positional_bias_n_spatial,
                cutoff_temporal=window,
                n_temporal=window,
                dropout=dropout,
                mode=positional_bias,
                attn_dist_mode=attn_dist_mode,
            )

        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        knn_indices: torch.Tensor = None,
        dist_2d: torch.Tensor = None,
    ):
        x = self.norm1(x)
        y = self.norm2(y)
        # cross attention
        # setting coords to None disables positional bias
        a = self.attn(
            x,
            y,
            y,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
            dist_2d=dist_2d,
            knn_indices=knn_indices,
        )

        x = x + a
        x = x + self.mlp(self.norm3(x))

        return x


# class BidirectionalRelativePositionalAttention(RelativePositionalAttention):
#     def forward(
#         self,
#         query1: torch.Tensor,
#         query2: torch.Tensor,
#         coords: torch.Tensor,
#         padding_mask: torch.Tensor = None,
#     ):
#         B, N, D = query1.size()
#         q1 = self.q_pro(query1)  # (B, N, D)
#         q2 = self.q_pro(query2)  # (B, N, D)
#         v1 = self.v_pro(query1)  # (B, N, D)
#         v2 = self.v_pro(query2)  # (B, N, D)

#         # (B, nh, N, hs)
#         q1 = q1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         v1 = v1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         q2 = q2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         v2 = v2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

#         attn_mask = torch.zeros(
#             (B, self.n_head, N, N), device=query1.device, dtype=q1.dtype
#         )

#         # add negative value but not too large to keep mixed precision loss from becoming nan
#         attn_ignore_val = -1e3

#         # spatial cutoff
#         yx = coords[..., 1:]
#         spatial_dist = torch.cdist(yx, yx)
#         spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
#         attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

#         # dont add positional bias to self-attention if coords is None
#         if coords is not None:
#             if self._mode == "bias":
#                 attn_mask = attn_mask + self.pos_bias(coords)
#             elif self._mode == "rope":
#                 q1, q2 = self.rot_pos_enc(q1, q2, coords)
#             else:
#                 pass

#             dist = torch.cdist(coords, coords, p=2)
#             attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))

#         # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
#         if padding_mask is not None:
#             ignore_mask = torch.logical_or(
#                 padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
#             ).unsqueeze(1)
#             attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

#         self.attn_mask = attn_mask.clone()

#         y1 = nn.functional.scaled_dot_product_attention(
#             q1,
#             q2,
#             v1,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout if self.training else 0,
#         )
#         y2 = nn.functional.scaled_dot_product_attention(
#             q2,
#             q1,
#             v2,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout if self.training else 0,
#         )

#         y1 = y1.transpose(1, 2).contiguous().view(B, N, D)
#         y1 = self.proj(y1)
#         y2 = y2.transpose(1, 2).contiguous().view(B, N, D)
#         y2 = self.proj(y2)
#         return y1, y2


# class BidirectionalCrossAttention(nn.Module):
#     def __init__(
#         self,
#         coord_dim: int = 2,
#         d_model=256,
#         num_heads=4,
#         dropout=0.1,
#         window: int = 16,
#         cutoff_spatial: int = 256,
#         positional_bias: Literal["bias", "rope", "none"] = "bias",
#         positional_bias_n_spatial: int = 32,
#     ):
#         super().__init__()
#         self.positional_bias = positional_bias
#         self.attn = BidirectionalRelativePositionalAttention(
#             coord_dim,
#             d_model,
#             num_heads,
#             cutoff_spatial=cutoff_spatial,
#             n_spatial=positional_bias_n_spatial,
#             cutoff_temporal=window,
#             n_temporal=window,
#             dropout=dropout,
#             mode=positional_bias,
#         )

#         self.mlp = FeedForward(d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         coords: torch.Tensor,
#         padding_mask: torch.Tensor = None,
#     ):
#         x = self.norm1(x)
#         y = self.norm1(y)

#         # cross attention
#         # setting coords to None disables positional bias
#         x2, y2 = self.attn(
#             x,
#             y,
#             coords=coords if self.positional_bias else None,
#             padding_mask=padding_mask,
#         )
#         # print(torch.norm(x2).item()/torch.norm(x).item())
#         x = x + x2
#         x = x + self.mlp(self.norm2(x))
#         y = y + y2
#         y = y + self.mlp(self.norm2(y))

#         return x, y


class TrackingTransformer(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 1,
        window: int = 6,
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: Literal["bias", "rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
        attn_dist_mode: str = "v0",
        use_dino: bool = False,
        use_cnn: bool = False,
        cnn_checkpoint: str | None = None,
        cnn_trainable: bool = False,
        knn_neighbors: int = -1,
        lambda_decay: bool = False,
    ):
        super().__init__()

        self.config = dict(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            pos_embed_per_dim=pos_embed_per_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            window=window,
            dropout=dropout,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            spatial_pos_cutoff=spatial_pos_cutoff,
            feat_embed_per_dim=feat_embed_per_dim,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
            use_dino=use_dino,
            use_cnn=use_cnn,
            cnn_checkpoint=cnn_checkpoint,
            cnn_trainable=cnn_trainable,
            knn_neighbors=knn_neighbors,
            lambda_decay=lambda_decay,
        )

        self.register_buffer("_lambda_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_lambda_total", torch.tensor(1, dtype=torch.long))

        self.proj = nn.Linear(
            (1 + coord_dim) * pos_embed_per_dim + feat_dim * feat_embed_per_dim, d_model
        )
        if use_dino:
            self.dino_proj = DINOProjection(d_model=d_model)
            pos_embed_dim = (1 + coord_dim) * pos_embed_per_dim
            self.dino_pos_proj = nn.Linear(pos_embed_dim, d_model)
        if use_cnn:
            self.cnn_proj = nn.Linear(128, d_model)
            if cnn_checkpoint is not None:
                self.cnn_encoder = load_cnn_checkpoint(cnn_checkpoint, scale='large')
                if cnn_trainable:
                    self.cnn_encoder.train()
                    for p in self.cnn_encoder.parameters():
                        p.requires_grad = True
                else:
                    # Explicitly freeze loaded checkpoint when not trainable
                    self.cnn_encoder.eval()
                    for p in self.cnn_encoder.parameters():
                        p.requires_grad = False
                    logger.info("Loaded CNN checkpoint (frozen)")
            else:
                self.cnn_encoder = ScaledCNN(scale='large', out_dim=128)
                self.cnn_encoder.eval()
                for p in self.cnn_encoder.parameters():
                    p.requires_grad = False
                logger.info("Created untrained frozen ScaledCNN (no checkpoint)")
            # When cnn_trainable=True, unfreeze CNN encoder regardless of checkpoint source
            if cnn_trainable:
                self.cnn_encoder.train()
                for p in self.cnn_encoder.parameters():
                    p.requires_grad = True
                logger.info("CNN encoder is trainable")
        self.norm = nn.LayerNorm(d_model)

        if knn_neighbors > 0:
            attn_factory = lambda: GatherSparseAttention(
                coord_dim,
                d_model,
                nhead,
                cutoff_spatial=spatial_pos_cutoff,
                cutoff_temporal=window,
                dropout=dropout,
                mode=attn_positional_bias,
                attn_dist_mode=attn_dist_mode,
                knn_neighbors=knn_neighbors,
            )
        else:
            attn_factory = lambda: CachedDistAttention(
                coord_dim,
                d_model,
                nhead,
                cutoff_spatial=spatial_pos_cutoff,
                cutoff_temporal=window,
                dropout=dropout,
                mode=attn_positional_bias,
                attn_dist_mode=attn_dist_mode,
            )

        self.encoder = nn.ModuleList([
            EncoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_pos_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
                attention_layer=attn_factory(),
            )
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_pos_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
                attention_layer=attn_factory(),
            )
            for _ in range(num_decoder_layers)
        ])

        self.head_x = FeedForward(d_model)
        self.head_y = FeedForward(d_model)

        if feat_embed_per_dim > 1:
            self.feat_embed = PositionalEncoding(
                cutoffs=(1000,) * feat_dim,
                n_pos=(feat_embed_per_dim,) * feat_dim,
                cutoffs_start=(0.01,) * feat_dim,
            )
        else:
            self.feat_embed = nn.Identity()

        self.pos_embed = PositionalEncoding(
            cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim,
            n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
        )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def set_lambda_step(self, step, total):
        """Update the λ(t) step counter for cosine-decayed CNN feature injection."""
        self._lambda_step.fill_(step)
        self._lambda_total.fill_(max(1, total))

    def _embed(self, coords, features, padding_mask, patches=None, patches_cnn=None):
        """Shared embedding logic for forward() and encode()."""
        if padding_mask is not None and padding_mask.any():
            coords = coords.clone()
            coords[padding_mask] = coords.amax(dim=(0, 1))

        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        coords = coords - min_time
        pos = self.pos_embed(coords)

        if patches is not None and self.config.get("use_dino", False):
            dino_feats = self.dino_proj(patches)
            pos_proj = self.dino_pos_proj(pos)
            features = pos_proj + dino_feats
            features = self.norm(features)
        elif features is None or features.numel() == 0:
            features = pos
        else:
            features = self.feat_embed(features)
            features = torch.cat((pos, features), axis=-1)
            features = self.proj(features)
            features = self.norm(features)

        # CNN residual feature injection (additive, after norm)
        if patches_cnn is not None and self.config.get("use_cnn", False):
            B, N = patches_cnn.shape[:2]
            cnn_in = patches_cnn.reshape(B * N, 1, 64, 64)
            cnn_trainable = self.config.get("cnn_trainable", False)
            with torch.set_grad_enabled(cnn_trainable):
                cnn_out = self.cnn_encoder(cnn_in)  # (B*N, 128)
            cnn_out = cnn_out.reshape(B, N, -1)      # (B, N, 128)
            # Scheduled mixing: λ(t) cosine decay from 1→0
            if self.config.get("lambda_decay", False) and self.training:
                progress = min(1.0, self._lambda_step.item() / self._lambda_total.item())
                lambda_t = 0.5 * (1.0 + math.cos(math.pi * progress))
                features = features + lambda_t * self.cnn_proj(cnn_out)
            else:
                features = features + self.cnn_proj(cnn_out)

        return features, coords

    def forward(self, coords, features=None, padding_mask=None, knn_indices=None, patches=None, patches_cnn=None):
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _B, _N, _D = coords.shape

        features, coords = self._embed(coords, features, padding_mask, patches, patches_cnn)

        x = features

        knn = self.config.get("knn_neighbors", -1)
        if knn > 0 and knn_indices is None and coords is not None:
            yx = coords[..., 1:].float()
            dist = torch.cdist(yx, yx)
            knn_indices = dist.topk(knn + 1, dim=-1, largest=False)[1][..., 1:]

        dist_2d = torch.cdist(coords[..., 1:].float(), coords[..., 1:].float())

        for enc in self.encoder:
            x = enc(x, coords=coords, padding_mask=padding_mask, dist_2d=dist_2d, knn_indices=knn_indices)

        y = features
        # decoder w cross attention
        for dec in self.decoder:
            y = dec(y, x, coords=coords, padding_mask=padding_mask, dist_2d=dist_2d, knn_indices=knn_indices)

        x = self.head_x(x)
        y = self.head_y(y)

        # outer product is the association matrix (logits)
        A = torch.einsum("bnd,bmd->bnm", x, y)

        return A

    def encode(self, coords, features=None, padding_mask=None, knn_indices=None, patches=None, patches_cnn=None):
        """Run encoder only, return per-cell embeddings (B,N,d_model).

        Used for ASCENT-style contrastive SSL pretraining (Han & Lu 2025 §3.2):
        encoder output serves as per-cell embedding for NT-Xent loss.
        Decoder is not used during SSL — only encoder + projection layers.
        """
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _N = coords.shape[1]

        if _N == 0:
            return torch.zeros(coords.shape[0], 0, self.config["d_model"], device=coords.device)

        features, coords = self._embed(coords, features, padding_mask, patches, patches_cnn)

        x = features

        knn = self.config.get("knn_neighbors", -1)
        if knn > 0 and knn_indices is None and coords is not None:
            yx = coords[..., 1:].float()
            dist = torch.cdist(yx, yx)
            knn_indices = dist.topk(knn + 1, dim=-1, largest=False)[1][..., 1:]

        dist_2d = torch.cdist(coords[..., 1:].float(), coords[..., 1:].float())

        for enc in self.encoder:
            x = enc(x, coords=coords, padding_mask=padding_mask, dist_2d=dist_2d, knn_indices=knn_indices)

        x = self.head_x(x)
        return x

    def normalize_output(
        self,
        A: torch.FloatTensor,
        timepoints: torch.LongTensor,
        coords: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply (parental) softmax, or elementwise sigmoid.

        Args:
            A: Tensor of shape B, N, N
            timepoints: Tensor of shape B, N
            coords: Tensor of shape B, N, (time + n_spatial)
        """
        assert A.ndim == 3
        assert timepoints.ndim == 2
        assert coords.ndim == 3
        assert coords.shape[2] == 1 + self.config["coord_dim"]

        # spatial distances
        dist = torch.cdist(coords[:, :, 1:], coords[:, :, 1:])
        invalid = dist > self.config["spatial_pos_cutoff"]
        invalid = (
            invalid | (timepoints.unsqueeze(1) == -1) | (timepoints.unsqueeze(2) == -1)
        )

        if self.config["causal_norm"] == "none":
            # Spatially distant entries are set to zero
            A = torch.sigmoid(A)
            A[invalid] = 0
        else:
            return torch.stack([
                blockwise_causal_norm(
                    _A, _t, mode=self.config["causal_norm"], mask_invalid=_m
                )
                for _A, _t, _m in zip(A, timepoints, invalid)
            ])
        return A

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self.config, open(folder / "config.yaml", "w"))
        torch.save(self.state_dict(), folder / "model.pt")

    @staticmethod
    def create(config):
        model_classes = {
            "default": TrackingTransformer,
        }
        try:
            from trackastra_pretrained_feats import TrackingTransformerwPretrainedFeats

            PRETRAINED_FEATS_INSTALLED = True
        except ImportError:
            PRETRAINED_FEATS_INSTALLED = False
        if PRETRAINED_FEATS_INSTALLED:
            model_classes["pretrained_feats"] = TrackingTransformerwPretrainedFeats

        model_type = (
            "pretrained_feats" if "pretrained_feat_dim" in config else "default"
        )
        # TODO instead we could add explicit field in config to dispatch to different model classes rather than a train arg

        if model_type == "pretrained_feats" and not PRETRAINED_FEATS_INSTALLED:
            raise ImportError(
                "Model was trained with pretrained features, but trackastra_pretrained_feats is not installed. "
                "Please install it with `pip install trackastra[etultra]`."
            )

        return model_classes[model_type](**config)

    @classmethod
    def from_folder(
        cls, folder, map_location=None, args=None, checkpoint_path: str = "model.pt"
    ):
        folder = Path(folder)
        config_path = folder / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}. Cannot load model from '{folder}'. "
                "Ensure the path is a valid model directory containing config.yaml and model.pt."
            )
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        if args:
            args = vars(args)
            for k, v in config.items():
                errors = []
                if k in args:
                    if config[k] != args[k]:
                        errors.append(
                            f"Loaded model config {k}={config[k]}, but current argument"
                            f" {k}={args[k]}."
                        )
            if errors:
                raise ValueError("\n".join(errors))
        model = cls.create(config)

        # try:
        #     # Try to load from lightning checkpoint first
        #     v_folder = sorted((folder / "tb").glob("version_*"))[version]
        #     checkpoint = sorted((v_folder / "checkpoints").glob("*epoch*.ckpt"))[0]
        #     pl_state_dict = torch.load(checkpoint, map_location=map_location)[
        #         "state_dict"
        #     ]
        #     state_dict = OrderedDict()

        #     # Hack
        #     for k, v in pl_state_dict.items():
        #         if k.startswith("model."):
        #             state_dict[k[6:]] = v
        #         else:
        #             raise ValueError(f"Unexpected key {k} in state_dict")

        #     model.load_state_dict(state_dict)
        #     logger.info(f"Loaded model from {checkpoint}")
        # except:
        #     # Default: Load manually saved model (legacy)

        fpath = folder / checkpoint_path
        logger.info(f"Loading model state from {fpath}")

        state = torch.load(fpath, map_location=map_location, weights_only=True)
        # if state is a checkpoint, we have to extract state_dict
        if "state_dict" in state:
            state = state["state_dict"]
            state = OrderedDict(
                (k[6:], v) for k, v in state.items() if k.startswith("model.")
            )
        model.load_state_dict(state)

        return model
