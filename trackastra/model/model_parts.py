"""Transformer class."""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryPositionalEncoding

logger = logging.getLogger(__name__)


def _pos_embed_fourier1d_init(
    cutoff: float = 256, n: int = 32, cutoff_start: float = 1
):
    return (
        torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n))
        .unsqueeze(0)
        .unsqueeze(0)
    )


class FeedForward(nn.Module):
    def __init__(self, d_model, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        cutoffs: tuple[float] = (256,),
        n_pos: tuple[int] = (32,),
        cutoffs_start=None,
    ):
        """Positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.
        """
        super().__init__()
        if cutoffs_start is None:
            cutoffs_start = (1,) * len(cutoffs)

        assert len(cutoffs) == len(n_pos)
        self.freqs = nn.ParameterList([
            nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
            for cutoff, n, cutoff_start in zip(cutoffs, n_pos, cutoffs_start)
        ])

    def forward(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        embed = torch.cat(
            tuple(
                torch.cat(
                    (
                        torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq),
                        torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq),
                    ),
                    axis=-1,
                )
                / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return embed


class NoPositionalEncoding(nn.Module):
    def __init__(self, d):
        """One learnable input token that ignores positional information."""
        super().__init__()
        self.d = d
        # self.token = nn.Parameter(torch.randn(d))

    def forward(self, coords: torch.Tensor):
        B, N, _ = coords.shape
        return (
            # torch.ones((B, N, self.d), device=coords.device) * 0.1
            # torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.01
            torch.randn((B, N, self.d), device=coords.device) * 0.01
            + torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.1
        )
        # return self.token.view(1, 1, -1).expand(B, N, -1)


def _bin_init_exp(cutoff: float, n: int):
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _bin_init_linear(cutoff: float, n: int):
    return torch.linspace(-cutoff, cutoff, n)


class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor, knn_indices: torch.Tensor = None):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        
        if knn_indices is not None:
            B_idx = torch.arange(_B, device=coords.device).view(_B, 1, 1)
            t_knn = t[B_idx, knn_indices]
            yx_knn = yx[B_idx, knn_indices, :]
            
            temporal_dist = t.unsqueeze(-1) - t_knn
            spatial_dist = torch.norm(yx.unsqueeze(-2) - yx_knn, dim=-1)
        else:
            temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
            spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        
        if knn_indices is not None:
            # -> B, nH, N, K
            bias = bias.permute(0, 3, 1, 2)
        else:
            # -> B, nH, N, N
            bias = bias.transpose(-1, 1)
            
        return bias


class RelativePositionalAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
        attn_dist_mode: str = "v0",
        use_knn_attention: bool = False,
        knn_neighbors: int = 12,
    ):
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self._mode = mode
        self.attn_dist_mode = attn_dist_mode
        self.cutoff_spatial = cutoff_spatial

        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.use_knn_attention = use_knn_attention
        self.knn_neighbors = knn_neighbors

        if mode == "bias" or mode is True:
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning(
                "attn_positional_bias is not set (None or False), no positional bias."
            )
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        knn_indices: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        if coords is not None and self._mode == "rope":
            q, k = self.rot_pos_enc(q, k, coords)

        if self.use_knn_attention and knn_indices is not None:
            B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
            H_idx = torch.arange(self.n_head, device=q.device).view(1, self.n_head, 1, 1)
            idx_exp = knn_indices.unsqueeze(1).expand(B, self.n_head, N, self.knn_neighbors)

            k_knn = k[B_idx, H_idx, idx_exp, :]
            v_knn = v[B_idx, H_idx, idx_exp, :]

            q_reshaped = q.transpose(1, 2).reshape(B * N, self.n_head, 1, q.shape[-1])
            k_reshaped = k_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, k.shape[-1])
            v_reshaped = v_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, v.shape[-1])

            # Prepare attention mask for KNN
            attn_mask_knn = torch.zeros((B, self.n_head, N, self.knn_neighbors), device=query.device, dtype=q.dtype)
            
            # dont add positional bias to self-attention if coords is None
            if coords is not None:
                if self._mode == "bias":
                    attn_mask_knn = attn_mask_knn + self.pos_bias(coords, knn_indices=knn_indices)

                # spatial cutoff
                yx = coords[..., 1:]
                yx_knn = yx[B_idx.view(B, 1, 1), knn_indices, :]
                spatial_dist_knn = torch.norm(yx.unsqueeze(-2) - yx_knn, dim=-1)
                
                spatial_mask_knn = (spatial_dist_knn > self.cutoff_spatial).unsqueeze(1)
                attn_mask_knn.masked_fill_(spatial_mask_knn, attn_ignore_val)
                
                if self.attn_dist_mode == "v0":
                    t = coords[..., 0:1]
                    t_knn = t[B_idx.view(B, 1, 1), knn_indices, :]
                    coords_knn = torch.cat([t_knn, yx_knn], dim=-1)
                    dist_knn = torch.norm(coords.unsqueeze(-2) - coords_knn, dim=-1)
                    attn_mask_knn += torch.exp(-0.1 * dist_knn.unsqueeze(1))
                elif self.attn_dist_mode == "v1":
                    attn_mask_knn += torch.exp(-5 * spatial_dist_knn.unsqueeze(1) / self.cutoff_spatial)
                else:
                    raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

            # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
            if padding_mask is not None:
                ignore_mask_knn = padding_mask[B_idx.view(B, 1, 1), knn_indices]
                ignore_mask_knn = ignore_mask_knn.unsqueeze(1)
                attn_mask_knn.masked_fill_(ignore_mask_knn, attn_ignore_val)

            attn_mask_knn = attn_mask_knn.permute(0, 2, 1, 3).reshape(B * N, self.n_head, 1, self.knn_neighbors)

            # self.attn_mask = attn_mask.clone()

            y_reshaped = F.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped, attn_mask=attn_mask_knn, dropout_p=self.dropout if self.training else 0
            )

            y = y_reshaped.view(B, N, self.n_head, -1).transpose(1, 2)
        else:
            attn_mask = torch.zeros(
                (B, self.n_head, N, N), device=query.device, dtype=q.dtype
            )

            # spatial cutoff
            yx = coords[..., 1:]
            spatial_dist = torch.cdist(yx, yx)
            spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
            attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

            # dont add positional bias to self-attention if coords is None
            if coords is not None:
                if self._mode == "bias":
                    attn_mask = attn_mask + self.pos_bias(coords)
                elif self._mode == "rope":
                    pass # already applied before if block
                else:
                    pass

                if self.attn_dist_mode == "v0":
                    dist = torch.cdist(coords, coords, p=2)
                    attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))
                elif self.attn_dist_mode == "v1":
                    attn_mask += torch.exp(
                        -5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial
                    )
                else:
                    raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

            # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
            if padding_mask is not None:
                ignore_mask = torch.logical_or(
                    padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
                ).unsqueeze(1)
                attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

            # self.attn_mask = attn_mask.clone()

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
            )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)

        return y
