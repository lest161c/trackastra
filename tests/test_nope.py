"""Tests for the coordinate-free (NoPE) model configuration (SPEC 0002 / T2).

If any new path by which centroid coordinates reach the model creeps in,
the permutation-invariance test below fails.
"""

import pytest
import torch

from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import NoPositionalEncoding, PositionalEncoding

pytestmark = pytest.mark.core


def _build_nope_model() -> TrackingTransformer:
    """Build a small dense ``TrackingTransformer`` with every NoPE switch on.

    Returns:
        Model with ``pos_embed_per_dim=0`` (constant no-position embedding),
        ``attn_positional_bias="none"`` (no RoPE in attention), and
        ``attn_dist_mode="none"`` (no spatial-cutoff mask, no distance
        decay).  Together these disable every path by which coordinates
        reach the model.
    """
    torch.manual_seed(0)
    return TrackingTransformer(
        coord_dim=2,
        feat_dim=7,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        window=4,
        pos_embed_per_dim=0,
        feat_embed_per_dim=8,
        attn_positional_bias="none",
        attn_dist_mode="none",
        knn_neighbors=-1,
    )


def test_nope_model_is_invariant_to_spatial_coordinate_permutation():
    """Under NoPE, centroid coordinates must not affect the model output.

    Feeds the same 7D regionprops features twice — once with the original
    centroids, then with two different random permutations of the spatial
    centroids across nodes (time kept aligned to each node, features
    unchanged) — and asserts the association matrix is identical for all
    three inputs.  Together with the existence checks on the default
    configuration, this is the experimental guarantee that coordinates
    reach the model only through the (here disabled) positional encoding
    paths.
    """
    nope_model = _build_nope_model()
    nope_model.eval()
    assert isinstance(nope_model.pos_embed, NoPositionalEncoding)

    torch.manual_seed(0)
    batch_size = 2
    num_nodes = 16
    coords = torch.rand(batch_size, num_nodes, 3) * 100.0
    coords[:, :, 0] = torch.arange(num_nodes, dtype=torch.float32) % 4  # frame index
    features = torch.rand(batch_size, num_nodes, 7) * 10.0

    centroid_permutations = [torch.randperm(num_nodes) for _ in range(2)]

    with torch.no_grad():
        association_reference = nope_model(coords, features=features)
        assert association_reference.shape == (batch_size, num_nodes, num_nodes)
        assert torch.isfinite(association_reference).all()

        for centroid_permutation in centroid_permutations:
            permuted_coords = coords.clone()
            permuted_coords[:, :, 1:] = coords[:, :, 1:][:, centroid_permutation]
            association_permuted = nope_model(permuted_coords, features=features)
            assert torch.allclose(association_reference, association_permuted)


def test_default_model_keeps_fourier_positional_encoding():
    """The default configuration must still use Fourier ``PositionalEncoding``.

    Guards the NoPE wiring against accidentally changing the default
    (``pos_embed_per_dim=32``) path: with default arguments the model
    must instantiate ``PositionalEncoding``, not ``NoPositionalEncoding``.
    """
    torch.manual_seed(0)
    default_model = TrackingTransformer(coord_dim=2, feat_dim=7)
    assert isinstance(default_model.pos_embed, PositionalEncoding)
    assert not isinstance(default_model.pos_embed, NoPositionalEncoding)
