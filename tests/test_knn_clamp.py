"""Tests for the K>N guard in TrackingTransformer (SPEC 0002 / T4).

Reproduces the cluster crash where ``dist.topk(knn + 1)`` raised
"selected index k out of range" on crops with fewer cells than
``knn + 1`` (K-sweep K in {16, 32, 64}). The historical sweep only
survived because the old cluster clone carried an uncommitted local
K>N clamp patch that was never pushed (parent commit 65e616f).

The fix gates the topk computation in ``forward()`` and ``encode()``:
when a graph has fewer cells than ``knn + 1`` the KNN index tensor is
left as ``None`` and the sparse-attention layers fall back to their
existing dense-SDPA path. For graphs with enough cells the gate is a
strict no-op (the original three-line topk block runs verbatim).
"""

import logging

import pytest
import torch

from trackastra.model import TrackingTransformer

pytestmark = pytest.mark.core


def _build_knn_model(knn_neighbors: int) -> TrackingTransformer:
    """Build a tiny sparse ``TrackingTransformer`` for K>N regression tests.

    Uses ``attn_positional_bias="none"`` and ``attn_dist_mode="none"``
    so the only behaviour under test is the KNN index gating; the same
    NoPE-style wiring is used in ``tests/test_nope.py`` for comparable
    isolation.

    Args:
        knn_neighbors: Sparse-K value passed to the model. Must be
            ``> 0`` to wire :class:`KNNMaskSparseAttention`; the
            default dense path is exercised in ``tests/test_nope.py``.

    Returns:
        A ``TrackingTransformer`` ready for ``forward()`` /
        ``encode()`` calls.
    """
    torch.manual_seed(0)
    return TrackingTransformer(
        coord_dim=2,
        feat_dim=7,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        window=4,
        pos_embed_per_dim=0,
        feat_embed_per_dim=8,
        attn_positional_bias="none",
        attn_dist_mode="none",
        knn_neighbors=knn_neighbors,
    )


def _coords_and_features(batch_size: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build well-formed (B, N, 3) coords and (B, N, 7) features.

    Args:
        batch_size: Number of graphs in the batch.
        num_nodes: Number of cells (padded) per graph.

    Returns:
        ``(coords, features)`` where ``coords[..., 0]`` holds the
        frame index and ``coords[..., 1:]`` holds distinct spatial
        positions so that ``cdist`` is non-degenerate.
    """
    torch.manual_seed(num_nodes)
    coords = torch.randn(batch_size, num_nodes, 3) * 50.0
    coords[:, :, 0] = torch.arange(num_nodes, dtype=torch.float32) % 4
    features = torch.randn(batch_size, num_nodes, 7)
    return coords, features


def test_forward_succeeds_when_graph_has_fewer_nodes_than_knn_plus_one(caplog):
    """forward() must not crash for crops with N < knn + 1.

    This is the regression for the cluster K-sweep crash: with
    ``knn_neighbors=16`` and ``num_nodes=8``, the pre-fix
    ``dist.topk(17)`` on an 8-node matrix raised
    "selected index k out of range". With the guard, knn_indices
    is left as None and the sparse attention falls back to dense
    SDPA, returning a finite association matrix.
    """
    model = _build_knn_model(knn_neighbors=16)
    model.eval()

    batch_size, knn_neighbors, num_nodes = 2, 16, 8
    coords, features = _coords_and_features(batch_size, num_nodes)
    assert num_nodes < knn_neighbors + 1

    with caplog.at_level(logging.DEBUG, logger="trackastra.model.model"):
        with torch.no_grad():
            association = model(coords, features=features)

    assert association.shape == (batch_size, num_nodes, num_nodes)
    assert torch.isfinite(association).all()
    assert any(
        "Skipping KNN index computation" in record.message
        for record in caplog.records
    ), "guard debug log must fire when the graph is too small for KNN"


def test_forward_succeeds_when_num_nodes_equals_knn_exactly(caplog):
    """forward() must not crash at the N == knn boundary.

    A naive ``knn_eff = min(knn, N - 1)`` clamp would still break
    here because :class:`KNNMaskSparseAttention` and
    :class:`GatherSparseAttention` call
    ``knn_indices.expand(..., knn)`` with the *unclamped* ``knn``.
    Leaving ``knn_indices=None`` routes to the existing dense
    fallback at the attention layer instead.
    """
    model = _build_knn_model(knn_neighbors=16)
    model.eval()

    batch_size, num_nodes = 1, 16
    coords, features = _coords_and_features(batch_size, num_nodes)

    with caplog.at_level(logging.DEBUG, logger="trackastra.model.model"):
        with torch.no_grad():
            association = model(coords, features=features)

    assert association.shape == (batch_size, num_nodes, num_nodes)
    assert torch.isfinite(association).all()
    assert any(
        "Skipping KNN index computation" in record.message
        for record in caplog.records
    )


def test_forward_unchanged_when_graph_is_large_enough_for_knn(caplog):
    """The common (N >= knn + 1) path must remain a strict no-op.

    The gate only adds an ``if num_nodes >= knn + 1`` branch around
    the original three-line topk block; for any graph that already
    had enough cells, the exact same topk call still runs. This test
    asserts that the guard's debug log does NOT fire (proving the
    gate took the original branch) and that the model still produces
    a finite association matrix.
    """
    model = _build_knn_model(knn_neighbors=16)
    model.eval()

    batch_size, knn_neighbors, num_nodes = 1, 16, 64
    coords, features = _coords_and_features(batch_size, num_nodes)
    assert num_nodes >= knn_neighbors + 1

    with caplog.at_level(logging.DEBUG, logger="trackastra.model.model"):
        with torch.no_grad():
            association = model(coords, features=features)

    assert association.shape == (batch_size, num_nodes, num_nodes)
    assert torch.isfinite(association).all()
    assert not any(
        "Skipping KNN index computation" in record.message
        for record in caplog.records
    ), "guard must NOT fire when the graph has enough cells for KNN"


def test_forward_unchanged_for_dense_knn_neighbors_minus_one():
    """The dense path (knn_neighbors=-1) must be untouched.

    The guard lives inside ``if knn > 0 and ...``, so a negative
    ``knn_neighbors`` never enters the branch — confirming the
    pre-existing dense behaviour (e.g. NoPE baseline) is unchanged.
    """
    model = _build_knn_model(knn_neighbors=-1)
    model.eval()

    batch_size, num_nodes = 2, 12
    coords, features = _coords_and_features(batch_size, num_nodes)

    with torch.no_grad():
        association = model(coords, features=features)

    assert association.shape == (batch_size, num_nodes, num_nodes)
    assert torch.isfinite(association).all()


def test_encode_succeeds_when_graph_has_fewer_nodes_than_knn_plus_one():
    """encode() must not crash either; SSL crops hit the same topk.

    ``encode()`` carries the identical topk call (line ~614 pre-fix)
    and is exercised by the ASCENT-style SSL pretraining path. The
    second site of the guard is tested here to keep both crash sites
    covered by a single regression test.
    """
    model = _build_knn_model(knn_neighbors=16)
    model.eval()

    batch_size, num_nodes = 2, 8
    coords, features = _coords_and_features(batch_size, num_nodes)

    with torch.no_grad():
        embeddings = model.encode(coords, features=features)

    assert embeddings.shape == (batch_size, num_nodes, model.config["d_model"])
    assert torch.isfinite(embeddings).all()
