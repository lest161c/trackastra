"""Tests for seeding the wrfeat augmentation/crop RNG (SPEC 0002 / T1).

These tests protect the contract that two wrfeat feature pipelines
constructed with the same seed draw the same augmentation/crop sequence,
while ``seed=None`` keeps the historical unseeded (OS-entropy) behavior.

The strongest check is end-to-end through ``CTCData._setup_features_augs_wrfeat``
(the real construction site used by ``train.py``): build two pipelines,
apply cropper + augmenter on identical features, and compare the resulting
crop index, coords and feature arrays.
"""

from collections import OrderedDict

import numpy as np
import pytest

from trackastra.data import wrfeat
from trackastra.data.data import CTCData

pytestmark = pytest.mark.core


def _make_features():
    """Build a small deterministic WRFeatures on a dense integer grid.

    The grid (12x12 = 144 points) is dense enough that two crops at
    different corners keep strictly different point sets, so end-to-end
    equality checks are not fooled by geometric coincidence.
    """
    coords = np.stack(
        np.meshgrid(
            np.linspace(5, 120, 12).astype(int),
            np.linspace(5, 120, 12).astype(int),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2).astype(np.float32)
    labels = np.arange(1, len(coords) + 1, dtype=np.int32)
    timepoints = np.zeros(len(coords), dtype=np.int32)
    features = OrderedDict(
        [
            (
                "intensity_mean",
                np.linspace(0.1, 0.9, len(coords)).astype(np.float32)[:, None],
            ),
            (
                "equivalent_diameter_area",
                np.full((len(coords), 1), 5.0, np.float32),
            ),
        ]
    )
    return wrfeat.WRFeatures(
        coords=coords, labels=labels, timepoints=timepoints, features=features
    )


def _build_pipeline(seed):
    """Mirror the train.py threading path: call the real setup method.

    Returns the ``(augmenter, cropper)`` built by the same code that
    ``CTCData.__init__`` uses for ``augment=3`` / ``features="wrfeat"``.
    """
    _, augmenter, cropper = CTCData._setup_features_augs_wrfeat(
        None,
        ndim=2,
        features="wrfeat",
        augment=3,
        crop_size=(48, 48),
        seed=seed,
    )
    return augmenter, cropper


def _reset_global_rng():
    """Seed the global numpy RNG, mirroring what train.py does once.

    ``WRRandomCrop.__call__`` draws its center index from the global
    ``np.random``, which ``train.py`` seeds from ``--seed`` before the
    data pipeline is built; fixing it here isolates the cropper's own
    RNG as the only varying source.
    """
    np.random.seed(0)


def _crop_and_augment(features, augmenter, cropper):
    """Apply the cropper then the augmenter, returning coords and features."""
    _reset_global_rng()
    cropped_features, kept_idx = cropper(features)
    augmented_features = augmenter(cropped_features)
    return kept_idx, augmented_features


def _crop_indices(cropper, features, n_calls):
    """Return the kept index arrays of ``n_calls`` sequential crop draws."""
    kept_indices = []
    for _ in range(n_calls):
        _reset_global_rng()
        _, kept_idx = cropper(features)
        kept_indices.append(kept_idx)
    return kept_indices


def _stream_draws(rng_owner, n_draws=16):
    """Draw ``n_draws`` ints from an augmentation object's private RNG.

    Comparing raw streams is the strict, collision-free way to assert
    that two RNGs are different (or identical) regardless of how the
    draws would be interpreted geometrically.
    """
    return rng_owner._rng.randint(0, 2**31, n_draws)


def test_same_seed_produces_identical_crop_and_aug_draws():
    """Two pipelines built with the same seed are identical end-to-end.

    This is the strongest acceptance check for SPEC 0002 / T1 criterion
    (a): identical seeds -> identical augmentation/crop behavior, across
    the crop index, the augmented coords, and every feature array.
    """
    features = _make_features()
    augmenter_a, cropper_a = _build_pipeline(seed=42)
    augmenter_b, cropper_b = _build_pipeline(seed=42)

    idx_a, out_a = _crop_and_augment(features, augmenter_a, cropper_a)
    idx_b, out_b = _crop_and_augment(features, augmenter_b, cropper_b)

    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_array_equal(out_a.coords, out_b.coords)
    np.testing.assert_array_equal(out_a.labels, out_b.labels)
    assert set(out_a.features.keys()) == set(out_b.features.keys())
    for key in out_a.features:
        np.testing.assert_array_equal(out_a.features[key], out_b.features[key])


def test_different_seeds_produce_different_draws():
    """Different seeds -> different cropper and augmentation RNG streams.

    The stream-level comparison is the strict negative control proving
    that seeding is not a no-op; the 5-call crop sequence additionally
    shows the difference end-to-end on identical features.
    """
    features = _make_features()
    augmenter_a, cropper_a = _build_pipeline(seed=42)
    augmenter_b, cropper_b = _build_pipeline(seed=43)

    assert not np.array_equal(
        _stream_draws(cropper_a), _stream_draws(cropper_b)
    )
    assert not np.array_equal(
        _stream_draws(augmenter_a.augmentations[0]),
        _stream_draws(augmenter_b.augmentations[0]),
    )

    idxs_a = _crop_indices(cropper_a, features, n_calls=5)
    idxs_b = _crop_indices(cropper_b, features, n_calls=5)
    assert not all(
        np.array_equal(idx_a, idx_b) for idx_a, idx_b in zip(idxs_a, idxs_b)
    )


def test_seed_none_keeps_unseeded_behavior():
    """``seed=None`` keeps OS entropy: no crash and independent streams.

    Each pipeline gets its own unseeded ``np.random.RandomState()``, so
    the two streams are independent (state space ~2^128, collision
    essentially impossible) and the end-to-end outputs are not coupled.
    """
    features = _make_features()
    augmenter_a, cropper_a = _build_pipeline(seed=None)
    augmenter_b, cropper_b = _build_pipeline(seed=None)

    # No crash on construction or end-to-end use.
    _idx_a, _out_a = _crop_and_augment(features, augmenter_a, cropper_a)
    _idx_b, _out_b = _crop_and_augment(features, augmenter_b, cropper_b)

    assert not np.array_equal(
        _stream_draws(cropper_a), _stream_draws(cropper_b)
    )
    assert not np.array_equal(
        _stream_draws(augmenter_a.augmentations[0]),
        _stream_draws(augmenter_b.augmentations[0]),
    )
