"""Distortion families for SSL contrastive learning.

Each distortion independently transforms a synthetic "view" of a single frame.
For contrastive SSL: two independent augmented views are generated from each
original frame. The model learns to produce similar embeddings for the same
cell across views while pushing apart embeddings of different cells.

Design principle: augmentations must be destructive enough to prevent
the encoder from using trivial geometric shortcuts (coordinate memorization).
"""

import numpy as np
from collections import OrderedDict
from scipy.ndimage import map_coordinates, gaussian_filter


def _transform_affine_feature(k, v, M):
    """Transform WRFeatures under affine matrix M (ndim×ndim)."""
    ndim = M.shape[-1]
    if k == "area":
        return np.linalg.det(M) * v
    elif k == "equivalent_diameter_area":
        return np.linalg.det(M) ** (1 / ndim) * v
    elif k == "inertia_tensor":
        v = v.reshape(-1, ndim, ndim)
        v = np.einsum("ijk,mk->ijm", v, M)
        v = np.einsum("ij,kjm->kim", M, v)
        return v.reshape(-1, ndim * ndim)
    elif k in ("intensity_mean", "intensity_max", "intensity_min", "border_dist"):
        return v
    else:
        return v


class AffineDistortion:
    """Global affine warp: rotation, scale, shear."""

    def __init__(self, degrees=15, scale=(0.85, 1.15), shear=(0.1, 0.1), rng=None):
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.rng = rng if rng is not None else np.random.RandomState()

    def _make_matrix(self, ndim):
        theta = self.rng.uniform(-self.degrees, self.degrees) / 180 * np.pi
        s = self.rng.uniform(*self.scale, 3)
        shy = self.rng.uniform(-self.shear[0], self.shear[0])
        shx = self.rng.uniform(-self.shear[1], self.shear[1])
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])
        S = np.diag(s)
        Sh = np.array([[1, 0, 0], [0, 1 + shx * shy, shy], [0, shx, 1]])
        M = R @ S @ Sh
        return M[-ndim:, -ndim:]

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        M = self._make_matrix(ndim)
        coords_t = coords @ M.T
        feats_t = OrderedDict(
            (k, _transform_affine_feature(k, v, M))
            for k, v in features.items() if k != "pretrained_feats"
        )
        return coords_t, feats_t


class ElasticDistortion:
    """Elastic deformation via interpolated random displacement field."""

    def __init__(self, alpha=(10, 50), sigma=(5, 15), rng=None):
        self.alpha_range = alpha
        self.sigma_range = sigma
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        alpha = self.rng.uniform(*self.alpha_range)
        sigma = self.rng.uniform(*self.sigma_range)
        grid_size = 16
        grid_axes = [np.linspace(0, 1, grid_size) for _ in range(ndim)]
        grid_pts = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, ndim)
        rand_disp = self.rng.randn(grid_size ** ndim, ndim).astype(np.float32)
        for d_i in range(ndim):
            field = rand_disp[:, d_i].reshape(*([grid_size] * ndim))
            field = gaussian_filter(field, sigma, mode="nearest")
            rand_disp[:, d_i] = field.ravel()
        rand_disp *= alpha
        coords_norm = coords.copy().astype(np.float32)
        cmin = coords_norm.min(axis=0)
        cmax = coords_norm.max(axis=0)
        cmax = np.where(cmax == cmin, cmin + 1, cmax)
        coords_norm = (coords_norm - cmin) / (cmax - cmin)
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_pts)
        _, idx = tree.query(coords_norm)
        cell_disp = rand_disp[idx]
        coords_t = coords + cell_disp.astype(np.float32)
        feats_t = OrderedDict()
        for k, v in features.items():
            if k == "pretrained_feats":
                continue
            elif k in ("area", "equivalent_diameter_area"):
                noise = self.rng.uniform(0.95, 1.05, size=v.shape).astype(np.float32)
                feats_t[k] = v * noise
            else:
                feats_t[k] = v.copy()
        return coords_t, feats_t


class JitterDistortion:
    """Per-cell independent jitter — simulates independent cell motion.

    This is the single most effective augmentation (per ASCENT ablation).
    Each cell moves independently, forcing the model to learn identity
    features beyond coordinate matching.
    """

    def __init__(self, std=(2, 8), p_cell_jitter=0.8, rng=None):
        self.std_range = std
        self.p_cell_jitter = p_cell_jitter
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        n = len(labels)
        std = self.rng.uniform(*self.std_range)
        jitter = self.rng.randn(n, ndim).astype(np.float32) * std
        mask = self.rng.rand(n) < self.p_cell_jitter
        jitter[~mask] = 0
        coords_t = coords + jitter
        feats_t = OrderedDict(
            (k, v.copy()) for k, v in features.items() if k != "pretrained_feats"
        )
        return coords_t, feats_t


class DropoutDistortion:
    """Simulate segmentation failures — drop random subset of cells.

    Dropped cells have NO counterpart in the other view.
    Teaches the model to handle false negatives / detection failures.
    """

    def __init__(self, p_drop=(0.05, 0.2), rng=None):
        self.p_drop_range = p_drop
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        n = len(labels)
        p_drop = self.rng.uniform(*self.p_drop_range)
        keep = self.rng.rand(n) > p_drop
        coords_t = coords[keep]
        feats_t = OrderedDict(
            (k, v[keep]) for k, v in features.items() if k != "pretrained_feats"
        )
        return coords_t, feats_t


class PhotometricDistortion:
    """Intensity/value shifts for intensity-based features."""

    def __init__(self, scale=(0.5, 2.0), shift=(-0.1, 0.1), rng=None):
        self.scale_range = scale
        self.shift_range = shift
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        scale = self.rng.uniform(*self.scale_range)
        shift = self.rng.uniform(*self.shift_range)
        feats_t = OrderedDict()
        for k, v in features.items():
            if k == "pretrained_feats":
                continue
            if "intensity" in k:
                feats_t[k] = v * scale + shift
            else:
                feats_t[k] = v.copy()
        return coords.copy(), feats_t


class FeatureNoise:
    """Add Gaussian noise to all features to prevent shortcut memorization.

    Without feature noise, the encoder can memorize exact feature values
    (area, inertia_tensor, etc.) to match cells across views instead of
    learning robust identity representations.
    """

    def __init__(self, std=(0.02, 0.15), rng=None):
        self.std_range = std
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        std = self.rng.uniform(*self.std_range)
        feats_t = OrderedDict()
        for k, v in features.items():
            if k == "pretrained_feats":
                continue
            noise = self.rng.randn(*v.shape).astype(np.float32) * std
            feat_std = np.std(v)
            if feat_std > 1e-6:
                noise = noise * (feat_std * 0.1)  # Scale noise to ~10% of feature std
            feats_t[k] = v + noise
        return coords.copy(), feats_t


class DistortionPipeline:
    """Apply sequence of distortions to generate two independent augmented views.

    Each call produces two views with independently sampled distortions.
    This is the core of the contrastive SSL pretext task: the encoder must
    learn to produce consistent embeddings for the same cell across two
    differently-distorted views of the same frame.
    """

    def __init__(self, distortions, seed=None):
        self.rng = np.random.RandomState(seed)
        self.distortions = distortions

    @classmethod
    def from_config(cls, config):
        rng = np.random.RandomState(config.get("seed", 42))
        distortion_names = config.get("distortions", ["jitter"])
        affine_cfg = config.get("affine", {})
        elastic_cfg = config.get("elastic", {})
        jitter_cfg = config.get("jitter", {})
        dropout_cfg = config.get("dropout", {})
        photometric_cfg = config.get("photometric", {})
        feature_noise_cfg = config.get("feature_noise", {})

        name_to_cls = {
            "affine": lambda: AffineDistortion(**affine_cfg, rng=rng),
            "elastic": lambda: ElasticDistortion(**elastic_cfg, rng=rng),
            "jitter": lambda: JitterDistortion(**jitter_cfg, rng=rng),
            "dropout": lambda: DropoutDistortion(**dropout_cfg, rng=rng),
            "photometric": lambda: PhotometricDistortion(**photometric_cfg, rng=rng),
            "feature_noise": lambda: FeatureNoise(**feature_noise_cfg, rng=rng),
        }

        distortions = []
        for name in distortion_names:
            if name not in name_to_cls:
                raise ValueError(f"Unknown distortion: {name}")
            distortions.append(name_to_cls[name]())

        return cls(distortions, seed=config.get("seed"))

    def __call__(self, coords, features, labels):
        """Generate two independent augmented views.

        Args:
            coords:   (N, ndim) — cell coordinates
            features: OrderedDict of (N, *) — regionprops features
            labels:   (N,) — cell identity labels

        Returns:
            coords1, feats1, labels1: view 1 (independently distorted)
            coords2, feats2, labels2: view 2 (independently distorted)
        """

        def _apply_view(coord, feat, lab):
            c, f, l = coord.copy(), {k: v.copy() for k, v in feat.items()}, lab.copy()
            for dist in self.distortions:
                c, f = dist(c, f, l)
                if isinstance(dist, DropoutDistortion):
                    l = l[:len(c)]
            return c, f, l

        c1, f1, l1 = _apply_view(coords, features, labels)
        c2, f2, l2 = _apply_view(coords, features, labels)

        return c1, f1, l1, c2, f2, l2
