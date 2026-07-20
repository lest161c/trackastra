"""Node-level representation probing for Trackastra.

Trains linear probes on frozen encoder node embeddings to measure
how much biological information the encoder captures.

This is standard representation probing (like linear probes in BERT)
applied to cell tracking. The encoder is FROZEN throughout — only
the probe head is trained. This measures what the encoder has ALREADY
learned, not what it can learn.

Probe attributes:
    P1. Division in next frame (binary)
    P2. Cell displacement magnitude (regression)
    P3. Cell displacement direction (multi-class, 8 octants)
    P4. Ancestor count / lineage depth (regression)
    P5. Descendant count / track span (regression)
    P6. Edge existence (binary, pairwise) — closest to HOCT edge probing

Reference:
    Bragantini et al. (2026). Higher-Order Cell Tracking Transformer (HOCT).
    HOCT §4.2: Edge probing with logistic regression on frozen edge embeddings
    achieves 59% AOGM reduction with 400 annotations.

    Our approach is adapted for node-centric architectures: we probe per-cell
    node embeddings rather than edge tokens, and for edge prediction we build
    pairwise features from pairs of node embeddings.
"""

import logging
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe heads
# ═══════════════════════════════════════════════════════════════════════════════


class NodeProbingHead(nn.Module):
    """Linear probe on frozen node embeddings for tracking-relevant attributes.

    A single linear layer (or small MLP) trained on top of frozen encoder
    node embeddings to predict per-cell attributes.

    Args:
        d_model: Embedding dimension from the encoder.
        output_dim: Number of output classes (1 for binary/regression, K for
            multi-class).
        head_type: 'linear' for single Linear layer, 'mlp' for a small 2-layer MLP.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int = 1,
        head_type: Literal["linear", "mlp"] = "linear",
    ):
        super().__init__()
        self.head_type = head_type
        if head_type == "linear":
            self.net = nn.Linear(d_model, output_dim)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, output_dim),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_embeddings: (N, D) tensor of node embeddings.

        Returns:
            (N, output_dim) tensor of predictions.
        """
        return self.net(node_embeddings)


class EdgeProbingHead(nn.Module):
    """MLP probe on paired node embeddings for edge existence prediction.

    Given two node embeddings (e_i, e_j), predict whether an association
    edge exists between them. This is the closest we can get to HOCT's
    edge probing in a node-centric architecture.

    Pairwise features: concat(e_i, e_j, e_i - e_j, e_i * e_j)
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = 4 * d_model  # concat + diff + hadamard
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _pair_features(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise features from two node embeddings.

        Returns concat(e1, e2, e1 - e2, e1 * e2).
        """
        return torch.cat([e1, e2, e1 - e2, e1 * e2], dim=-1)

    def forward(
        self, e1: torch.Tensor, e2: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            e1: (N, D) embedding of first cell in each pair.
            e2: (N, D) embedding of second cell in each pair.

        Returns:
            (N, 1) logits.
        """
        return self.net(self._pair_features(e1, e2))


# ═══════════════════════════════════════════════════════════════════════════════
#  Attribute extraction from batches
# ═══════════════════════════════════════════════════════════════════════════════


def extract_cell_attributes_from_batch(
    assoc_matrix: torch.Tensor,
    coords: torch.Tensor,
    timepoints: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    embeddings: torch.Tensor,
) -> dict:
    """Extract per-cell attributes from a single batch.

    For each non-padded cell in the batch, computes:
        - Whether it divides in the next frame
        - Its displacement to the next frame (if it persists)
        - Number of ancestor and descendant cells

    Args:
        assoc_matrix: (B, N, N) ground truth association matrix (float).
        coords: (B, N, C) coordinates [t, y, x, ...] (float).
        timepoints: (B, N) timepoint indices (long).
        labels: (B, N) cell label IDs (long).
        padding_mask: (B, N) boolean mask, True = padded token.
        embeddings: (B, N, D) encoder output (float).

    Returns:
        dict with keys:
            - embedding: list of (D,) tensors (on CPU)
            - divides_next: list of bool
            - displacement: list of (ndim,) tensors or None (on CPU)
            - n_ancestors: list of int
            - n_descendants: list of int
            - timepoint: list of int
            - label: list of int
    """
    B, N, D = embeddings.shape
    ndim = coords.shape[-1] - 1  # spatial dims (exclude time coordinate)

    results: dict = {
        "embedding": [],
        "divides_next": [],
        "displacement": [],
        "n_ancestors": [],
        "n_descendants": [],
        "timepoint": [],
        "label": [],
    }

    for b in range(B):
        valid = ~padding_mask[b]
        valid_indices = torch.where(valid)[0]

        for idx in valid_indices:
            t_i = timepoints[b, idx].item()
            label_i = labels[b, idx].item()
            coord_i = coords[b, idx, 1:]  # spatial only, shape (ndim,)
            emb_i = embeddings[b, idx]  # (D,)

            # --- Cells at next timepoint ---
            next_mask = valid & (timepoints[b] == t_i + 1)
            next_indices = torch.where(next_mask)[0]

            # Division: ≥2 connections to next-time cells
            if len(next_indices) > 0:
                connections = assoc_matrix[b, idx, next_indices]
                n_connections = connections.sum().item()
                divides_next = bool(n_connections >= 2)

                # Displacement: find matching cell at next time (same label)
                same_label_mask = labels[b, next_indices] == label_i
                if same_label_mask.any():
                    match_idx = next_indices[same_label_mask][0]
                    displacement = coords[b, match_idx, 1:] - coord_i
                    displacement = displacement.cpu()
                else:
                    displacement = None
            else:
                divides_next = False
                displacement = None

            # --- Ancestors: cells at earlier timepoints linked via assoc ---
            earlier_mask = valid & (timepoints[b] < t_i)
            if earlier_mask.any():
                n_ancestors = int(assoc_matrix[b, idx, earlier_mask].sum().item())
            else:
                n_ancestors = 0

            # --- Descendants: cells at later timepoints linked via assoc ---
            later_mask = valid & (timepoints[b] > t_i)
            if later_mask.any():
                n_descendants = int(assoc_matrix[b, idx, later_mask].sum().item())
            else:
                n_descendants = 0

            results["embedding"].append(emb_i.cpu())
            results["divides_next"].append(divides_next)
            results["displacement"].append(displacement)
            results["n_ancestors"].append(n_ancestors)
            results["n_descendants"].append(n_descendants)
            results["timepoint"].append(t_i)
            results["label"].append(label_i)

    return results


def extract_edge_pairs_from_batch(
    assoc_matrix: torch.Tensor,
    coords: torch.Tensor,
    timepoints: torch.Tensor,
    padding_mask: torch.Tensor,
    embeddings: torch.Tensor,
    max_delta: int = 2,
    max_pairs_per_batch: int = 1000,
) -> dict:
    """Extract pairwise (edge) data from a batch for edge existence probing.

    Samples pairs of cells (i, j) with t_j - t_i in [1, max_delta].
    Positive label: assoc_matrix[i, j] == 1 (same lineage).
    Negative label: assoc_matrix[i, j] == 0 (no direct relationship).

    Both directions (i→j and j→i) are NOT both sampled — we only sample
    forward in time (t_j > t_i) to avoid double-counting.

    Args:
        assoc_matrix: (B, N, N) ground truth association matrix.
        coords: (B, N, C) coordinates.
        timepoints: (B, N) timepoint indices.
        padding_mask: (B, N) boolean mask.
        embeddings: (B, N, D) encoder output.
        max_delta: Maximum time difference for a pair.
        max_pairs_per_batch: Maximum pairs to sample per batch (to cap
            memory for large windows).

    Returns:
        dict with keys 'embeddings_i', 'embeddings_j', 'labels'.
    """
    B, N, D = embeddings.shape
    result = {"embeddings_i": [], "embeddings_j": [], "labels": []}

    for b in range(B):
        valid = ~padding_mask[b]
        valid_indices = torch.where(valid)[0]
        n_valid = len(valid_indices)

        pairs_this_batch = 0
        for a in range(n_valid):
            if pairs_this_batch >= max_pairs_per_batch // B:
                break
            i = valid_indices[a]
            t_i = timepoints[b, i].item()

            # Only look forward in time (t_j > t_i)
            for jj in range(a + 1, n_valid):
                if pairs_this_batch >= max_pairs_per_batch // B:
                    break
                j = valid_indices[jj]
                t_j = timepoints[b, j].item()
                dt = t_j - t_i
                if dt < 1 or dt > max_delta:
                    continue

                label = assoc_matrix[b, i, j].item()
                result["embeddings_i"].append(embeddings[b, i].cpu())
                result["embeddings_j"].append(embeddings[b, j].cpu())
                result["labels"].append(label)
                pairs_this_batch += 1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset construction utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _build_probe_tensors(
    cell_data: dict,
    attribute: str,
    device: torch.device,
) -> tuple:
    """Build (X, y) tensors for a given probe attribute from collected cell data.

    Args:
        cell_data: dict from extract_cell_attributes_from_batch (lists).
        attribute: One of 'division', 'displacement_mag', 'displacement_dir',
            'n_ancestors', 'n_descendants'.
        device: Target torch device.

    Returns:
        X: (N, D) tensor of embeddings.
        y: (N,) or (N,) tensor of labels.
           For 'division': float32 (0/1).
           For 'displacement_mag': float32 (scalar).
           For 'displacement_dir': long (0-7).
           For 'n_ancestors'/'n_descendants': float32 (scalar).
    """
    if len(cell_data["embedding"]) == 0:
        empty = torch.zeros(0, device=device)
        return empty, empty

    embeddings = torch.stack(cell_data["embedding"]).to(device)

    if attribute == "division":
        y = torch.tensor(
            cell_data["divides_next"], dtype=torch.float32, device=device
        )
        return embeddings, y

    elif attribute == "displacement_mag":
        valid_mask = [d is not None for d in cell_data["displacement"]]
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        if len(valid_idx) == 0:
            return embeddings[:0], torch.zeros(0, device=device)
        X = torch.stack([cell_data["embedding"][i] for i in valid_idx]).to(device)
        y = torch.tensor(
            [
                torch.norm(cell_data["displacement"][i].float()).item()
                for i in valid_idx
            ],
            dtype=torch.float32,
            device=device,
        )
        return X, y

    elif attribute == "displacement_dir":
        valid_mask = [d is not None for d in cell_data["displacement"]]
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        if len(valid_idx) == 0:
            return embeddings[:0], torch.zeros(0, dtype=torch.long, device=device)
        X = torch.stack([cell_data["embedding"][i] for i in valid_idx]).to(device)
        displacements = torch.stack(
            [cell_data["displacement"][i].float() for i in valid_idx]
        )  # (M, ndim)
        # Compute angle and map to 8 octants
        # atan2(y, x) gives angle in [-pi, pi]
        angles = torch.atan2(displacements[:, 0], displacements[:, 1])
        octants = ((angles + torch.pi) / (torch.pi / 4)).long() % 8
        y = octants.to(device)
        return X, y

    elif attribute in ("n_ancestors", "n_descendants"):
        y = torch.tensor(
            cell_data[attribute], dtype=torch.float32, device=device
        )
        return embeddings, y

    else:
        raise ValueError(f"Unknown attribute: {attribute}")


def _build_edge_tensors(
    edge_data: dict,
    device: torch.device,
) -> tuple:
    """Build tensors for edge existence probing.

    Returns:
        X_i: (N, D) embeddings of first cells.
        X_j: (N, D) embeddings of second cells.
        y: (N,) float32 labels (0/1).
    """
    X_i = torch.stack(edge_data["embeddings_i"]).to(device)
    X_j = torch.stack(edge_data["embeddings_j"]).to(device)
    y = torch.tensor(edge_data["labels"], dtype=torch.float32, device=device)
    return X_i, X_j, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss and metrics
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_loss(pred: torch.Tensor, target: torch.Tensor, attribute: str) -> torch.Tensor:
    """Compute loss for a probe attribute."""
    if attribute in ("division",):
        # Binary classification: pred is (N, 1), target is (N,)
        return F.binary_cross_entropy_with_logits(pred.squeeze(-1), target)
    elif attribute in ("displacement_mag", "n_ancestors", "n_descendants"):
        # Regression
        return F.mse_loss(pred.squeeze(-1), target)
    elif attribute == "displacement_dir":
        # Multi-class classification: pred is (N, 8), target is (N,) long
        return F.cross_entropy(pred, target)
    else:
        raise ValueError(f"Unknown attribute: {attribute}")


def _compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, attribute: str
) -> dict:
    """Compute evaluation metrics for a probe attribute.

    Args:
        pred: Raw model output (logits for classification, values for regression).
        target: Ground truth labels.

    Returns:
        dict of metric names → values.
    """
    metrics: dict = {}

    if attribute == "division":
        pred_bin = (pred.squeeze(-1) > 0).float()
        t_np = target.cpu().numpy()
        p_np = pred_bin.cpu().numpy()
        metrics["accuracy"] = float(accuracy_score(t_np, p_np))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(t_np, p_np))
        metrics["f1"] = float(f1_score(t_np, p_np, zero_division=0))
        metrics["positive_rate"] = float(t_np.mean())

    elif attribute in ("displacement_mag", "n_ancestors", "n_descendants"):
        p_np = pred.squeeze(-1).detach().cpu().numpy()
        t_np = target.cpu().numpy()
        metrics["mse"] = float(mean_squared_error(t_np, p_np))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(t_np, p_np))
        metrics["r2"] = float(r2_score(t_np, p_np))
        metrics["target_mean"] = float(t_np.mean())
        metrics["target_std"] = float(t_np.std())

    elif attribute == "displacement_dir":
        pred_class = pred.argmax(dim=-1)
        t_np = target.cpu().numpy()
        p_np = pred_class.cpu().numpy()
        metrics["accuracy"] = float(accuracy_score(t_np, p_np))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(t_np, p_np))
        metrics["n_classes"] = 8

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  End-to-end encoder embedding + probe training
# ═══════════════════════════════════════════════════════════════════════════════


def run_encoder_and_collect(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run the frozen encoder over all batches and collect cell attributes.

    Args:
        encoder: Frozen TrackingTransformer (any nn.Module with encode() method).
        dataloader: DataLoader yielding batches with keys:
            'coords', 'features', 'timepoints', 'labels', 'padding_mask',
            'assoc_matrix', optionally 'patches_cnn'.
        device: torch device.
        verbose: Log progress.

    Returns:
        dict with keys matching extract_cell_attributes_from_batch output,
        aggregated over all batches.
    """
    encoder.eval()

    all_cell_data: dict = {
        "embedding": [],
        "divides_next": [],
        "displacement": [],
        "n_ancestors": [],
        "n_descendants": [],
        "timepoint": [],
        "label": [],
    }

    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and (batch_idx % max(1, total_batches // 10) == 0):
                logger.info(
                    f"  Encoding batch {batch_idx + 1}/{total_batches}"
                )

            # Move tensors to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Run encoder
            embeddings = encoder.encode(
                batch["coords"],
                batch["features"],
                padding_mask=batch.get("padding_mask"),
                patches_cnn=batch.get("patches_cnn"),
            )  # (B, N, D)

            # Extract per-cell attributes
            cell_data = extract_cell_attributes_from_batch(
                batch["assoc_matrix"],
                batch["coords"],
                batch["timepoints"],
                batch["labels"],
                batch["padding_mask"],
                embeddings,
            )

            for k in all_cell_data:
                all_cell_data[k].extend(cell_data[k])

    total_cells = len(all_cell_data["embedding"])
    n_divs = sum(all_cell_data["divides_next"])
    logger.info(
        f"  Collected {total_cells} cells ({n_divs} divisions, "
        f"{n_divs / max(total_cells, 1) * 100:.1f}%)"
    )

    return all_cell_data


def train_node_probe(
    encoder: nn.Module,
    dataloader: DataLoader,
    probe_attribute: str,
    d_model: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_batch_size: int = 256,
    verbose: bool = True,
) -> tuple:
    """Train a linear probe on frozen encoder node embeddings.

    This function:
        1. Runs the frozen encoder over all batches in the dataloader.
        2. Extracts per-cell attribute labels from ground truth.
        3. Trains a linear probe (NodeProbingHead) on the (embedding, label) pairs.

    Args:
        encoder: Frozen encoder with encode() method.
        dataloader: DataLoader for training data.
        probe_attribute: One of 'division', 'displacement_mag',
            'displacement_dir', 'n_ancestors', 'n_descendants'.
        d_model: Embedding dimension.
        device: torch device.
        epochs: Number of probe training epochs.
        lr: Learning rate for probe.
        weight_decay: Weight decay for probe.
        train_batch_size: Batch size for probe SGD.
        verbose: Log progress.

    Returns:
        probe_head: Trained NodeProbingHead (on CPU).
        metrics: dict of evaluation metrics on the training data.
        X: (N, D) embedding tensor (on CPU).
        y: (N,) label tensor (on CPU).
    """
    # Configure output dimension
    output_dim = 8 if probe_attribute == "displacement_dir" else 1

    # Collect embeddings and labels
    logger.info(f"Training '{probe_attribute}' probe (linear, d_model={d_model})...")
    all_cell_data = run_encoder_and_collect(encoder, dataloader, device, verbose)

    # Build tensors
    X, y = _build_probe_tensors(all_cell_data, probe_attribute, device)
    n_samples = len(y)
    logger.info(f"  Collected {n_samples} samples for '{probe_attribute}'")

    if n_samples == 0:
        logger.warning(f"No valid samples for '{probe_attribute}', returning untrained probe")
        probe = NodeProbingHead(d_model, output_dim=output_dim)
        return probe, {}, X.cpu(), y.cpu()

    # Create probe
    probe = NodeProbingHead(d_model, output_dim=output_dim, head_type="linear").to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Training loader
    if probe_attribute == "displacement_dir":
        train_y = y.long()
    else:
        train_y = y

    dataset = TensorDataset(X, train_y)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Train
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = probe(batch_X)
            loss = _compute_loss(pred, batch_y, probe_attribute)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 20 == 0 and verbose:
            logger.info(f"    Epoch {epoch:3d}: loss = {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred = probe(X)
        metrics = _compute_metrics(pred, y, probe_attribute)

    if verbose:
        logger.info(f"  '{probe_attribute}' results: {metrics}")

    return probe.cpu(), metrics, X.cpu(), y.cpu()


def evaluate_node_probe(
    probe_head: NodeProbingHead,
    encoder: nn.Module,
    dataloader: DataLoader,
    probe_attribute: str,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Evaluate a trained probe on held-out data.

    Runs the frozen encoder on the evaluation dataloader, extracts per-cell
    attributes, and computes metrics from the probe's predictions vs. ground truth.

    Args:
        probe_head: Trained NodeProbingHead.
        encoder: Frozen encoder.
        dataloader: DataLoader for evaluation data.
        probe_attribute: One of the supported attributes.
        device: torch device.
        verbose: Log progress.

    Returns:
        metrics: dict of evaluation metrics.
    """
    encoder.eval()
    probe_head.eval()

    logger.info(f"Evaluating '{probe_attribute}' probe on held-out data...")
    all_cell_data = run_encoder_and_collect(encoder, dataloader, device, verbose)

    X, y = _build_probe_tensors(all_cell_data, probe_attribute, device)
    n_samples = len(y)
    logger.info(f"  Evaluating on {n_samples} samples")

    if n_samples == 0:
        return {}

    with torch.no_grad():
        pred = probe_head(X.to(device))
        metrics = _compute_metrics(pred, y, probe_attribute)

    logger.info(f"  '{probe_attribute}' eval results: {metrics}")
    return metrics


def train_edge_probe(
    encoder: nn.Module,
    dataloader: DataLoader,
    d_model: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    max_delta: int = 2,
    max_pairs_per_batch: int = 500,
    verbose: bool = True,
) -> tuple:
    """Train a probe for edge existence prediction from paired node embeddings.

    This is the closest we can get to HOCT's edge probing (§4.2) in a
    node-centric architecture. HOCT operates on explicit edge tokens;
    we build pairwise features from pairs of node embeddings.

    Args:
        encoder: Frozen encoder.
        dataloader: DataLoader for training data.
        d_model: Embedding dimension.
        device: torch device.
        epochs: Number of probe training epochs.
        lr: Learning rate.
        max_delta: Maximum frame difference for pairs.
        max_pairs_per_batch: Max pairs sampled per batch.
        verbose: Log progress.

    Returns:
        probe: Trained EdgeProbingHead (on CPU).
        metrics: dict of evaluation metrics.
    """
    encoder.eval()
    logger.info(
        f"Training edge existence probe "
        f"(max_delta={max_delta}, max_pairs_per_batch={max_pairs_per_batch})..."
    )

    # Collect paired data
    all_data = {"embeddings_i": [], "embeddings_j": [], "labels": []}
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and (batch_idx % max(1, total_batches // 10) == 0):
                logger.info(f"  Processing batch {batch_idx + 1}/{total_batches}")

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            embeddings = encoder.encode(
                batch["coords"],
                batch["features"],
                padding_mask=batch.get("padding_mask"),
                patches_cnn=batch.get("patches_cnn"),
            )

            edge_data = extract_edge_pairs_from_batch(
                batch["assoc_matrix"],
                batch["coords"],
                batch["timepoints"],
                batch["padding_mask"],
                embeddings,
                max_delta=max_delta,
                max_pairs_per_batch=max_pairs_per_batch,
            )

            for k in all_data:
                all_data[k].extend(edge_data[k])

    # Build tensors
    X_i, X_j, y = _build_edge_tensors(all_data, device)
    n_pairs = len(y)
    n_pos = y.sum().item()
    logger.info(f"  Collected {n_pairs} edge pairs ({n_pos} positive, "
                f"{n_pairs - n_pos} negative)")

    if n_pairs == 0:
        logger.warning("No edge pairs collected, returning untrained probe")
        return EdgeProbingHead(d_model).cpu(), {}

    # Create probe
    probe = EdgeProbingHead(d_model).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    dataset = TensorDataset(X_i, X_j, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Train
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_i, batch_j, batch_y in loader:
            optimizer.zero_grad()
            pred = probe(batch_i, batch_j).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 20 == 0 and verbose:
            logger.info(f"    Epoch {epoch:3d}: loss = {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred = probe(X_i, X_j).squeeze(-1)
        pred_bin = (pred > 0).float()
        t_np = y.cpu().numpy()
        p_np = pred_bin.cpu().numpy()
        metrics = {
            "accuracy": float(accuracy_score(t_np, p_np)),
            "balanced_accuracy": float(balanced_accuracy_score(t_np, p_np)),
            "f1": float(f1_score(t_np, p_np, zero_division=0)),
            "positive_rate": float(t_np.mean()),
            "n_pairs": n_pairs,
        }

    logger.info(f"  Edge probe results: {metrics}")
    return probe.cpu(), metrics


def evaluate_edge_probe(
    probe_head: EdgeProbingHead,
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_delta: int = 2,
    max_pairs_per_batch: int = 500,
    verbose: bool = True,
) -> dict:
    """Evaluate a trained edge probe on held-out data.

    Args:
        probe_head: Trained EdgeProbingHead.
        encoder: Frozen encoder.
        dataloader: DataLoader for evaluation data.
        device: torch device.
        max_delta: Maximum frame difference for pairs.
        max_pairs_per_batch: Max pairs sampled per batch.
        verbose: Log progress.

    Returns:
        metrics: dict of evaluation metrics.
    """
    encoder.eval()
    probe_head.eval()
    logger.info("Evaluating edge probe on held-out data...")

    all_data = {"embeddings_i": [], "embeddings_j": [], "labels": []}

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            embeddings = encoder.encode(
                batch["coords"],
                batch["features"],
                padding_mask=batch.get("padding_mask"),
                patches_cnn=batch.get("patches_cnn"),
            )
            edge_data = extract_edge_pairs_from_batch(
                batch["assoc_matrix"],
                batch["coords"],
                batch["timepoints"],
                batch["padding_mask"],
                embeddings,
                max_delta=max_delta,
                max_pairs_per_batch=max_pairs_per_batch,
            )
            for k in all_data:
                all_data[k].extend(edge_data[k])

    X_i, X_j, y = _build_edge_tensors(all_data, device)
    if len(y) == 0:
        return {}

    with torch.no_grad():
        pred = probe_head(X_i, X_j).squeeze(-1)
        pred_bin = (pred > 0).float()
        t_np = y.cpu().numpy()
        p_np = pred_bin.cpu().numpy()
        metrics = {
            "accuracy": float(accuracy_score(t_np, p_np)),
            "balanced_accuracy": float(balanced_accuracy_score(t_np, p_np)),
            "f1": float(f1_score(t_np, p_np, zero_division=0)),
            "positive_rate": float(t_np.mean()),
            "n_pairs": len(y),
        }

    logger.info(f"  Edge probe eval results: {metrics}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: run all probes
# ═══════════════════════════════════════════════════════════════════════════════


ALL_PROBE_ATTRIBUTES = [
    "division",
    "displacement_mag",
    "displacement_dir",
    "n_ancestors",
    "n_descendants",
]

PROBE_DISPLAY_NAMES = {
    "division": "Division (next frame)",
    "displacement_mag": "Displacement magnitude",
    "displacement_dir": "Displacement direction (8 octants)",
    "n_ancestors": "Ancestor count",
    "n_descendants": "Descendant count",
    "edge_existence": "Edge existence (pairwise)",
}


def run_all_probes(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    d_model: int,
    device: torch.device,
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Train and evaluate all probe attributes.

    Args:
        encoder: Frozen encoder.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        d_model: Embedding dimension.
        device: torch device.
        probe_epochs: Training epochs per probe.
        probe_lr: Learning rate per probe.
        verbose: Log progress.

    Returns:
        dict mapping attribute names to dicts with 'train_metrics' and 'val_metrics'.
    """
    results: dict = {}

    # Per-cell probes
    for attr in ALL_PROBE_ATTRIBUTES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Probe: {PROBE_DISPLAY_NAMES[attr]}")
        logger.info(f"{'='*60}")

        probe, train_metrics, _, _ = train_node_probe(
            encoder, train_loader, attr, d_model, device,
            epochs=probe_epochs, lr=probe_lr, verbose=verbose,
        )

        val_metrics = evaluate_node_probe(
            probe, encoder, val_loader, attr, device, verbose=verbose,
        )

        results[attr] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "display_name": PROBE_DISPLAY_NAMES[attr],
        }

    # Edge probe
    logger.info(f"\n{'='*60}")
    logger.info(f"Probe: {PROBE_DISPLAY_NAMES['edge_existence']}")
    logger.info(f"{'='*60}")

    edge_probe, edge_train_metrics = train_edge_probe(
        encoder, train_loader, d_model, device,
        epochs=probe_epochs, lr=probe_lr, verbose=verbose,
    )

    edge_val_metrics = evaluate_edge_probe(
        edge_probe, encoder, val_loader, device, verbose=verbose,
    )

    results["edge_existence"] = {
        "train_metrics": edge_train_metrics,
        "val_metrics": edge_val_metrics,
        "display_name": PROBE_DISPLAY_NAMES["edge_existence"],
    }

    return results


def format_results_table(results: dict) -> str:
    """Format probe results as a human-readable table.

    Args:
        results: dict from run_all_probes().

    Returns:
        String table.
    """
    lines = []
    lines.append("=" * 90)
    lines.append("Node-Level Representation Probing — Results")
    lines.append("=" * 90)

    # Header
    header = (
        f"{'Probe':<35} {'Metric':<20} {'Train':<12} {'Val':<12}"
    )
    lines.append(header)
    lines.append("-" * 90)

    for attr in ALL_PROBE_ATTRIBUTES + ["edge_existence"]:
        if attr not in results:
            continue
        r = results[attr]
        name = r.get("display_name", attr)
        train_m = r.get("train_metrics", {})
        val_m = r.get("val_metrics", {})

        if attr in ("division", "edge_existence"):
            for metric in ("balanced_accuracy", "accuracy", "f1"):
                train_val = train_m.get(metric, "—")
                val_val = val_m.get(metric, "—")
                if isinstance(train_val, float):
                    lines.append(
                        f"{name:<35} {metric:<20} {train_val:<12.4f} {val_val:<12.4f}"
                    )
                else:
                    lines.append(
                        f"{name:<35} {metric:<20} {str(train_val):<12} {str(val_val):<12}"
                    )
                name = ""  # Only show name on first row

        elif attr in ("displacement_mag", "n_ancestors", "n_descendants"):
            for metric in ("r2", "rmse", "mae"):
                train_val = train_m.get(metric, "—")
                val_val = val_m.get(metric, "—")
                if isinstance(train_val, float):
                    lines.append(
                        f"{name:<35} {metric:<20} {train_val:<12.4f} {val_val:<12.4f}"
                    )
                else:
                    lines.append(
                        f"{name:<35} {metric:<20} {str(train_val):<12} {str(val_val):<12}"
                    )
                name = ""

        elif attr == "displacement_dir":
            for metric in ("balanced_accuracy", "accuracy"):
                train_val = train_m.get(metric, "—")
                val_val = val_m.get(metric, "—")
                if isinstance(train_val, float):
                    lines.append(
                        f"{name:<35} {metric:<20} {train_val:<12.4f} {val_val:<12.4f}"
                    )
                else:
                    lines.append(
                        f"{name:<35} {metric:<20} {str(train_val):<12} {str(val_val):<12}"
                    )
                name = ""

        lines.append("-" * 90)

    lines.append("=" * 90)
    return "\n".join(lines)
