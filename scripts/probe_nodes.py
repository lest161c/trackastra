#!/usr/bin/env python3
"""Probe frozen encoder node embeddings for tracking-relevant attributes.

Usage:
    python probe_nodes.py \\
        --checkpoint /path/to/model/folder \\
        --config configs/vanvliet_baseline_cnn.yaml \\
        --output results/probe_results.json

    # Dry run (data + model loading check, no probe training):
    python probe_nodes.py \\
        --checkpoint /path/to/model/folder \\
        --config configs/vanvliet_baseline_cnn.yaml \\
        --dry

    # Specify model checkpoint filename (default: model.pt):
    python probe_nodes.py \\
        --checkpoint /path/to/model/folder \\
        --checkpoint-filename model.pt \\
        --config configs/vanvliet_baseline_cnn.yaml

This script:
    1. Loads a trained (frozen) Trackastra model checkpoint
    2. Loads data using the same config system as train.py
    3. Runs the encoder to extract node embeddings for all cells
    4. Trains linear probes for each attribute (division, displacement, etc.)
    5. Prints a results table and saves to JSON
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import configargparse
import numpy as np
import torch

from trackastra.data import CTCData, collate_sequence_padding
from trackastra.data.distributed import BalancedDataModule
from trackastra.model import TrackingTransformer
from trackastra.model.node_probing import (
    ALL_PROBE_ATTRIBUTES,
    PROBE_DISPLAY_NAMES,
    evaluate_edge_probe,
    evaluate_node_probe,
    format_results_table,
    train_edge_probe,
    train_node_probe,
)
from trackastra.utils import none_or_str, seed, str2bool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("probe_nodes")


def load_model_from_checkpoint(
    checkpoint_path: str,
    checkpoint_filename: str = "model.pt",
    device: torch.device = torch.device("cpu"),
) -> TrackingTransformer:
    """Load a trained Trackastra model from a checkpoint folder.

    Args:
        checkpoint_path: Path to the model folder (containing config.yaml and model.pt).
        checkpoint_filename: Name of the checkpoint file within the folder.
        device: Device to load the model onto.

    Returns:
        Loaded TrackingTransformer in eval mode.
    """
    fpath = Path(checkpoint_path)
    if not fpath.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {fpath}")

    if fpath.is_file():
        # Path points directly to a model file
        model = TrackingTransformer.from_folder(
            fpath.parent,
            checkpoint_path=str(fpath.name),
        )
    else:
        model = TrackingTransformer.from_folder(
            fpath,
            checkpoint_path=checkpoint_filename,
        )

    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded model from {checkpoint_path} ({n_params / 1e6:.1f}M params, frozen)"
    )

    # Log model config
    use_cnn = model.config.get("use_cnn", False)
    cnn_checkpoint = model.config.get("cnn_checkpoint", None)
    cnn_trainable = model.config.get("cnn_trainable", False)
    logger.info(
        f"  Model config: d_model={model.config.get('d_model', '?')}, "
        f"use_cnn={use_cnn}, "
        f"cnn_trainable={cnn_trainable}"
    )

    return model


def create_data_loaders(args, device):
    """Create train and validation DataLoaders from args.

    Follows the same pattern as train.py's data loading logic.
    """
    dataset_kwargs = dict(
        ndim=args.ndim,
        detection_folders=args.detection_folders,
        window_size=args.window,
        max_tokens=args.max_tokens,
        features=args.features,
        downscale_temporal=args.downscale_temporal,
        downscale_spatial=args.downscale_spatial,
        sanity_dist=args.sanity_dist,
        crop_size=args.crop_size,
        compress=args.compress,
        use_gt=args.use_gt,
        slice_pct=(0.0, args.train_fraction),
        use_cnn=args.use_cnn,
        cnn_feat_dropout=0.0,  # No dropout during probing (deterministic)
    )
    sampler_kwargs = dict(
        batch_size=args.batch_size,
        n_pool=args.n_pool_sampler,
        num_samples=args.train_samples,
        weight_by_ndivs=args.weight_by_ndivs,
        weight_by_dataset=args.weight_by_dataset,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding,
    )

    def _expand_experiments(paths):
        expanded = []
        for p in paths:
            root = Path(p)
            if not root.exists():
                expanded.append(p)
                continue
            subdirs = sorted(
                d for d in root.iterdir()
                if d.is_dir() and (d / "TRA").exists()
            )
            expanded.extend(str(d) for d in subdirs) if subdirs else expanded.append(p)
        return expanded

    input_train = _expand_experiments(args.input_train)
    input_val = _expand_experiments(args.input_val) if args.input_val else []

    datamodule = BalancedDataModule(
        input_train=input_train,
        input_val=input_val,
        cachedir=args.cachedir,
        augment=0,  # No augmentation for probing (deterministic)
        distributed=False,
        dataset_kwargs=dataset_kwargs,
        sampler_kwargs=sampler_kwargs,
        loader_kwargs=loader_kwargs,
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    d_model = args.d_model

    logger.info(
        f"Data loaders created: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches"
    )

    return train_loader, val_loader, d_model


def parse_probe_args():
    """Parse command-line arguments for probe_nodes.py.

    Follows the same style as train.py's parse_train_args().
    """
    parser = configargparse.ArgumentParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
        description=__doc__,
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path (YAML)",
    )

    # Checkpoint loading
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model folder (containing config.yaml and model.pt)",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default="model.pt",
        help="Checkpoint filename within the model folder (default: model.pt)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: print only)",
    )

    # Probe configuration
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
        help="Number of training epochs for each probe",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for probe training",
    )
    parser.add_argument(
        "--probe-attributes",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_PROBE_ATTRIBUTES + ["edge_existence"],
        help="Specific attributes to probe (default: all)",
    )

    # Data arguments (same as train.py)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("-d", "--d_model", type=int, default=320)
    parser.add_argument("-w", "--window", type=int, default=4)
    parser.add_argument(
        "--detection_folders",
        type=str,
        nargs="+",
        default=["TRA"],
    )
    parser.add_argument("--input_train", type=str, nargs="+")
    parser.add_argument("--input_val", type=str, nargs="*")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--downscale_temporal", type=int, default=1)
    parser.add_argument("--downscale_spatial", type=int, default=1)
    parser.add_argument("--spatial_pos_cutoff", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--pos_embed_per_dim", type=int, default=32)
    parser.add_argument("--feat_embed_per_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--delta_cutoff", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--attn_positional_bias",
        type=str,
        choices=["rope", "bias", "none"],
        default="rope",
    )
    parser.add_argument("--attn_positional_bias_n_spatial", type=int, default=16)
    parser.add_argument("--attn_dist_mode", default="v1")
    parser.add_argument("--knn_neighbors", type=int, default=-1)
    parser.add_argument("--mixedp", type=str2bool, default=True)
    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "none",
            "regionprops",
            "regionprops2",
            "patch",
            "patch_regionprops",
            "wrfeat",
        ],
        default="wrfeat",
    )
    parser.add_argument(
        "--causal_norm",
        type=str,
        choices=["none", "linear", "softmax", "quiet_softmax"],
        default="quiet_softmax",
    )
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--sanity_dist", action="store_true")
    parser.add_argument(
        "--compress", type=str2bool, default=True, help="compress dataset"
    )
    parser.add_argument(
        "--use_gt",
        type=str2bool,
        default=True,
        help="use ground truth data",
    )
    parser.add_argument(
        "--cachedir",
        type=none_or_str,
        default=".cache",
        help="cache dir for CTCData. Set to `None` to disable caching.",
    )
    parser.add_argument(
        "--n_pool_sampler",
        type=int,
        default=8,
        help="pool size for balanced sampler",
    )
    parser.add_argument(
        "--weight_by_ndivs",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--weight_by_dataset",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=None,
        help="random crop size for augmentation",
    )

    # CNN feature injection (for data loading consistency)
    parser.add_argument(
        "--use_cnn",
        type=str2bool,
        default=False,
        help="Load CNN patches for cell tokens",
    )
    parser.add_argument(
        "--cnn_checkpoint",
        type=str,
        default=None,
        help="Path to frozen ScaledCNN checkpoint",
    )
    parser.add_argument(
        "--cnn_feat_dropout",
        type=float,
        default=0.0,
        help="CNN feature dropout (disabled for probing)",
    )
    parser.add_argument(
        "--cnn_trainable",
        type=str2bool,
        default=False,
        help="CNN trainable flag (ignored for probing)",
    )

    # Utility flags
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Dry run: load model and data but don't train probes",
    )
    parser.add_argument("--verbose", type=str2bool, default=True)

    args, unknown_args = parser.parse_known_args()

    # If checkpoint is provided, try to inherit data args from model config
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_file():
            config_path = ckpt_path.parent / "config.yaml"
        else:
            config_path = ckpt_path / "config.yaml"

        if config_path.exists():
            import yaml
            model_config = yaml.safe_load(open(config_path))
            # Inherit architecture-relevant settings
            if args.d_model is None and "d_model" in model_config:
                args.d_model = model_config["d_model"]
            if "use_cnn" in model_config:
                args.use_cnn = model_config["use_cnn"]
            if "cnn_checkpoint" in model_config:
                args.cnn_checkpoint = model_config["cnn_checkpoint"]

    return args


def main():
    args = parse_probe_args()
    seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # 1. Load model
    logger.info("Loading model...")
    model = load_model_from_checkpoint(
        args.checkpoint,
        checkpoint_filename=args.checkpoint_filename,
        device=device,
    )

    # 2. Determine d_model from model config (most reliable)
    d_model = model.config.get("d_model", args.d_model)

    # 3. Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, _ = create_data_loaders(args, device)

    logger.info(f"  Train loader: {len(train_loader)} batches")
    logger.info(f"  Val loader:   {len(val_loader)} batches")

    if args.dry:
        logger.info("Dry run complete. Model and data loaded successfully.")
        # Print a few stats about the data
        for name, loader in [("train", train_loader), ("val", val_loader)]:
            n_cells = 0
            n_windows = 0
            for batch in loader:
                n_windows += batch["coords"].shape[0]
                n_cells += (~batch["padding_mask"]).sum().item()
            logger.info(f"  {name}: {n_windows} windows, ~{n_cells} cells")
        logger.info("Exiting (dry run).")
        return

    # 4. Determine which probes to run
    probe_attributes = args.probe_attributes or (ALL_PROBE_ATTRIBUTES + ["edge_existence"])
    logger.info(f"Probe attributes: {probe_attributes}")
    logger.info(f"Probe epochs: {args.probe_epochs}, lr: {args.probe_lr}")

    # 5. Run probes
    results = {}
    t_start = time.time()

    for attr in probe_attributes:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running probe: {PROBE_DISPLAY_NAMES.get(attr, attr)}")
        logger.info(f"{'=' * 70}")

        if attr == "edge_existence":
            edge_probe, train_metrics = train_edge_probe(
                model, train_loader, d_model, device,
                epochs=args.probe_epochs, lr=args.probe_lr,
                verbose=args.verbose,
            )
            val_metrics = evaluate_edge_probe(
                edge_probe, model, val_loader, device,
                verbose=args.verbose,
            )
        else:
            probe, train_metrics, _, _ = train_node_probe(
                model, train_loader, attr, d_model, device,
                epochs=args.probe_epochs, lr=args.probe_lr,
                verbose=args.verbose,
            )
            val_metrics = evaluate_node_probe(
                probe, model, val_loader, attr, device,
                verbose=args.verbose,
            )

        results[attr] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "display_name": PROBE_DISPLAY_NAMES.get(attr, attr),
        }

    elapsed = time.time() - t_start

    # 6. Print results table
    print()
    results_table = format_results_table(results)
    print(results_table)
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # 7. Save to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy/torch values to native Python types for JSON
        def _convert(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            elif isinstance(v, (np.floating,)):
                return float(v)
            elif isinstance(v, (np.ndarray,)):
                return v.tolist()
            elif isinstance(v, (torch.Tensor,)):
                return v.item() if v.numel() == 1 else v.tolist()
            return v

        serializable = {}
        for attr, r in results.items():
            serializable[attr] = {
                "display_name": r["display_name"],
                "train_metrics": {k: _convert(v) for k, v in r["train_metrics"].items()},
                "val_metrics": {k: _convert(v) for k, v in r["val_metrics"].items()},
            }

        output_data = {
            "checkpoint": args.checkpoint,
            "config": vars(args),
            "results": serializable,
            "elapsed_minutes": elapsed / 60.0,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
