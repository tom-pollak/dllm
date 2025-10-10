#!/usr/bin/env python3
"""Generate batch visualizations for ARC train/validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from dllm import (
    ARCTaskDataset,
    DiffusionTransformer,
    DiffusionTransformerConfig,
    arc_collate,
    build_diffusion_schedule,
    create_batch_visualization,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the ARC dataset root containing the 'training' folder.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--examples-per-batch", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=288)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--dim-feedforward", type=int, default=1152)
    parser.add_argument("--time-embed-dim", type=int, default=512)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to a checkpoint produced by train_diffusion_arc.py.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gather_batches(loader: DataLoader, count: int) -> List[dict[str, torch.Tensor]]:
    batches: List[dict[str, torch.Tensor]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= count:
            break
    return batches


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device)

    dataset = ARCTaskDataset(
        args.data_dir,
        split="training",
        max_grid_size=args.max_grid_size,
    )
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=arc_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=arc_collate,
        num_workers=0,
    )

    config = DiffusionTransformerConfig(
        max_timesteps=args.timesteps,
        max_grid_size=args.max_grid_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        time_embed_dim=args.time_embed_dim,
    )
    model = DiffusionTransformer(config).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    schedule = build_diffusion_schedule(args.timesteps, device=device)

    train_batches = _gather_batches(train_loader, args.num_batches)
    val_batches = _gather_batches(val_loader, args.num_batches)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fig = create_batch_visualization(
        train_batches,
        model=model,
        diffusion_schedule=schedule,
        max_grid_size=args.max_grid_size,
        examples_per_batch=args.examples_per_batch,
        title="Training batches",
    )
    train_path = output_dir / "train_batches.png"
    train_fig.savefig(train_path, bbox_inches="tight")
    plt.close(train_fig)

    val_fig = create_batch_visualization(
        val_batches,
        model=model,
        diffusion_schedule=schedule,
        max_grid_size=args.max_grid_size,
        examples_per_batch=args.examples_per_batch,
        title="Validation batches",
    )
    val_path = output_dir / "val_batches.png"
    val_fig.savefig(val_path, bbox_inches="tight")
    plt.close(val_fig)

    print(f"Saved training visualization to {train_path}")
    print(f"Saved validation visualization to {val_path}")


if __name__ == "__main__":
    main()
