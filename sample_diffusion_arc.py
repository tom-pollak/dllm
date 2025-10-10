#!/usr/bin/env python3
"""Sample from a pretrained diffusion transformer on ARC validation set."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from dllm import (
    ARCTaskDataset,
    DiffusionTransformer,
    DiffusionTransformerConfig,
    arc_collate,
    build_diffusion_schedule,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("data_dir", type=str, help="Path to ARC dataset root directory")
    parser.add_argument("--output-dir", type=str, default="outputs/samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--sampling-steps", type=int, default=None, help="Number of denoising steps (defaults to timesteps)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale for sampling")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to sample (None = all)")
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=288)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--dim-feedforward", type=int, default=1152)
    parser.add_argument("--time-embed-dim", type=int, default=512)
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--save-samples", action="store_true", help="Save generated samples to disk")
    parser.add_argument("--visualize", action="store_true", help="Print text visualization of samples")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def decode_tokens(embeddings: torch.Tensor, token_embed: torch.nn.Embedding) -> torch.Tensor:
    """Convert embeddings back to token IDs."""
    weight = token_embed.weight
    logits = torch.einsum("bld,vd->blv", embeddings, weight)
    tokens = logits.argmax(dim=-1)
    return tokens


def compute_accuracy(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    """Compute per-token accuracy on valid (non-padded) positions."""
    matches = (pred_tokens == target_tokens).float() * target_mask
    accuracy = matches.sum() / target_mask.sum().clamp(min=1.0)
    return accuracy.item()


def visualize_grids(
    condition: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    condition_mask: torch.Tensor,
    target_mask: torch.Tensor,
    grid_size: int = 30,
    num_examples: int = 3,
) -> None:
    """Print colored block visualization of condition, target, and predicted grids."""
    # ANSI color codes for ARC color palette
    DEFAULT_ANSI = [
        "38;5;254m",  # 0: white
        "36m",  # 1: cyan
        "31m",  # 2: red
        "38;5;219m",  # 3: light pink
        "33m",  # 4: yellow
        "38;5;87m",  # 5: light turquoise
        "32m",  # 6: green
        "38;5;106m",  # 7: olive green
        "34m",  # 8: blue
        "38;5;208m",  # 9: orange
        "35m",  # 10: magenta (pad token)
    ]
    BLOCK = "██"

    def fmt_block(v: int) -> str:
        code = DEFAULT_ANSI[min(int(v), 10)]
        return f"\033[{code}{BLOCK}\033[0m"

    batch_size = condition.size(0)
    num_to_show = min(num_examples, batch_size)

    for idx in range(num_to_show):
        print(f"\n{'='*60}")
        print(f"Example {idx + 1}:")
        print(f"{'='*60}")

        cond_grid = condition[idx].view(grid_size, grid_size).cpu().numpy()
        targ_grid = target[idx].view(grid_size, grid_size).cpu().numpy()
        pred_grid = prediction[idx].view(grid_size, grid_size).cpu().numpy()
        cond_mask = condition_mask[idx].view(grid_size, grid_size).cpu().numpy()
        targ_mask = target_mask[idx].view(grid_size, grid_size).cpu().numpy()

        # Find actual grid boundaries
        cond_h, cond_w = 0, 0
        targ_h, targ_w = 0, 0
        for i in range(grid_size):
            if cond_mask[i, :].sum() > 0:
                cond_h = i + 1
            if targ_mask[i, :].sum() > 0:
                targ_h = i + 1
        for j in range(grid_size):
            if cond_mask[:, j].sum() > 0:
                cond_w = j + 1
            if targ_mask[:, j].sum() > 0:
                targ_w = j + 1

        print("\nInput  →  Target  →  Prediction")
        print("-" * 60)

        # Print grids side by side
        max_h = max(cond_h, targ_h)
        for i in range(max_h):
            # Input grid
            if i < cond_h:
                input_line = "".join(fmt_block(cond_grid[i, j]) for j in range(cond_w))
            else:
                input_line = ""

            # Target grid
            if i < targ_h:
                target_line = "".join(fmt_block(targ_grid[i, j]) for j in range(targ_w))
            else:
                target_line = ""

            # Prediction grid
            if i < targ_h:
                pred_line = "".join(fmt_block(pred_grid[i, j]) for j in range(targ_w))
            else:
                pred_line = ""

            print(f"{input_line}  →  {target_line}  →  {pred_line}")


@torch.no_grad()
def sample_validation_set(
    model: DiffusionTransformer,
    loader: DataLoader,
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
    sampling_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    max_batches: Optional[int] = None,
    save_dir: Optional[Path] = None,
    visualize: bool = False,
) -> Dict[str, float]:
    """Sample from the model on validation set and compute metrics."""
    model.eval()

    total_accuracy = 0.0
    total_samples = 0
    batch_accuracies = []
    all_samples = []

    num_batches = max_batches or len(loader)

    print(f"\nSampling from {num_batches} batches...")

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = to_device(batch, device)

        # Sample from the diffusion model
        sampled_embeddings = model.sample(
            batch["condition"],
            batch["condition_mask"],
            schedule,
            steps=sampling_steps,
            guidance_scale=guidance_scale,
        )

        # Decode embeddings to tokens
        pred_tokens = decode_tokens(sampled_embeddings, model.token_embed)
        target_tokens = batch["target"]

        # Compute accuracy
        batch_acc = compute_accuracy(pred_tokens, target_tokens, batch["target_mask"])
        batch_accuracies.append(batch_acc)

        batch_samples = pred_tokens.size(0)
        total_accuracy += batch_acc * batch_samples
        total_samples += batch_samples

        print(f"Batch {batch_idx + 1}/{num_batches}: accuracy={batch_acc:.4f}")

        # Visualize first batch if requested
        if visualize and batch_idx == 0:
            visualize_grids(
                batch["condition"],
                target_tokens,
                pred_tokens,
                batch["condition_mask"],
                batch["target_mask"],
                grid_size=model.config.max_grid_size,
                num_examples=3,
            )

        # Store samples if saving
        if save_dir is not None:
            all_samples.append({
                "condition": batch["condition"].cpu(),
                "condition_mask": batch["condition_mask"].cpu(),
                "target": target_tokens.cpu(),
                "target_mask": batch["target_mask"].cpu(),
                "prediction": pred_tokens.cpu(),
            })

    # Save samples to disk
    if save_dir is not None:
        save_path = save_dir / "validation_samples.pt"
        torch.save(all_samples, save_path)
        print(f"\nSaved {len(all_samples)} batches to {save_path}")

    # Compute overall metrics
    avg_accuracy = total_accuracy / max(total_samples, 1)

    return {
        "accuracy": avg_accuracy,
        "num_samples": total_samples,
        "num_batches": len(batch_accuracies),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device)

    # Create output directory if saving
    save_dir = None
    if args.save_samples:
        save_dir = Path(args.output_dir)
        os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = ARCTaskDataset(
        args.data_dir,
        split="training",
        max_grid_size=args.max_grid_size,
        augment=False,  # No augmentation for validation
    )

    # Split into train/val (same as training script)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=arc_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Validation set size: {len(val_dataset)}")

    # Create model
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    state = torch.load(args.checkpoint, map_location=device)

    if args.use_ema and "ema" in state:
        print("Using EMA weights")
        model.load_state_dict(state["ema"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        # Assume raw state dict
        model.load_state_dict(state)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    # Build diffusion schedule
    schedule = build_diffusion_schedule(args.timesteps, device=device)

    sampling_steps = args.sampling_steps or args.timesteps
    print(f"Sampling with {sampling_steps} denoising steps (guidance scale: {args.guidance_scale})")

    # Sample from validation set
    metrics = sample_validation_set(
        model,
        val_loader,
        schedule,
        device,
        sampling_steps=sampling_steps,
        guidance_scale=args.guidance_scale,
        max_batches=args.max_batches,
        save_dir=save_dir,
        visualize=args.visualize,
    )

    # Print results
    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(f"Token Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Samples:  {metrics['num_samples']}")
    print(f"Batches:        {metrics['num_batches']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
