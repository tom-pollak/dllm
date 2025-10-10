"""Utilities for visualizing ARC batches and diffusion corruption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


# ARC uses 10 colors (0-9). Index 10 is reserved for padding/empty cells.
# The palette below mirrors the canonical ARC visualization colors.
_COLOR_PALETTE = (
    np.array(
        [
            (0, 0, 0),  # 0 - black
            (0, 119, 187),  # 1 - blue
            (221, 95, 0),  # 2 - orange
            (0, 158, 115),  # 3 - green
            (204, 204, 0),  # 4 - yellow
            (148, 0, 211),  # 5 - purple
            (255, 105, 180),  # 6 - pink
            (0, 191, 196),  # 7 - cyan
            (255, 0, 0),  # 8 - red
            (128, 128, 128),  # 9 - gray
            (255, 255, 255),  # 10 - white / padding
        ],
        dtype=np.float32,
    )
    / 255.0
)


@dataclass
class ExampleVisualization:
    """Container for a single visualization example."""

    condition: torch.Tensor
    condition_mask: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    corrupted: torch.Tensor
    timestep: int
    batch_index: int
    example_index: int


def _tokens_to_color_grid(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_grid_size: int,
    pad_token_id: int = 10,
) -> np.ndarray:
    """Convert a flattened token grid into an RGB image using the ARC palette."""

    grid_tokens = tokens.view(max_grid_size, max_grid_size).cpu().numpy()
    grid_mask = mask.view(max_grid_size, max_grid_size).cpu().numpy()
    painted = np.full_like(grid_tokens, fill_value=pad_token_id)
    painted[grid_mask > 0.5] = grid_tokens[grid_mask > 0.5]
    return _COLOR_PALETTE[painted]


def _plot_single(ax: plt.Axes, grid: np.ndarray, title: str) -> None:
    ax.imshow(grid, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _corrupt_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    timesteps: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus: torch.Tensor,
    vocab_size: int = 11,
) -> torch.Tensor:
    """Add noise to tokens in probability space and return argmax tokens.

    Args:
        tokens: Target tokens [batch_size, seq_len]
        mask: Token mask [batch_size, seq_len]
        timesteps: Diffusion timestep for each sample [batch_size]
        sqrt_alphas_cumprod: sqrt(alpha_bar_t) schedule
        sqrt_one_minus: sqrt(1 - alpha_bar_t) schedule
        vocab_size: Number of tokens in vocabulary

    Returns:
        Corrupted tokens [batch_size, seq_len] obtained by argmax over noisy logits
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    # Create one-hot encodings of original tokens [batch_size, seq_len, vocab_size]
    one_hot = torch.nn.functional.one_hot(tokens, num_classes=vocab_size).float()

    # Add Gaussian noise to the one-hot probabilities
    noise = torch.randn_like(one_hot)

    # Get noise schedule values for each sample
    sqrt_alpha = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
    sqrt_one = sqrt_one_minus[timesteps].view(-1, 1, 1)

    # Apply noise: noisy_probs = sqrt_alpha * one_hot + sqrt_one * noise
    noisy_probs = sqrt_alpha * one_hot + sqrt_one * noise

    # Get discrete tokens via argmax
    corrupted = noisy_probs.argmax(dim=-1)

    # Apply mask to keep only valid positions
    corrupted = corrupted * mask.long() + tokens * (1 - mask.long())

    return corrupted


def _collect_examples(
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    diffusion_schedule: Dict[str, torch.Tensor],
    max_examples_per_batch: int,
    max_timesteps: int,
    vocab_size: int = 11,
    device: torch.device,
) -> List[ExampleVisualization]:
    """Collect visualization examples from batches by corrupting tokens."""
    examples: List[ExampleVisualization] = []

    sqrt_alphas_cumprod = diffusion_schedule["sqrt_alphas_cumprod"]
    sqrt_one_minus = diffusion_schedule["sqrt_one_minus"]

    for batch_index, batch in enumerate(batches):
        condition = batch["condition"].to(device)
        condition_mask = batch["condition_mask"].to(device)
        target = batch["target"].to(device)
        target_mask = batch["target_mask"].to(device)

        batch_size = target.size(0)
        take = min(max_examples_per_batch, batch_size)
        if take == 0:
            continue

        # Sample random timesteps for each example
        timesteps = torch.randint(
            low=0,
            high=max_timesteps,
            size=(batch_size,),
            device=device,
        )

        # Corrupt tokens by adding noise in probability space
        corrupted_tokens = _corrupt_tokens(
            target,
            target_mask,
            timesteps,
            sqrt_alphas_cumprod,
            sqrt_one_minus,
            vocab_size=vocab_size,
        )

        for example_index in range(take):
            examples.append(
                ExampleVisualization(
                    condition=condition[example_index].cpu(),
                    condition_mask=condition_mask[example_index].cpu(),
                    target=target[example_index].cpu(),
                    target_mask=target_mask[example_index].cpu(),
                    corrupted=corrupted_tokens[example_index].cpu(),
                    timestep=int(timesteps[example_index].item()),
                    batch_index=batch_index,
                    example_index=example_index,
                )
            )
    return examples


def create_batch_visualization(
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    diffusion_schedule: Dict[str, torch.Tensor],
    max_grid_size: int,
    max_timesteps: int = 50,
    vocab_size: int = 11,
    examples_per_batch: int = 1,
    device: str | torch.device = "cpu",
    title: str | None = None,
) -> plt.Figure:
    """Create a matplotlib figure visualizing dataset batches and corruption.

    Args:
        batches: Sequence of data batches containing condition and target tokens
        diffusion_schedule: Dictionary with noise schedule tensors
        max_grid_size: Maximum grid size for visualization
        max_timesteps: Maximum number of diffusion timesteps
        vocab_size: Size of token vocabulary
        examples_per_batch: Number of examples to visualize per batch
        device: Device to run corruption on
        title: Optional title for the figure

    Returns:
        Matplotlib figure with visualizations
    """
    if isinstance(device, str):
        device = torch.device(device)

    examples = _collect_examples(
        batches,
        diffusion_schedule=diffusion_schedule,
        max_examples_per_batch=examples_per_batch,
        max_timesteps=max_timesteps,
        vocab_size=vocab_size,
        device=device,
    )
    if not examples:
        raise ValueError("No examples available to visualize.")

    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, max(3, 3 * n_rows / 2)))
    if title:
        fig.suptitle(title, fontsize=16)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, example in enumerate(examples):
        row_axes = axes[row]
        cond_grid = _tokens_to_color_grid(
            example.condition,
            example.condition_mask,
            max_grid_size=max_grid_size,
        )
        tgt_grid = _tokens_to_color_grid(
            example.target,
            example.target_mask,
            max_grid_size=max_grid_size,
        )
        corrupted_grid = _tokens_to_color_grid(
            example.corrupted,
            example.target_mask,
            max_grid_size=max_grid_size,
        )

        _plot_single(
            row_axes[0],
            cond_grid,
            title=f"Batch {example.batch_index + 1} Cond #{example.example_index + 1}",
        )
        _plot_single(row_axes[1], tgt_grid, title="Target")
        _plot_single(
            row_axes[2],
            corrupted_grid,
            title=f"Corrupted (t={example.timestep})",
        )

    if title:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig


def create_corruption_progression_visualization(
    batch: Dict[str, torch.Tensor],
    *,
    diffusion_schedule: Dict[str, torch.Tensor],
    max_grid_size: int,
    max_timesteps: int = 50,
    vocab_size: int = 11,
    example_index: int | None = None,
    device: str | torch.device = "cpu",
    title: str | None = None,
) -> plt.Figure:
    """Visualize corruption progression of examples through all timesteps.

    Args:
        batch: A single batch containing condition and target tokens
        diffusion_schedule: Dictionary with noise schedule tensors
        max_grid_size: Maximum grid size for visualization
        max_timesteps: Maximum number of diffusion timesteps
        vocab_size: Size of token vocabulary
        example_index: Which example from the batch to visualize (None = all examples)
        device: Device to run corruption on
        title: Optional title for the figure

    Returns:
        Matplotlib figure showing the example(s) at t=0, 1, 2, ..., max_timesteps
    """
    if isinstance(device, str):
        device = torch.device(device)

    sqrt_alphas_cumprod = diffusion_schedule["sqrt_alphas_cumprod"]
    sqrt_one_minus = diffusion_schedule["sqrt_one_minus"]

    condition = batch["condition"].to(device)
    condition_mask = batch["condition_mask"].to(device)
    target = batch["target"].to(device)
    target_mask = batch["target_mask"].to(device)

    # Select examples to visualize
    if example_index is not None:
        condition = condition[example_index : example_index + 1]
        condition_mask = condition_mask[example_index : example_index + 1]
        target = target[example_index : example_index + 1]
        target_mask = target_mask[example_index : example_index + 1]

    num_examples = condition.size(0)
    num_timesteps = max_timesteps

    # Generate corrupted versions for each example at each timestep
    corrupted_at_each_t_and_example = []

    for t in range(num_timesteps):
        timesteps = torch.full((num_examples,), t, device=device)
        corrupted = _corrupt_tokens(
            target,
            target_mask,
            timesteps,
            sqrt_alphas_cumprod,
            sqrt_one_minus,
            vocab_size=vocab_size,
        )
        corrupted_at_each_t_and_example.append(corrupted.cpu())

    # Create visualization grid: (3 * num_examples) rows x num_timesteps columns
    # For each example:
    #   Row 1: Condition (repeated across timesteps)
    #   Row 2: Target (repeated across timesteps)
    #   Row 3: Corrupted at each timestep
    fig, axes = plt.subplots(
        3 * num_examples,
        num_timesteps,
        figsize=(2.5 * num_timesteps, 2.5 * 3 * num_examples)
    )
    if title:
        fig.suptitle(title, fontsize=16, y=0.995)

    # Handle single example case
    if num_examples == 1 and num_timesteps == 1:
        axes = np.array([[axes]])
    elif num_examples == 1:
        axes = axes.reshape(3, num_timesteps)
    elif num_timesteps == 1:
        axes = axes.reshape(3 * num_examples, 1)

    for example_idx in range(num_examples):
        row_offset = example_idx * 3

        for col in range(num_timesteps):
            # Plot condition (same for all timesteps)
            cond_grid = _tokens_to_color_grid(
                condition[example_idx].cpu(),
                condition_mask[example_idx].cpu(),
                max_grid_size=max_grid_size,
            )
            title_str = f"Ex{example_idx+1} Condition" if col == 0 else ""
            _plot_single(axes[row_offset, col], cond_grid, title=title_str)

            # Plot target (same for all timesteps)
            tgt_grid = _tokens_to_color_grid(
                target[example_idx].cpu(),
                target_mask[example_idx].cpu(),
                max_grid_size=max_grid_size,
            )
            title_str = f"Ex{example_idx+1} Target" if col == 0 else f"t={col}"
            _plot_single(axes[row_offset + 1, col], tgt_grid, title=title_str)

            # Plot corrupted at this timestep
            corrupted_grid = _tokens_to_color_grid(
                corrupted_at_each_t_and_example[col][example_idx],
                target_mask[example_idx].cpu(),
                max_grid_size=max_grid_size,
            )
            _plot_single(
                axes[row_offset + 2, col],
                corrupted_grid,
                title=f"Corrupted (t={col})"
            )

    fig.tight_layout()
    return fig


__all__ = [
    "ExampleVisualization",
    "create_batch_visualization",
    "create_corruption_progression_visualization",
]
