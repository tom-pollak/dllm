"""Utilities for visualizing ARC batches and diffusion corruption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .diffusion_transformer import DiffusionTransformer

# ARC uses 10 colors (0-9). Index 10 is reserved for padding/empty cells.
# The palette below mirrors the canonical ARC visualization colors.
_COLOR_PALETTE = np.array(
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
) / 255.0


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


def decode_tokens(embeddings: torch.Tensor, token_embed: torch.nn.Embedding) -> torch.Tensor:
    """Map embedding vectors back to discrete token ids via nearest embedding."""

    weight = token_embed.weight
    logits = torch.einsum("bld,vd->blv", embeddings, weight)
    tokens = logits.argmax(dim=-1)
    return tokens


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


def _collect_examples(
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    model: DiffusionTransformer,
    diffusion_schedule: Dict[str, torch.Tensor],
    max_examples_per_batch: int,
) -> List[ExampleVisualization]:
    device = next(model.parameters()).device
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

        timesteps = torch.randint(
            low=0,
            high=model.config.max_timesteps,
            size=(batch_size,),
            device=device,
        )
        target_embed = model.token_embed(target) * target_mask.unsqueeze(-1)
        noise = torch.randn_like(target_embed)
        sqrt_alpha = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one = sqrt_one_minus[timesteps].view(-1, 1, 1)
        noisy = sqrt_alpha * target_embed + sqrt_one * noise
        corrupted_tokens = decode_tokens(noisy, model.token_embed)

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
    model: DiffusionTransformer,
    diffusion_schedule: Dict[str, torch.Tensor],
    max_grid_size: int,
    examples_per_batch: int = 1,
    title: str | None = None,
) -> plt.Figure:
    """Create a matplotlib figure visualizing dataset batches and corruption."""

    examples = _collect_examples(
        batches,
        model=model,
        diffusion_schedule=diffusion_schedule,
        max_examples_per_batch=examples_per_batch,
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


__all__ = [
    "ExampleVisualization",
    "create_batch_visualization",
    "decode_tokens",
]
