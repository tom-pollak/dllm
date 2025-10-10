"""Test corruption progression visualization."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from dllm import (
    ARCTaskDataset,
    arc_collate,
    build_diffusion_schedule,
    create_corruption_progression_visualization,
)


@pytest.fixture
def arc_batch():
    """Load a single batch from the ARC dataset."""
    data_dir = Path(__file__).parent.parent / "data" / "ARC-AGI-master" / "data"
    if not data_dir.exists():
        pytest.skip(f"ARC dataset not found at {data_dir}")

    dataset = ARCTaskDataset(str(data_dir), split="training", max_grid_size=30)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=arc_collate)
    return next(iter(loader))


def test_corruption_progression_visualization(arc_batch):
    """Test that corruption progression visualization can be created and saved."""
    device = torch.device("cpu")
    timesteps = 50

    # Build diffusion schedule with linear progression for even corruption
    # Linear schedule gives constant noise addition rate (more predictable/gradual)
    # Adjust beta_end to control max corruption (0.02 is standard, lower = gentler)
    schedule = build_diffusion_schedule(
        timesteps, device=device, schedule_type="linear", beta_start=1e-4, beta_end=0.03
    )

    # Create visualization
    fig = create_corruption_progression_visualization(
        arc_batch,
        diffusion_schedule=schedule,
        max_grid_size=30,
        max_timesteps=timesteps,
        vocab_size=11,
        example_index=0,
        device=device,
        title="Corruption Progression Test",
    )

    # Save output
    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_corruption_progression.png"

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved corruption progression visualization to {output_path}")

    # Verify file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
