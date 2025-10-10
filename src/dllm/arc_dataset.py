"""Utilities for loading ARC-AGI style datasets.

The loader expects the canonical ARC directory structure from the
`fchollet/ARC <https://github.com/fchollet/ARC>`_ repository (or an
equivalent mirror). ``root`` should point at the folder that contains the
``training`` and ``evaluation`` sub-directories with ``*.json`` task files.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

ColorGrid = List[List[int]]


@dataclass
class ARCExample:
    """Single ARC training example."""

    input_grid: ColorGrid
    output_grid: ColorGrid


def _pad_grid(
    grid: ColorGrid,
    max_size: int,
    pad_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a grid to ``max_size`` and return (values, mask)."""

    height, width = len(grid), len(grid[0])
    tensor = torch.full((max_size, max_size), pad_value, dtype=torch.long)
    mask = torch.zeros((max_size, max_size), dtype=torch.float32)
    tensor[:height, :width] = torch.tensor(grid, dtype=torch.long)
    mask[:height, :width] = 1.0
    return tensor.view(-1), mask.view(-1)


class ARCTaskDataset(Dataset):
    """Dataset that reads ARC style json files."""

    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        max_grid_size: int = 30,
        pad_token_id: int = 10,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.max_grid_size = max_grid_size
        self.pad_token_id = pad_token_id
        self.augment = augment
        self.examples: List[ARCExample] = []
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory {split_dir} was not found.")
        for path in sorted(split_dir.glob("*.json")):
            with path.open("r") as fp:
                data = json.load(fp)
            for pair in data.get("train", []):
                self.examples.append(
                    ARCExample(input_grid=pair["input"], output_grid=pair["output"])
                )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        input_grid, input_mask = _pad_grid(
            ex.input_grid, self.max_grid_size, self.pad_token_id
        )
        target_grid, target_mask = _pad_grid(
            ex.output_grid, self.max_grid_size, self.pad_token_id
        )
        sample = {
            "condition": input_grid,
            "condition_mask": input_mask,
            "target": target_grid,
            "target_mask": target_mask,
        }
        if self.augment and random.random() < 0.5:
            sample = self._augment(sample)
        return sample

    def _augment(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply simple random flips to encourage invariances."""

        side = self.max_grid_size
        cond = sample["condition"].view(side, side)
        targ = sample["target"].view(side, side)
        mask_c = sample["condition_mask"].view(side, side)
        mask_t = sample["target_mask"].view(side, side)

        if random.random() < 0.5:
            cond = torch.flip(cond, dims=[0])
            targ = torch.flip(targ, dims=[0])
            mask_c = torch.flip(mask_c, dims=[0])
            mask_t = torch.flip(mask_t, dims=[0])
        if random.random() < 0.5:
            cond = torch.flip(cond, dims=[1])
            targ = torch.flip(targ, dims=[1])
            mask_c = torch.flip(mask_c, dims=[1])
            mask_t = torch.flip(mask_t, dims=[1])

        return {
            "condition": cond.view(-1),
            "condition_mask": mask_c.view(-1),
            "target": targ.view(-1),
            "target_mask": mask_t.view(-1),
        }


def arc_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    condition = torch.stack([sample["condition"] for sample in batch])
    condition_mask = torch.stack([sample["condition_mask"] for sample in batch])
    target = torch.stack([sample["target"] for sample in batch])
    target_mask = torch.stack([sample["target_mask"] for sample in batch])
    return {
        "condition": condition,
        "condition_mask": condition_mask,
        "target": target,
        "target_mask": target_mask,
    }
