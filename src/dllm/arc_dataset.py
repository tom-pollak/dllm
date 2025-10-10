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
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

ColorGrid = List[List[int]]


@dataclass
class ARCDemonstration:
    """Single input/output pair provided inside an ARC task."""

    input_grid: ColorGrid
    output_grid: ColorGrid


@dataclass
class ARCTask:
    """Collection of demonstrations that describe one ARC puzzle."""

    name: str
    demonstrations: List[ARCDemonstration]


def load_arc_tasks(root: str | Path, split: str = "training") -> List[ARCTask]:
    """Load ARC tasks from ``root/split`` preserving task boundaries."""

    root_path = Path(root)
    split_dir = root_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} was not found.")

    tasks: List[ARCTask] = []
    for path in sorted(split_dir.glob("*.json")):
        with path.open("r") as fp:
            data = json.load(fp)

        demonstrations = [
            ARCDemonstration(input_grid=pair["input"], output_grid=pair["output"])
            for pair in data.get("train", [])
        ]
        if not demonstrations:
            continue
        tasks.append(ARCTask(name=path.stem, demonstrations=demonstrations))

    if not tasks:
        raise ValueError(f"No ARC tasks were found in {split_dir}.")
    return tasks


def split_arc_tasks(
    tasks: Sequence[ARCTask],
    val_fraction: float,
    seed: int,
) -> Tuple[List[ARCTask], List[ARCTask]]:
    """Split a sequence of tasks into train/validation lists.

    Splitting happens at the *task* level to avoid leaking demonstrations
    from the same puzzle across splits.
    """

    task_list = list(tasks)
    if not task_list:
        return [], []

    rng = random.Random(seed)
    indices = list(range(len(task_list)))
    rng.shuffle(indices)

    val_count = int(len(task_list) * val_fraction)
    if val_fraction > 0 and val_count == 0 and len(task_list) > 1:
        val_count = 1
    if val_count >= len(task_list):
        val_count = len(task_list) - 1

    val_indices = set(indices[:val_count])
    train_tasks = [task_list[i] for i in indices if i not in val_indices]
    val_tasks = [task_list[i] for i in indices if i in val_indices]
    return train_tasks, val_tasks


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
    """Dataset that exposes ARC demonstrations without breaking task groups."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        tasks: Iterable[ARCTask] | None = None,
        split: str = "training",
        max_grid_size: int = 30,
        pad_token_id: int = 10,
        augment: bool = False,
    ) -> None:
        super().__init__()
        if augment:
            raise ValueError(
                "Data augmentation is not supported because it corrupts ARC tasks."
            )

        if tasks is None:
            if root is None:
                raise ValueError("Either `root` or `tasks` must be provided.")
            tasks = load_arc_tasks(root, split)

        self.tasks: List[ARCTask] = list(tasks)
        self.max_grid_size = max_grid_size
        self.pad_token_id = pad_token_id

        self._examples: List[Tuple[int, ARCDemonstration]] = []
        for task_idx, task in enumerate(self.tasks):
            for demo in task.demonstrations:
                self._examples.append((task_idx, demo))

    @property
    def num_tasks(self) -> int:
        """Return the number of unique ARC tasks in this dataset."""

        return len(self.tasks)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _, demo = self._examples[idx]
        condition, condition_mask = _pad_grid(
            demo.input_grid, self.max_grid_size, self.pad_token_id
        )
        target, target_mask = _pad_grid(
            demo.output_grid, self.max_grid_size, self.pad_token_id
        )
        return {
            "condition": condition,
            "condition_mask": condition_mask,
            "target": target,
            "target_mask": target_mask,
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
