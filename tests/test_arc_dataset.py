import json
from pathlib import Path
from typing import Iterable

import pytest

torch = pytest.importorskip("torch")

from dllm.arc_dataset import (
    ARCDemonstration,
    ARCTask,
    ARCTaskDataset,
    arc_collate,
    load_arc_tasks,
    split_arc_tasks,
)


def _write_arc_task(path: Path, demonstrations: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump({"train": list(demonstrations)}, fp)


def test_load_arc_tasks_from_synthetic_dataset(tmp_path: Path) -> None:
    _write_arc_task(
        tmp_path / "training" / "puzzle_one.json",
        [
            {"input": [[1, 1], [0, 0]], "output": [[2, 2], [0, 0]]},
            {"input": [[3, 3], [3, 3]], "output": [[4, 4], [4, 4]]},
        ],
    )
    _write_arc_task(
        tmp_path / "training" / "puzzle_two.json",
        [
            {"input": [[5]], "output": [[5]]},
        ],
    )

    tasks = load_arc_tasks(tmp_path, split="training")

    assert [task.name for task in tasks] == ["puzzle_one", "puzzle_two"]
    assert len(tasks[0].demonstrations) == 2
    assert len(tasks[1].demonstrations) == 1


@pytest.mark.parametrize("val_fraction", [0.0, 0.25, 0.5, 0.9])
def test_split_arc_tasks_is_task_level(val_fraction: float) -> None:
    tasks = [
        ARCTask(
            name=f"task_{i}",
            demonstrations=[
                ARCDemonstration(input_grid=[[i]], output_grid=[[i + 1]])
                for _ in range(2)
            ],
        )
        for i in range(6)
    ]

    train_a, val_a = split_arc_tasks(tasks, val_fraction=val_fraction, seed=42)
    train_b, val_b = split_arc_tasks(tasks, val_fraction=val_fraction, seed=42)

    assert {task.name for task in train_a}.isdisjoint(
        {task.name for task in val_a}
    )
    assert {task.name for task in val_a} == {task.name for task in val_b}
    assert {task.name for task in train_a} == {task.name for task in train_b}
    if tasks and 0 < val_fraction < 1:
        assert val_a, "Expected at least one validation task when fraction > 0"
        assert train_a, "Expected at least one training task when fraction < 1"


@pytest.mark.parametrize("max_grid_size", [3, 5])
@pytest.mark.parametrize("use_tasks", [True, False])
def test_arc_task_dataset_and_collate(
    tmp_path: Path, max_grid_size: int, use_tasks: bool
) -> None:
    _write_arc_task(
        tmp_path / "training" / "sample.json",
        [
            {"input": [[1, 0], [0, 0]], "output": [[2, 2], [0, 0]]},
            {"input": [[3, 3], [0, 0]], "output": [[4, 4], [0, 0]]},
        ],
    )

    tasks = load_arc_tasks(tmp_path, split="training")
    if use_tasks:
        dataset = ARCTaskDataset(
            tasks=tasks, max_grid_size=max_grid_size, pad_token_id=9
        )
    else:
        dataset = ARCTaskDataset(
            root=tmp_path, max_grid_size=max_grid_size, pad_token_id=9
        )

    assert dataset.num_tasks == 1
    assert len(dataset) == 2

    example = dataset[0]
    assert set(example) == {"condition", "condition_mask", "target", "target_mask"}
    for key in ("condition", "target"):
        tensor = example[key]
        assert tensor.shape == (max_grid_size * max_grid_size,)
        assert tensor.dtype == torch.long
    for key in ("condition_mask", "target_mask"):
        mask = example[key]
        assert mask.shape == (max_grid_size * max_grid_size,)
        assert mask.dtype == torch.float32
        assert mask.max().item() <= 1.0
        assert mask.min().item() >= 0.0

    batch = arc_collate([dataset[0], dataset[1]])
    for key, value in batch.items():
        assert value.shape == (2, max_grid_size * max_grid_size)

