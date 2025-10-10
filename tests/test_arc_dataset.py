import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from dllm.arc_dataset import ARCTaskDataset, arc_collate, load_arc_tasks

DATA_ROOT = ROOT / "data" / "ARC-AGI-master" / "data"


def test_load_arc_tasks_from_canonical_arc_dataset() -> None:
    if not DATA_ROOT.exists():
        pytest.skip("ARC dataset not available at data/ARC-AGI-master/data")

    tasks = load_arc_tasks(DATA_ROOT, split="training")
    assert len(tasks) > 0
    for task in tasks[:5]:
        assert task.demonstrations, f"Task {task.name} has no demonstrations"
        for demo in task.demonstrations:
            assert demo.input_grid and demo.output_grid
            assert len(demo.input_grid) > 0
            assert len(demo.output_grid) > 0


def test_arc_task_dataset_produces_padded_tensors() -> None:
    if not DATA_ROOT.exists():
        pytest.skip("ARC dataset not available at data/ARC-AGI-master/data")

    dataset = ARCTaskDataset(root=DATA_ROOT, split="training", max_grid_size=30, pad_token_id=10)

    assert dataset.num_tasks > 0
    assert len(dataset) > 0

    sample = dataset[0]
    assert set(sample) == {"condition", "condition_mask", "target", "target_mask"}

    for key in ("condition", "target"):
        tensor = sample[key]
        assert tensor.shape == (30 * 30,)
        assert tensor.dtype == torch.long

    for key in ("condition_mask", "target_mask"):
        mask = sample[key]
        assert mask.shape == (30 * 30,)
        assert mask.dtype == torch.float32
        assert mask.max().item() <= 1.0
        assert mask.min().item() >= 0.0

    indices = [0]
    if len(dataset) > 1:
        indices.append(1)
    batch = arc_collate([dataset[i] for i in indices])
    batch_size = len(indices)
    assert batch["condition"].shape == (batch_size, 30 * 30)
    assert batch["condition_mask"].shape == (batch_size, 30 * 30)
    assert batch["target"].shape == (batch_size, 30 * 30)
    assert batch["target_mask"].shape == (batch_size, 30 * 30)
