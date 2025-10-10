import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from train_diffusion_arc import main


def test_main_runs_with_tiny_model(tmp_path: Path) -> None:
    data_dir = tmp_path / "arc"
    training_dir = data_dir / "training"
    training_dir.mkdir(parents=True)

    task_content = {
        "train": [
            {
                "input": [[0, 1, 2], [2, 1, 0], [0, 0, 0]],
                "output": [[1, 2, 0], [0, 1, 2], [2, 2, 2]],
            },
            {
                "input": [[3, 3, 3], [3, 4, 3], [3, 3, 3]],
                "output": [[4, 4, 4], [4, 5, 4], [4, 4, 4]],
            },
        ],
        "test": [],
    }
    (training_dir / "task.json").write_text(json.dumps(task_content))

    output_dir = tmp_path / "outputs"

    argv = [
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "1",
        "--epochs",
        "10",
        "--timesteps",
        "4",
        "--num-workers",
        "0",
        "--log-interval",
        "50",
        "--skip-param-check",
        "--max-grid-size",
        "3",
        "--d-model",
        "16",
        "--num-heads",
        "4",
        "--num-layers",
        "3",
        "--dim-feedforward",
        "32",
        "--time-embed-dim",
        "10",
    ]

    main(argv)

    assert (output_dir / "final_model.pt").exists()
