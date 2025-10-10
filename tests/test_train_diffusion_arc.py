import json
import sys
import textwrap
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

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            data_dir: arc
            output_dir: outputs
            batch_size: 1
            epochs: 1
            timesteps: 4
            num_workers: 0
            log_interval: 1
            device: cpu
            max_grid_size: 3
            d_model: 16
            num_heads: 4
            num_layers: 1
            dim_feedforward: 32
            time_embed_dim: 32
            duality_weight: 0.0
            """
        ).strip()
        + "\n"
    )

    main(config_path)

    assert (output_dir / "final_model.pt").exists()
