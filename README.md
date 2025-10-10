# Diffusion Transformer for ARC

This repository contains an experimental diffusion transformer training pipeline for ARC-AGI style reasoning tasks.

## Dataset

The training script expects the canonical ARC task JSON files from the official [fchollet/ARC](https://github.com/fchollet/ARC) repository. Download or clone that repository and point the training command at the `data` directory inside it, which contains the `training/` and `evaluation/` folders:

```bash
# Example setup
mkdir -p data
cd data
curl -L https://github.com/fchollet/ARC/archive/refs/heads/master.zip -o arc.zip
unzip arc.zip 'ARC-AGI-master/data/*'
cd ..

# Run training with a YAML config (see the Configuration section below)
python train_diffusion_arc.py path/to/config.yaml
```

Any mirror with the same folder structure will also work. The `ARCTaskDataset` loader simply walks every `*.json` file inside the specified split directory.

## Visualization

To inspect the batches used during training, including how the diffusion process corrupts the targets, run the visualization helper:

```bash
python batch_visualization.py data/ARC-AGI-master/data --checkpoint outputs/diffusion_arc/final_model.pt
```

This command saves `train_batches.png` and `val_batches.png` under `outputs/visualizations/`, each showing five batches of samples with the condition, target, and a randomly corrupted view at different diffusion timesteps (defaulting to a compact 0–99 range).

## Configuration

Training is configured through a YAML file validated by `DiffusionArcTrainingConfig` from `train_diffusion_arc.py` using [`pydantic_config`](https://github.com/samsja/pydantic_config). Install the dependency with:

```bash
pip install pydantic_config pyyaml
```

Create a YAML file describing your run. Every field has a sensible default except `data_dir` which must point at the ARC dataset root. The available options are:

| Field | Description |
| --- | --- |
| `data_dir` | Path to the ARC dataset root containing `training/` and `evaluation/` folders. The directory must exist. |
| `output_dir` | Directory where checkpoints and the final model will be written (created automatically when missing). |
| `batch_size` | Batch size for both training and validation loaders (must be ≥ 1). |
| `epochs` | Number of full passes over the training set (must be ≥ 1). |
| `lr` / `weight_decay` | AdamW optimizer hyper-parameters (learning rate must be > 0). |
| `timesteps` | Number of diffusion steps in the schedule (must be ≥ 1). |
| `val_fraction` | Fraction of the dataset used for validation. Values > 0 reserve at least one example when possible and must be < 1. |
| `seed` | Random seed for Python, PyTorch and data splits. |
| `grad_clip` | Gradient clipping value (set to `0` to disable). |
| `device` | Device string understood by `torch.device`, defaults to `cuda` when available. |
| `ema` | Exponential moving average decay for model weights (`0` disables EMA, must be between `0` and `1`). |
| `duality_weight` | Weight applied to the clean target reconstruction loss term (must be ≥ 0). |
| `log_interval` | Number of training steps between log messages (must be ≥ 1). |
| `num_workers` | Data loader worker count (must be ≥ 0). |
| `save_interval` | Save a checkpoint every N epochs (must be ≥ 1). |
| `resume` | Optional path to a checkpoint to resume from. The file must exist when provided. |
| `augment` | Enable random grid flips during dataset loading. |
| `mixed_precision` | Enable automatic mixed precision training. |
| `max_grid_size`, `d_model`, `num_heads`, `num_layers`, `dim_feedforward`, `time_embed_dim` | Architectural parameters passed to `DiffusionTransformerConfig`. |

Example configuration:

```yaml
data_dir: data/ARC-master/data
output_dir: outputs/diffusion_arc
batch_size: 32
epochs: 50
lr: 0.0003
weight_decay: 0.01
timesteps: 1000
val_fraction: 0.1
seed: 42
grad_clip: 1.0
device: cuda
ema: 0.0
duality_weight: 0.5
log_interval: 100
num_workers: 2
save_interval: 5
augment: false
mixed_precision: false
max_grid_size: 30
d_model: 288
num_heads: 8
num_layers: 7
dim_feedforward: 1152
time_embed_dim: 512
```

Run training by pointing the script at your YAML file:

```bash
python train_diffusion_arc.py path/to/config.yaml
```

## Tests

A minimal CPU smoke test is available via:

```bash
pytest tests/test_train_diffusion_arc.py -k tiny_cpu
```
