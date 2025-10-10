# Diffusion Transformer for ARC

This repository contains an experimental diffusion transformer training pipeline for ARC-AGI style reasoning tasks.

## Dataset

The training script expects the canonical ARC task JSON files from the official [fchollet/ARC](https://github.com/fchollet/ARC) repository. Download or clone that repository and point the training command at the `data` directory inside it, which contains the `training/` and `evaluation/` folders:

```bash
# Example setup
mkdir -p data
cd data
curl -L https://github.com/fchollet/ARC/archive/refs/heads/master.zip -o arc.zip
unzip arc.zip 'ARC-master/data/*'
cd ..

# Run training
python train_diffusion_arc.py data/ARC-master/data
```

Any mirror with the same folder structure will also work. The `ARCTaskDataset` loader simply walks every `*.json` file inside the specified split directory.

## Visualization

To inspect the batches used during training, including how the diffusion process corrupts the targets, run the visualization helper:

```bash
python batch_visualization.py data/ARC-master/data --checkpoint outputs/diffusion_arc/final_model.pt
```

This command saves `train_batches.png` and `val_batches.png` under `outputs/visualizations/`, each showing five batches of samples with the condition, target, and a randomly corrupted view at different diffusion timesteps (defaulting to a compact 0–99 range).

## Tests

A minimal CPU smoke test is available via:

```bash
pytest tests/test_train_diffusion_arc.py -k tiny_cpu
```
