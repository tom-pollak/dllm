# ARC Dataset and Data Pipeline

This document describes the refactored utilities that prepare ARC-AGI tasks for
training. The new implementation keeps task boundaries intact, prevents train ↔
validation leakage, and avoids augmentations that mutate the puzzles.

## Loading tasks

`dllm/arc_dataset.py` exposes `load_arc_tasks`, which walks the canonical
`training/` or `evaluation/` directories from the official dataset dump and
returns a list of `ARCTask` objects. Each task stores the filename stem and all
input/output demonstrations found in the JSON file. Empty tasks are skipped, and
an informative error is raised if no tasks exist for the requested split.【F:dllm/arc_dataset.py†L39-L62】

```python
from dllm import load_arc_tasks

tasks = load_arc_tasks("/data/arc", split="training")
print(len(tasks))  # number of ARC puzzles discovered
```

## Task-aware splitting

To create train/validation partitions without leaking demonstrations, use
`split_arc_tasks`. The helper shuffles the task list with a reproducible seed and
selects full tasks for validation. At least one task is kept for training unless
the caller asks for an empty validation fraction. Returning whole tasks prevents
the previous issue where demonstrations from the same puzzle landed in different
splits.【F:dllm/arc_dataset.py†L65-L93】

```python
from dllm import split_arc_tasks

train_tasks, val_tasks = split_arc_tasks(tasks, val_fraction=0.1, seed=42)
```

## Dataset construction

`ARCTaskDataset` now accepts an iterable of `ARCTask` objects (or loads them
directly when given a root directory). It flattens each task into individual
demonstrations **after** the split, ensuring that every sample in a dataset
belongs to a task assigned to that dataset.【F:dllm/arc_dataset.py†L112-L166】 Each
sample provides the tensors expected by the diffusion model:

| Key | Shape | Description |
| --- | ----- | ----------- |
| `"condition"` | `(max_grid_size**2,)` | Flattened input grid padded with the dataset pad token. |
| `"condition_mask"` | `(max_grid_size**2,)` | Binary mask (1.0 for real cells). |
| `"target"` | `(max_grid_size**2,)` | Flattened output grid padded to the same length. |
| `"target_mask"` | `(max_grid_size**2,)` | Binary mask aligned with the output padding. |

Data augmentation is intentionally unsupported: the loader raises an error if
`augment=True`. Earlier flips distorted the provided demonstrations, so the
training data is now kept identical to the authored puzzles.【F:dllm/arc_dataset.py†L124-L128】

`arc_collate` stacks the per-sample tensors into batched tensors with shapes
`(batch_size, max_grid_size**2)` so the training loop can move them to the target
device.【F:dllm/arc_dataset.py†L169-L178】

## Training script integration

`train_diffusion_arc.py` now loads tasks, splits them with `split_arc_tasks`, and
creates independent datasets for each partition. A warning explains that the
`--augment` flag is ignored to keep inputs faithful to the original puzzles.
Because the split happens before dataset construction, no task leaks across
train/validation anymore.【F:train_diffusion_arc.py†L91-L125】

## Respecting grid masks during sampling

The diffusion transformer’s `sample` method previously denoised a full
`max_grid_size × max_grid_size` canvas even when the dataset mask signaled that
only a smaller region was valid. The method now accepts an optional
`target_mask`, initializes the noise only on masked positions, and preserves the
mask throughout the diffusion steps. Sampling no longer hallucinates padded
regions, aligning the generative process with the dataset provided masks.【F:dllm/diffusion_transformer.py†L120-L157】

Together, these changes yield a scalable and clean pipeline: tasks are loaded
once, split without leakage, converted into consistent tensors, and consumed by a
model that respects puzzle geometry.
