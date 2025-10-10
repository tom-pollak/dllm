# ARC Dataset, DataLoader, and Known Problems

This document explains how the project loads ARC-AGI style tasks, how the
`torch.utils.data.Dataset` and `DataLoader` are configured, what tensors are
contained in each training batch, and the main problems with the current
implementation.

## Directory structure and input format

The dataset utilities expect the canonical directory layout distributed in the
[`fchollet/ARC`](https://github.com/fchollet/ARC) repository. When you download
that dataset the root directory contains the sub-folders:

```
<root>/training/
<root>/evaluation/
```

Each sub-folder stores multiple `*.json` task files. Every file contains a list
of training examples under the `"train"` key (the original ARC format also
provides a `"test"` list, which we do not consume during model training).

Within the JSON file each entry inside `"train"` is a dictionary with `"input"`
and `"output"` fields. Each field is a 2-D list of integers representing a color
grid. The integers fall in the range `[0, 9]` for the ten canonical ARC colors.

## `ARCTaskDataset`

[`dllm/arc_dataset.py`](../dllm/arc_dataset.py) defines the
`ARCTaskDataset` class, which inherits from `torch.utils.data.Dataset`.
Key behaviors:

* **Initialization** – the constructor walks the chosen `training` or
  `evaluation` split directory and loads every JSON file. For every pair inside
  the `"train"` list the dataset stores an `ARCExample` dataclass with
  `input_grid` and `output_grid` attributes.【F:dllm/arc_dataset.py†L39-L73】
* **Grid padding** – ARC tasks contain grids of varying size. Before they can
  be fed to the model, each grid is padded to a fixed `max_grid_size ×
  max_grid_size` square (default 30×30). Padding is handled by the private
  `_pad_grid` helper, which returns both the flattened token tensor and a mask
  that marks real (value `1.0`) versus padded (value `0.0`) cells. The padding
  token defaults to `10`, which lies outside the normal color range so models
  can distinguish padding from real pixels.【F:dllm/arc_dataset.py†L22-L63】
* **Samples** – calling `dataset[idx]` yields a dictionary with four keys:
  `"condition"`, `"condition_mask"`, `"target"`, and `"target_mask"`. Each is a
  1-D tensor of length `max_grid_size ** 2`. `condition` and `condition_mask`
  correspond to the example’s input grid, while `target` and `target_mask`
  describe the desired output grid. When `augment=True`, random horizontal and
  vertical flips are applied to both grids (and masks) with independent
  probability `0.5` each.【F:dllm/arc_dataset.py†L65-L116】

The dataset’s length equals the number of `train` pairs found across every JSON
file in the selected split. Importantly, ARC refers to each JSON file as a
single *task* that bundles several input/output demonstrations. The
`ARCTaskDataset` flattens those demonstrations so that every individual
`{"input": ..., "output": ...}` pair becomes its own dataset element. When a
`DataLoader` batches items together (often with `shuffle=True`), the batch may
contain examples originating from many different tasks. There is no special
grouping to keep demonstrations from the same task adjacent, because the
current training objective treats every demonstration independently.

## Collation and DataLoader configuration

Training scripts construct PyTorch `DataLoader` instances using the custom
`arc_collate` function defined alongside the dataset class.

* **`arc_collate`** – this function receives a list of per-item dictionaries and
  stacks the `condition`, `condition_mask`, `target`, and `target_mask` tensors
  into batched tensors with shape `(batch_size, max_grid_size**2)`. The output
  is a dictionary with the same four keys expected by the model.【F:dllm/arc_dataset.py†L118-L128】
* **`DataLoader` setup** – for example, `train_diffusion_arc.py` creates the
  dataset, randomly splits it into training and validation subsets, and then
  wraps them with `DataLoader` objects that specify:
  * `collate_fn=arc_collate`
  * `shuffle=True` for the training loader and `False` for validation
  * `batch_size` configured from the command-line (default `32`)
  * `num_workers` and `pin_memory` tuned for efficient GPU feeding.【F:train_diffusion_arc.py†L69-L115】

## Batch contents

Each batch produced by the `DataLoader` is a dictionary with four entries:

| Key                | Shape                            | DType           | Description                                                              |
| ------------------ | -------------------------------- | --------------- | ------------------------------------------------------------------------ |
| `"condition"`      | `(batch_size, max_grid_size**2)` | `torch.long`    | Flattened input grid tokens with padding tokens (`10`) filling leftovers. |
| `"condition_mask"` | `(batch_size, max_grid_size**2)` | `torch.float32` | Binary mask (1.0 where the input grid is real, 0.0 on padding).           |
| `"target"`         | `(batch_size, max_grid_size**2)` | `torch.long`    | Flattened output grid tokens padded to the same length.                   |
| `"target_mask"`    | `(batch_size, max_grid_size**2)` | `torch.float32` | Binary mask for the output grid, matching the padding pattern.            |

You can move the entire batch to a device using a simple comprehension, as done
in the training script’s `to_device` helper.【F:train_diffusion_arc.py†L57-L64】

These tensors supply the diffusion transformer with both the conditioning input
and the desired target, while the masks allow the loss function to ignore padded
cells when computing reconstruction errors.

## Known problems

Although the sections above describe the intended pipeline, several issues in
the current codebase prevent the ARC loader from matching the canonical task
structure and modelling objective.

### Tasks are flattened into unrelated examples

`ARCTaskDataset` loads every `{"input", "output"}` pair independently and stores
them as separate items in the `examples` list.【F:dllm/arc_dataset.py†L44-L63】 In
effect, the dataset breaks the ARC convention that all demonstrations belonging
to a task should be seen together. When the training script later shuffles the
dataset and slices it with `random_split`, individual demonstrations from the
same task can land in different batches, and even in different train/validation
splits.【F:train_diffusion_arc.py†L74-L105】 This destroys the contextual signal
that ARC solvers rely on (observing multiple demonstrations before producing an
answer for a held-out input), and it introduces leakage where the validation set
may still expose partial information from training tasks.

### Data augmentation corrupts the provided grids

When the `--augment` flag is enabled, `_augment` performs random horizontal and
vertical flips on both the input (`condition`) and the output (`target`) grids in
each sample.【F:dllm/arc_dataset.py†L86-L116】 ARC demonstrations are carefully
constructed; transforming the input grid changes the puzzle itself and can make
the paired output meaningless. Because the goal is to generate the output grid
given an unmodified input, these flips effectively corrupt the supervision
signal by altering the examples that should remain fixed.

A safer strategy would be to apply the **same** augmentation to every
demonstration belonging to a task so that relative relationships stay intact, or
to restrict augmentation to the generated output while leaving the conditioning
input untouched. Another promising idea is to treat all demonstrations in a task
as a candidate target: for a task that ships four examples, the loader could
pick one of the demonstrations as the "output" and repurpose the remaining three
as conditioning inputs, cycling this choice across epochs. Either approach would
respect the intent of ARC tasks while still expanding the variety of supervision
the model sees.

### Diffusion objective ignores task-specific geometry

During training, `compute_loss` embeds the target tokens, applies Gaussian noise,
and asks the model to predict that noise.【F:train_diffusion_arc.py†L187-L235】
While this is standard for diffusion models, the implementation does not supply
the true target mask to the sampler: `DiffusionTransformer.sample` always
constructs an all-ones `target_mask`, forcing the model to denoise a full
30×30 grid regardless of the original puzzle size.【F:dllm/diffusion_transformer.py†L120-L160】
Consequently the network must learn to hallucinate outputs for padded regions
that should remain unused, and the sampling procedure cannot take advantage of
the sparsity information available in the dataset.
