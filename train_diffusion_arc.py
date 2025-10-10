#!/usr/bin/env python3
"""Train a diffusion transformer on ARC style tasks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader, random_split

from pydantic import Field
from pydantic_config import SettingsConfig, SettingsModel
from pydantic_config.main import ConfigFileSettingsSource

from dllm import (
    ARCTaskDataset,
    DiffusionTransformer,
    DiffusionTransformerConfig,
    arc_collate,
    build_diffusion_schedule,
)

def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class DiffusionArcTrainingConfig(SettingsModel):
    """Configuration for diffusion transformer ARC training."""

    data_dir: Path
    output_dir: Path = Path("outputs/diffusion_arc")
    batch_size: int = 32
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.01
    timesteps: int = 1000
    val_fraction: float = 0.1
    seed: int = 42
    grad_clip: float = 1.0
    device: str = Field(default_factory=_default_device)
    ema: float = 0.0
    duality_weight: float = 0.5
    log_interval: int = 100
    num_workers: int = 2
    save_interval: int = 5
    resume: Path | None = None
    augment: bool = False
    mixed_precision: bool = False
    max_grid_size: int = 30
    d_model: int = 288
    num_heads: int = 8
    num_layers: int = 7
    dim_feedforward: int = 1152
    time_embed_dim: int = 512

    model_config = SettingsConfig(extra="forbid")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DiffusionArcTrainingConfig":
        source = ConfigFileSettingsSource(
            cls,
            config_file=Path(path),
            config_file_required=True,
        )
        data = source()
        return cls.model_validate(data)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    mse = (diff ** 2).sum() / mask.sum().clamp(min=1.0)
    return mse


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def load_training_config(config: DiffusionArcTrainingConfig | Path | str) -> DiffusionArcTrainingConfig:
    if isinstance(config, DiffusionArcTrainingConfig):
        return config
    return DiffusionArcTrainingConfig.from_yaml(config)


def main(config: DiffusionArcTrainingConfig | Path | str) -> None:
    cfg = load_training_config(config)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset = ARCTaskDataset(
        cfg.data_dir,
        split="training",
        max_grid_size=cfg.max_grid_size,
        augment=cfg.augment,
    )
    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=arc_collate,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=arc_collate,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model_config = DiffusionTransformerConfig(
        max_timesteps=cfg.timesteps,
        max_grid_size=cfg.max_grid_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        time_embed_dim=cfg.time_embed_dim,
    )
    model = DiffusionTransformer(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    ema_model = None
    if cfg.ema > 0:
        ema_model = DiffusionTransformer(model_config).to(device)
        ema_model.load_state_dict(model.state_dict())

    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema_model is not None and "ema" in state:
            ema_model.load_state_dict(state["ema"])
        print(f"Resumed from {cfg.resume}")

    schedule = build_diffusion_schedule(cfg.timesteps, device=device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                loss = compute_loss(model, batch, schedule, cfg.duality_weight)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if cfg.ema > 0 and ema_model is not None:
                update_ema(model, ema_model, cfg.ema)
            if step % cfg.log_interval == 0:
                print(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, schedule, cfg.duality_weight, device)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        if epoch % cfg.save_interval == 0:
            save_path = Path(cfg.output_dir) / f"checkpoint_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, ema_model, save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = Path(cfg.output_dir) / "final_model.pt"
    save_checkpoint(model, optimizer, scheduler, ema_model, final_path)
    print(f"Training completed, model saved to {final_path}")


@torch.no_grad()
def evaluate(
    model: DiffusionTransformer,
    loader: DataLoader,
    schedule: Dict[str, torch.Tensor],
    duality_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        loss = compute_loss(model, batch, schedule, duality_weight)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def compute_loss(
    model: DiffusionTransformer,
    batch: Dict[str, torch.Tensor],
    schedule: Dict[str, torch.Tensor],
    duality_weight: float,
) -> torch.Tensor:
    target_tokens = batch["target"]
    target_mask = batch["target_mask"].unsqueeze(-1)
    cond_tokens = batch["condition"]
    cond_mask = batch["condition_mask"]
    cfg = model.config

    timesteps = torch.randint(0, cfg.max_timesteps, (target_tokens.size(0),), device=target_tokens.device)
    target_embed = model.token_embed(target_tokens) * target_mask
    noise = torch.randn_like(target_embed)
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"][timesteps].view(-1, 1, 1)
    sqrt_one_minus = schedule["sqrt_one_minus"][timesteps].view(-1, 1, 1)
    noisy = sqrt_alphas_cumprod * target_embed + sqrt_one_minus * noise
    pred_noise = model(noisy, cond_tokens, cond_mask, batch["target_mask"], timesteps)
    weighted_mask = target_mask
    noise_loss = mask_mse(pred_noise, noise, weighted_mask)
    if duality_weight > 0:
        pred_x0 = (noisy - sqrt_one_minus * pred_noise) / sqrt_alphas_cumprod
        clean_loss = mask_mse(pred_x0, target_embed, weighted_mask)
        return noise_loss + duality_weight * clean_loss
    return noise_loss


def update_ema(model: DiffusionTransformer, ema: DiffusionTransformer, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1 - decay)
            else:
                v.copy_(msd[k])


def save_checkpoint(
    model: DiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema_model: DiffusionTransformer | None,
    path: Path,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if ema_model is not None:
        payload["ema"] = ema_model.state_dict()
    torch.save(payload, path)


def _main_cli(argv: Sequence[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("Usage: python train_diffusion_arc.py <config.yaml>")
    main(Path(argv[0]))


if __name__ == "__main__":
    _main_cli(sys.argv[1:])
