#!/usr/bin/env python3
"""Train a diffusion transformer on ARC style tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader, random_split

from pydantic import Field, model_validator
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
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(50, ge=1)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    timesteps: int = Field(1000, ge=1)
    val_fraction: float = Field(0.1, ge=0.0, lt=1.0)
    seed: int = 42
    grad_clip: float = Field(1.0, ge=0.0)
    device: str = Field(default_factory=_default_device)
    ema: float = Field(0.0, ge=0.0, le=1.0)
    duality_weight: float = Field(0.5, ge=0.0)
    log_interval: int = Field(100, ge=1)
    num_workers: int = Field(2, ge=0)
    save_interval: int = Field(5, ge=1)
    resume: Path | None = None
    augment: bool = False
    mixed_precision: bool = False
    max_grid_size: int = Field(30, ge=1)
    d_model: int = Field(288, ge=1)
    num_heads: int = Field(8, ge=1)
    num_layers: int = Field(7, ge=1)
    dim_feedforward: int = Field(1152, ge=1)
    time_embed_dim: int = Field(512, ge=1)

    model_config = SettingsConfig(extra="forbid")

    @model_validator(mode="after")
    def _normalise_paths(self) -> "DiffusionArcTrainingConfig":
        self.data_dir = self.data_dir.expanduser()
        if not self.data_dir.exists():
            raise ValueError(f"data_dir does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"data_dir must be a directory: {self.data_dir}")

        self.output_dir = self.output_dir.expanduser()

        if self.resume is not None:
            self.resume = self.resume.expanduser()
            if not self.resume.exists():
                raise ValueError(f"resume checkpoint not found: {self.resume}")

        return self

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
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ARCTaskDataset(
        cfg.data_dir,
        split="training",
        max_grid_size=cfg.max_grid_size,
        augment=cfg.augment,
    )
    total_items = len(dataset)
    if total_items == 0:
        raise RuntimeError(f"No ARC tasks found in {cfg.data_dir}")

    val_size = int(total_items * cfg.val_fraction)
    if cfg.val_fraction > 0 and val_size == 0 and total_items > 1:
        val_size = 1
    if val_size >= total_items:
        val_size = total_items - 1
    train_size = total_items - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=arc_collate,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=arc_collate,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
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
            save_path = cfg.output_dir / f"checkpoint_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, ema_model, save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = cfg.output_dir / "final_model.pt"
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
