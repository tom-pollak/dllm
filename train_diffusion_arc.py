#!/usr/bin/env python3
"""Train a diffusion transformer on ARC style tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split

from dllm import (
    ARCTaskDataset,
    DiffusionTransformer,
    DiffusionTransformerConfig,
    arc_collate,
    build_diffusion_schedule,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=str,
        help=(
            "Path to ARC dataset root directory (the folder containing the "
            "'training' and 'evaluation' sub-directories from the official "
            "fchollet/ARC data dump)."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_arc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay for weights")
    parser.add_argument("--duality-weight", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--resume", type=str, default="", help="Resume checkpoint path")
    parser.add_argument("--augment", action="store_true", help="Enable random flips for augmentation")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=288)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--dim-feedforward", type=int, default=1152)
    parser.add_argument("--time-embed-dim", type=int, default=512)
    parser.add_argument(
        "--skip-param-check",
        action="store_true",
        help="Skip enforcing the ~7M parameter count. Useful for tests.",
    )
    return parser.parse_args(argv)


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


def decode_tokens(embeddings: torch.Tensor, token_embed: torch.nn.Embedding) -> torch.Tensor:
    weight = token_embed.weight
    logits = torch.einsum("bld,vd->blv", embeddings, weight)
    tokens = logits.argmax(dim=-1)
    return tokens


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ARCTaskDataset(
        args.data_dir,
        split="training",
        max_grid_size=args.max_grid_size,
        augment=args.augment,
    )
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=arc_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=arc_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = DiffusionTransformerConfig(
        max_timesteps=args.timesteps,
        max_grid_size=args.max_grid_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        time_embed_dim=args.time_embed_dim,
    )
    model = DiffusionTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    if not args.skip_param_check and not (6.5e6 <= total_params <= 7.5e6):
        raise RuntimeError("Model parameter count deviates from 7M target")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    ema_model = None
    if args.ema > 0:
        ema_model = DiffusionTransformer(config).to(device)
        ema_model.load_state_dict(model.state_dict())

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema_model is not None and "ema" in state:
            ema_model.load_state_dict(state["ema"])
        print(f"Resumed from {args.resume}")

    schedule = build_diffusion_schedule(args.timesteps, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                loss = compute_loss(model, batch, schedule, args.duality_weight)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if args.ema > 0:
                update_ema(model, ema_model, args.ema)
            if step % args.log_interval == 0:
                print(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, schedule, args.duality_weight, device)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        if epoch % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, ema_model, save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = Path(args.output_dir) / "final_model.pt"
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


if __name__ == "__main__":
    main()
