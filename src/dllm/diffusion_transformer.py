"""Diffusion Transformer model tailored for ARC tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule for beta values.

    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value (low noise)
        beta_end: Ending beta value (high noise)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from https://arxiv.org/abs/2102.09672.

    Args:
        timesteps: Number of diffusion steps
        s: Offset parameter controlling schedule steepness (higher = gentler)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


@dataclass
class DiffusionTransformerConfig:
    vocab_size: int = 11
    pad_token_id: int = 10
    max_grid_size: int = 30
    d_model: int = 288
    num_heads: int = 8
    num_layers: int = 7
    dim_feedforward: int = 1152
    dropout: float = 0.1
    max_timesteps: int = 10
    time_embed_dim: int = 512

    @property
    def max_tokens(self) -> int:
        return self.max_grid_size * self.max_grid_size

    @property
    def seq_len(self) -> int:
        return self.max_tokens * 2


class DiffusionTransformer(nn.Module):
    def __init__(self, config: DiffusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.seq_len, config.d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        with torch.no_grad():
            self.token_embed.weight[self.config.pad_token_id].zero_()

    @property
    def pad_token_id(self) -> int:  # pragma: no cover - simple proxy
        return self.config.pad_token_id

    def forward(
        self,
        noisy_target: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        target_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for the target tokens."""

        cfg = self.config
        _, target_tokens, _ = noisy_target.shape
        device = noisy_target.device
        total_tokens = cfg.max_tokens

        cond_pos = torch.arange(total_tokens, device=device)
        tgt_pos = torch.arange(total_tokens, device=device) + total_tokens

        cond_emb = self.token_embed(condition_tokens) + self.position_embed(cond_pos).unsqueeze(0)
        noisy_emb = noisy_target + self.position_embed(tgt_pos).unsqueeze(0)

        time_emb = timestep_embedding(timesteps, cfg.time_embed_dim).to(device)
        time_emb = self.time_embed(time_emb)
        cond_emb = cond_emb + time_emb.unsqueeze(1)
        noisy_emb = noisy_emb + time_emb.unsqueeze(1)

        sequence = torch.cat([cond_emb, noisy_emb], dim=1)
        key_padding_mask = torch.cat([
            (condition_mask == 0),
            (target_mask == 0),
        ], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=key_padding_mask)
        encoded = self.layer_norm(encoded)
        pred = self.output_proj(encoded[:, -target_tokens:, :])
        return pred

    def sample(
        self,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        diffusion_schedule: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Iteratively sample a target grid given a condition."""

        cfg = self.config
        device = condition_tokens.device
        total_tokens = cfg.max_tokens
        timesteps = steps or cfg.max_timesteps
        sqrt_recip_alphas = diffusion_schedule["sqrt_recip_alphas"]
        betas = diffusion_schedule["betas"]
        posterior_variance = diffusion_schedule["posterior_variance"]
        target_shape = (condition_tokens.size(0), total_tokens, cfg.d_model)
        if target_mask is None:
            target_mask = torch.ones_like(condition_mask)
        mask = target_mask.unsqueeze(-1)
        x = torch.randn(target_shape, device=device) * mask

        for i in reversed(range(timesteps)):
            t = torch.full((condition_tokens.size(0),), i, device=device, dtype=torch.long)
            model_out = self.forward(x, condition_tokens, condition_mask, target_mask, t)
            if guidance_scale != 1.0:
                model_out = model_out * guidance_scale
            sqrt_recip_alpha = sqrt_recip_alphas[i]
            beta = betas[i]
            model_mean = sqrt_recip_alpha * x - beta * model_out
            if i > 0:
                noise = torch.randn_like(x)
                x = (model_mean + torch.sqrt(posterior_variance[i]) * noise) * mask
            else:
                x = model_mean * mask
        return x


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def build_diffusion_schedule(
    timesteps: int,
    device: torch.device,
    schedule_type: str = "cosine",
    s: float = 0.008,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Build diffusion noise schedule.

    Args:
        timesteps: Number of diffusion steps
        device: Device to place tensors on
        schedule_type: Type of schedule ("linear" or "cosine")
        s: Cosine schedule offset parameter (only used if schedule_type="cosine")
        beta_start: Linear schedule start value (only used if schedule_type="linear")
        beta_end: Linear schedule end value (only used if schedule_type="linear")

    Returns:
        Dictionary containing schedule tensors
    """
    if schedule_type == "linear":
        betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end).to(device)
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(timesteps, s=s).to(device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}. Must be 'linear' or 'cosine'")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus": sqrt_one_minus,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }
