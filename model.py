"""
model.py — Student Conditional U-Net for the Lines Model

A lightweight U-Net (~2-3M params) designed for single-step image generation
via distillation from a 35.7M-param DDPM teacher.  This is "Model Coarsening":
fewer channels, fewer blocks, shallower depth.

Architecture summary:
    Teacher (DDPM):   [128, 256, 256, 256] × 2 blocks/level  → 35.7M params
    Student (ours):   [64,  128, 256]      × 1 block/level   → ~2.5M params

    Input:  (B, 3, 32, 32)  image  +  (B,) timestep/mixing factor t ∈ [0, 1]
    Output: (B, 3, 32, 32)  predicted clean image
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Map scalar t → high-dimensional embedding via sinusoidal positional encoding,
    then project through a small MLP.

    This is the same encoding used in the original DDPM / Transformer papers,
    adapted to accept continuous t ∈ [0, 1] (we scale by 1000 internally so that
    the frequencies span a useful range).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of mixing factors in [0, 1].
        Returns:
            (B, dim) time embedding.
        """
        # Scale to [0, 1000] so sinusoidal frequencies are useful
        t = t * 1000.0

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        # (B, half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

        return self.mlp(emb)


class ResBlock(nn.Module):
    """
    Residual block with time-conditioning.

        x ──→ GroupNorm → SiLU → Conv ──→ (+t_emb) → GroupNorm → SiLU → Conv ──→ (+skip) → out
                                                                                    ↑
        x ──────────────────────── (optional 1×1 conv if channels change) ──────────┘
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)          # project time emb
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding (broadcast over spatial dims)
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Simple channel-wise self-attention block (QKV from 1×1 convolutions).
    Applied at the bottleneck only to keep the model small.
    """

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-1, -2) @ k) * self.scale   # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)  # (B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided 2× downsampling via convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2× nearest-neighbor upsampling + convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ──────────────────────────────────────────────────────────────
#  Student U-Net
# ──────────────────────────────────────────────────────────────

class StudentUNet(nn.Module):
    """
    Lightweight Conditional U-Net for the Lines Model distillation.

    Spatial resolution pyramid (CIFAR-10, 32×32):
        Encoder: 32 → 16 → 8
        Bottleneck: 8×8 (with self-attention)
        Decoder: 8 → 16 → 32

    Args:
        in_channels:  Input image channels (3 for RGB).
        out_channels: Output channels (3 for RGB).
        channels:     Channel widths at each encoder level.
        t_dim:        Dimensionality of time embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: tuple[int, ...] = (64, 128, 256),
        t_dim: int = 128,
    ):
        super().__init__()
        self.channels = channels

        # ── Time embedding ──
        self.time_embed = SinusoidalTimeEmbedding(t_dim)

        # ── Input projection ──
        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ── Encoder (downsampling path) ──
        #   Level 0: ResBlock(64→64)  → skip=64,  down→16×16
        #   Level 1: ResBlock(64→128) → skip=128, down→8×8
        #   Level 2: ResBlock(128→256)→ skip=256, down→4×4
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_prev = channels[0]
        for ch in channels:
            self.down_blocks.append(ResBlock(ch_prev, ch, t_dim))
            self.downsamples.append(Downsample(ch))
            ch_prev = ch

        # ── Bottleneck (at 4×4) ──
        self.mid_block1 = ResBlock(channels[-1], channels[-1], t_dim)
        self.mid_attn = SelfAttention(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], t_dim)

        # ── Decoder (upsampling path) ──
        # Reverse through encoder levels, concat matching skip each time.
        #   i=0: up(256→8×8),  cat skip[2]=256 → 512, Res(512→256)
        #   i=1: up(256→16×16),cat skip[1]=128 → 384, Res(384→128)
        #   i=2: up(128→32×32),cat skip[0]=64  → 192, Res(192→64)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(channels))  # [256, 128, 64]
        h_ch = channels[-1]  # bottleneck output channels
        for i, out_ch in enumerate(reversed_channels):
            self.upsamples.append(Upsample(h_ch))
            skip_ch = reversed_channels[i]  # matching encoder skip
            self.up_blocks.append(ResBlock(h_ch + skip_ch, out_ch, t_dim))
            h_ch = out_ch

        # ── Output ──
        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 32, 32) input image (the mixed state X_t).
            t: (B,) mixing factor in [0, 1].
        Returns:
            (B, 3, 32, 32) predicted clean image.
        """
        t_emb = self.time_embed(t)       # (B, t_dim)

        # ── Encoder ──
        h = self.in_conv(x)               # (B, 64, 32, 32)
        skips = []
        for res, down in zip(self.down_blocks, self.downsamples):
            h = res(h, t_emb)
            skips.append(h)
            h = down(h)
        # h is now (B, 256, 4, 4)

        # ── Bottleneck ──
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ── Decoder ──
        for i, (up, res) in enumerate(zip(self.upsamples, self.up_blocks)):
            h = up(h)                                 # upsample
            skip = skips[-(i + 1)]                    # matching encoder skip
            h = torch.cat([h, skip], dim=1)           # concat skip connection
            h = res(h, t_emb)

        # ── Output ──
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ──────────────────────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = StudentUNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"Student params: {total:,}  ({total / 1e6:.1f}M)")

    x = torch.randn(4, 3, 32, 32)
    t = torch.rand(4)
    out = model(x, t)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! {out.shape} != {x.shape}"
    print("✓ Shape check passed.")
