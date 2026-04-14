"""Bobby Carrot DQN Neural Network — Dueling DQN with NoisyNet exploration.

Rainbow-lite upgrade:
  - NoisyLinear layers in value/advantage heads for learned exploration
  - 14-channel input with directional BFS hint + collected carrot memory
  - Residual connection in FC block
  - torch.compile() auto-applied on CUDA when available

NoisyNet replaces ε-greedy: the network learns WHERE to explore by adding
parametric noise to its weights. This is far more efficient than uniform
random exploration, especially in maze-like environments.
"""
from __future__ import annotations

import math
from typing import cast


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise RuntimeError("PyTorch is required. Install it first.") from e

from dqn_env import GRID_CHANNELS, INV_FEATURES, N_ACTIONS


# ── NoisyLinear — learned exploration ─────────────────────────────────────────
class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer.
    
    Instead of fixed ε-greedy noise, this layer learns noise parameters
    (σ_w, σ_b) alongside the usual weights (μ_w, μ_b). The network decides
    WHERE and HOW MUCH to explore — e.g., exploring more at decision points
    and less in obvious corridors.

    Reference: Fortunato et al., "Noisy Networks for Exploration" (2018)
    """

    epsilon_weight: torch.Tensor
    epsilon_bias: torch.Tensor
    mu_weight: nn.Parameter
    sigma_weight: nn.Parameter
    mu_bias: nn.Parameter
    sigma_bias: nn.Parameter

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.17) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("epsilon_weight", torch.empty(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("epsilon_bias", torch.empty(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorised Gaussian noise — sign(x) * sqrt(|x|)."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample noise. Call before each forward pass during training."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.epsilon_weight.copy_(eps_out.outer(eps_in))
        self.epsilon_bias.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


# ── Dueling DQN with NoisyNet ─────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """3 strided convolutions: 16x16 -> 8x8 -> 4x4 -> 2x2.
    Wider channels (64/128/128) => flat=512 for better planning capacity.
    Dueling architecture separates value from advantage stream.
    NoisyLinear in heads for learned exploration.
    """

    def __init__(self, n_actions: int = N_ACTIONS, noisy: bool = True) -> None:
        super().__init__()
        self.noisy = noisy

        # 16x16 -> 8x8 -> 4x4 -> 2x2
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 64, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(64,           128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128,          128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Flatten(),  # -> 128*2*2 = 512
        )
        flat  = 128 * 2 * 2  # 512
        merge = flat + INV_FEATURES  # 520

        self.fc_proj = nn.Linear(merge, 512)
        self.fc_norm = nn.LayerNorm(512)
        self.fc_act  = nn.ReLU(inplace=True)

        # Value and advantage heads — NoisyLinear for exploration
        Linear = NoisyLinear if noisy else nn.Linear
        self.val_hidden = Linear(512, 128)
        self.val_act    = nn.ReLU(inplace=True)
        self.val_out    = Linear(128, 1)

        self.adv_hidden = Linear(512, 128)
        self.adv_act    = nn.ReLU(inplace=True)
        self.adv_out    = Linear(128, n_actions)

    def forward(self, g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(g)
        merged = torch.cat([conv_out, v], dim=1)
        x = self.fc_act(self.fc_norm(self.fc_proj(merged)))
        val = self.val_out(self.val_act(self.val_hidden(x)))
        adv = self.adv_out(self.adv_act(self.adv_hidden(x)))
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def build_model(device: torch.device, compile_model: bool = True,
                noisy: bool = True) -> nn.Module:
    """Create a DuelingDQN and optionally torch.compile it for speed."""
    model = DuelingDQN(noisy=noisy).to(device)
    if compile_model and device.type == "cuda":
        try:
            # Removed mode="reduce-overhead" because CUDAGraphs interacts poorly 
            # with RL gradient accumulation and Double DQN un-stepped graph histories
            model = torch.compile(model)  # type: ignore[assignment]
            print("  [torch.compile] Model compiled for CUDA acceleration")
        except Exception as e:
            print(f"  [torch.compile] Skipped: {e}")
    return cast(nn.Module, model)
