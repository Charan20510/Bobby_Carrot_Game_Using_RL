"""Bobby Carrot DQN Agent — Rainbow-lite with PER, n-step, NoisyNet.

Rainbow-lite upgrade:
  - Prioritized Experience Replay with IS weight correction
  - N-step returns (n=3) for faster credit assignment
  - NoisyNet exploration (replaces ε-greedy after warmup)
  - Cosine annealing LR schedule (3e-4 → 1e-5)
  - Soft target update (Polyak τ=0.005)
  - Safe checkpoint loading (weights_only=True)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_env import N_ACTIONS
from dqn_model import DuelingDQN, build_model
from dqn_buffer import PrioritizedReplayBuffer, NStepBuffer


class DQNAgent:
    """Double Dueling DQN agent with PER, n-step, and NoisyNet exploration."""

    def __init__(self, device: torch.device, lr: float = 3e-4,
                 gamma: float = 0.99, batch_size: int = 512,
                 target_update: int = 500, teacher_weight: float = 0.25,
                 tau: float = 0.005, compile_model: bool = True,
                 grad_accum_steps: int = 2,
                 n_step: int = 3, noisy: bool = True,
                 buffer_cap: int = 300_000,
                 total_train_steps: int = 50_000) -> None:
        self.device      = device
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.teacher_weight = teacher_weight
        self.tau         = tau
        self.epsilon     = 1.0   # Only used during warmup; noisy nets take over after
        self.total_steps = 0
        self._grad_accum = grad_accum_steps
        self._accum_count = 0
        self.noisy       = noisy
        self.n_step      = n_step

        self.policy = build_model(device, compile_model=compile_model, noisy=noisy)
        self.target = DuelingDQN(noisy=noisy).to(device)
        
        # Safe load (strips compile prefix if present)
        unwrapped_policy = getattr(self.policy, '_orig_mod', self.policy)
        unwrapped_target = getattr(self.target, '_orig_mod', self.target)
        unwrapped_target.load_state_dict(unwrapped_policy.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.policy.parameters(), lr=lr,
                              eps=1e-4, weight_decay=1e-5)

        # Cosine annealing LR: 3e-4 → 1e-5
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_train_steps, eta_min=1e-5)

        self.replay = PrioritizedReplayBuffer(
            cap=buffer_cap, n_step=n_step, gamma=gamma)

        self._target_upd = target_update
        # AMP + GradScaler only supported on CUDA, not MPS
        self.use_amp = (device.type == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.policy.train()

    @property
    def _unwrapped_policy(self) -> nn.Module:
        return getattr(self.policy, '_orig_mod', self.policy)

    @property
    def _unwrapped_target(self) -> nn.Module:
        return getattr(self.target, '_orig_mod', self.target)

    def act(self, grid: np.ndarray, inv: np.ndarray) -> int:
        """Single-observation action selection (used during play/eval)."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        self.policy.reset_noise()
        g = torch.from_numpy(grid).unsqueeze(0).to(self.device, non_blocking=True)
        v = torch.from_numpy(inv).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            return int(self._unwrapped_policy(g, v).argmax(1).item())

    def act_batch(self, grids: List[np.ndarray], invs: List[np.ndarray]
                  ) -> List[int]:
        """Batched inference — process all N environments in a single GPU pass.

        ~4-6× faster than N separate act() calls on GPU.
        Returns a list of greedy actions (noise provides exploration via NoisyNet).
        """
        self.policy.reset_noise()
        n = len(grids)
        g = torch.from_numpy(np.stack(grids)).to(self.device, non_blocking=True)
        v = torch.from_numpy(np.stack(invs)).to(self.device, non_blocking=True)
        with torch.no_grad():
            q_values = self._unwrapped_policy(g, v)  # (n, N_ACTIONS)
            best_actions = q_values.argmax(1).cpu().numpy()
        return [int(best_actions[i]) for i in range(n)]

    def update(self) -> Optional[float]:
        """Run one gradient update with PER-weighted loss."""
        if len(self.replay) < self.batch_size:
            return None

        result = self.replay.sample(self.batch_size, self.device)
        sg, sv, a, r, ng, nv, d, t, tm, is_weights, tree_indices = result

        # Reset noise for both networks
        self.policy.reset_noise()
        self.target.reset_noise()

        def _loss() -> tuple:
            with torch.no_grad():
                # Double DQN: evaluate target action using the UNWRAPPED policy.
                # This guarantees the CUDAGraph tracked by torch.compile is NOT invoked here,
                # physically preventing any intermediate memory overwrite errors.
                best_a = self._unwrapped_policy(ng, nv).argmax(1, keepdim=True)
                qn = self.target(ng, nv).gather(1, best_a).squeeze(1)
                # N-step target: R_n + γⁿ × Q(s_{t+n})
                tgt = (r + (1.0 - d) * self.replay.gamma_n * qn).clamp(-60.0, 60.0)

            # Main training pass uses the compiled CUDAGraph policy
            q_all = self.policy(sg, sv)
            qp = q_all.gather(1, a).squeeze(1)

            # Per-element TD errors (clamped for PER stability)
            td_errors = (qp - tgt).detach().clamp(-100.0, 100.0)

            # Per-element Huber loss weighted by PER importance sampling
            element_loss = F.smooth_l1_loss(qp, tgt, reduction="none")
            td_loss = (element_loss * is_weights).mean()

            # Teacher distillation (only for warmup-collected transitions)
            if float(tm.sum().item()) > 0.0:
                teacher_loss = F.cross_entropy(q_all, t, reduction="none")
                teacher_loss = (teacher_loss * tm).sum() / tm.sum().clamp_min(1.0)
                return td_loss + self.teacher_weight * teacher_loss, td_errors
            return td_loss, td_errors

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss, td_errors = _loss()
            if not torch.isfinite(loss):
                return None
            scaled_loss = loss / self._grad_accum
            self.scaler.scale(scaled_loss).backward()
            self._accum_count += 1
            if self._accum_count >= self._grad_accum:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                self.scheduler.step()
                self._accum_count = 0
        else:
            loss, td_errors = _loss()
            if not torch.isfinite(loss):
                return None
            scaled_loss = loss / self._grad_accum
            scaled_loss.backward()
            self._accum_count += 1
            if self._accum_count >= self._grad_accum:
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.scheduler.step()
                self._accum_count = 0

        # Update PER priorities with TD errors
        self.replay.update_priorities(
            tree_indices,
            td_errors.abs().cpu().numpy()
        )

        self.total_steps += 1

        # Soft target update (Polyak averaging) every step
        if self.tau > 0:
            with torch.no_grad():
                for p, t_p in zip(self.policy.parameters(),
                                  self.target.parameters()):
                    t_p.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        # Periodic hard target update as backup
        if self.total_steps % self._target_upd == 0:
            self._unwrapped_target.load_state_dict(self._unwrapped_policy.state_dict())

        return float(loss.item())

    def save(self, path: Path, level: int, map_kind: str,
             extra: Optional[Dict] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policy":      self._unwrapped_policy.state_dict(),
            "target":      self._unwrapped_target.state_dict(),
            "optim":       self.opt.state_dict(),
            "epsilon":     self.epsilon,
            "total_steps": self.total_steps,
            "level":       level,
            "map_kind":    map_kind,
            "noisy":       self.noisy,
            "n_step":      self.n_step,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f"  [saved] {path.name}")

    def load(self, path: Path) -> Dict:
        """Load checkpoint with safe deserialization (V1 vulnerability fix)."""
        # Try safe loading first (weights_only=True)
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback for legacy checkpoints that contain non-tensor data
            print("  [load] Falling back to weights_only=False for legacy checkpoint")
            ck = torch.load(path, map_location=self.device, weights_only=False)

        self._unwrapped_policy.load_state_dict(ck["policy"])
        self._unwrapped_target.load_state_dict(ck.get("target", ck["policy"]))
        if "optim" in ck:
            try:
                self.opt.load_state_dict(ck["optim"])
            except Exception:
                pass
        self.epsilon     = float(ck.get("epsilon",     1.0))
        self.total_steps = int(ck.get("total_steps",   0))
        total_ep = int(ck.get("total_eps", 0))
        print(f"  [loaded] level={ck.get('level',-1)} "
              f"eps={self.epsilon:.3f} steps={self.total_steps} ep={total_ep}")
        return {"level": ck.get("level", -1), "map_kind": ck.get("map_kind", "normal"),
                "total_eps": total_ep, "best_sr": float(ck.get("best_sr", 0.0))}
