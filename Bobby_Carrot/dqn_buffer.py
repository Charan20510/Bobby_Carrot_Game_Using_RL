"""Bobby Carrot DQN Replay Buffer — Prioritized Experience Replay + N-step.

Rainbow-lite upgrade:
  - SumTree-based PER: important transitions sampled up to 10× more often
  - N-step returns (default n=3): faster credit assignment over long episodes
  - Importance sampling weights with β annealing to correct PER bias
  - New transitions get max priority (guaranteed first sample)

Performance: iterative SumTree propagation + vectorised batch operations.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch

from dqn_env import GRID_CHANNELS, INV_FEATURES


# ── SumTree for O(log n) prioritized sampling ─────────────────────────────────
class SumTree:
    """Binary tree where each leaf stores a priority value.

    Parent nodes store the sum of children, enabling O(log n) sampling
    proportional to priority. Uses ITERATIVE propagation for speed.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Iterative bottom-up propagation (faster than recursive)."""
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """Add a new priority, return the tree leaf index."""
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return idx

    def get(self, s: float) -> Tuple[int, float, int]:
        """Sample a leaf proportional to priority (iterative)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left] or right >= len(self.tree):
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), data_idx

    def batch_get(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised batch sampling — sample N leaves at once."""
        n = len(values)
        tree_indices = np.empty(n, dtype=np.int64)
        priorities = np.empty(n, dtype=np.float64)
        data_indices = np.empty(n, dtype=np.int64)
        for i in range(n):
            tree_indices[i], priorities[i], data_indices[i] = self.get(values[i])
        return tree_indices, priorities, data_indices

    def batch_update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Batch priority update — much faster than individual updates."""
        for idx, p in zip(indices, priorities):
            self.update(int(idx), float(p))

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_leaf(self) -> float:
        leaves = self.tree[self.capacity - 1:]
        return float(np.max(leaves)) if len(leaves) > 0 else 1.0


# ── N-step transition accumulator ─────────────────────────────────────────────
class NStepBuffer:
    """Per-environment buffer that accumulates n-step returns.

    Instead of storing (s, a, r, s'), stores (s, a, R_n, s_{t+n}) where
    R_n = r₁ + γr₂ + γ²r₃. This propagates reward signals n steps back.
    Returns are clamped to [-60, 60] for numerical stability.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99) -> None:
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n)

    def append(self, transition: tuple) -> Optional[tuple]:
        """Append a transition, return a completed n-step transition if ready.

        transition: (sg, sv, a, r, ng, nv, done, teacher)
        returns:    (sg, sv, a, R_n, ng_n, nv_n, done_n, teacher) or None
        """
        self.buffer.append(transition)

        if len(self.buffer) < self.n:
            return None

        # Compute n-step return
        R = 0.0
        for i in reversed(range(self.n)):
            sg, sv, a, r, ng, nv, done, teacher = self.buffer[i]
            R = r + self.gamma * R * (1.0 - float(done))
            if done:
                # If episode ended within n steps, truncate
                sg0, sv0, a0, _, _, _, _, teacher0 = self.buffer[0]
                R = max(-60.0, min(60.0, R))  # clamp for stability
                return (sg0, sv0, a0, R, ng, nv, done, teacher0)

        sg0, sv0, a0, _, _, _, _, teacher0 = self.buffer[0]
        _, _, _, _, ng_n, nv_n, done_n, _ = self.buffer[-1]
        R = max(-60.0, min(60.0, R))  # clamp for stability
        return (sg0, sv0, a0, R, ng_n, nv_n, done_n, teacher0)

    def flush(self) -> List[tuple]:
        """Flush remaining transitions at episode end (with shorter n-step)."""
        results = []
        while len(self.buffer) > 0:
            R = 0.0
            for i in reversed(range(len(self.buffer))):
                sg, sv, a, r, ng, nv, done, teacher = self.buffer[i]
                R = r + self.gamma * R * (1.0 - float(done))

            sg0, sv0, a0, _, _, _, _, teacher0 = self.buffer[0]
            _, _, _, _, ng_last, nv_last, done_last, _ = self.buffer[-1]
            R = max(-60.0, min(60.0, R))  # clamp for stability
            results.append((sg0, sv0, a0, R, ng_last, nv_last, done_last, teacher0))
            self.buffer.popleft()
        return results


# ── Prioritized Experience Replay Buffer ──────────────────────────────────────
class PrioritizedReplayBuffer:
    """SumTree-based PER with importance sampling and n-step returns.

    Key stability features:
      - TD error clamped to [-100, 100] before priority computation
      - Max priority capped at 100.0 to prevent runaway feedback loops
      - IS weights normalized to [0, 1]
      - N-step returns clamped to [-60, 60]
    """

    # Maximum allowed priority to prevent runaway feedback loops
    MAX_PRIORITY_CAP = 100.0

    def __init__(self, cap: int = 300_000, alpha: float = 0.5,
                 beta_start: float = 0.4, n_step: int = 3,
                 gamma: float = 0.99) -> None:
        self._cap   = cap
        self._ptr   = 0
        self._size  = 0
        self.alpha  = alpha        # priority exponent (0=uniform, 1=full PER)
        self.beta   = beta_start   # IS weight exponent (annealed to 1.0)
        self._eps   = 1e-6         # small constant to avoid zero priority
        self.n_step = n_step
        self.gamma  = gamma
        self._gamma_n = gamma ** n_step  # precomputed for n-step target

        self.tree = SumTree(cap)

        pin = torch.cuda.is_available()
        self._sg = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16, pin_memory=pin)
        self._sv = torch.zeros((cap, INV_FEATURES),          dtype=torch.float32, pin_memory=pin)
        self._a  = torch.zeros(cap,                          dtype=torch.int64,   pin_memory=pin)
        self._r  = torch.zeros(cap,                          dtype=torch.float32, pin_memory=pin)
        self._ng = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16, pin_memory=pin)
        self._nv = torch.zeros((cap, INV_FEATURES),          dtype=torch.float32, pin_memory=pin)
        self._d  = torch.zeros(cap,                          dtype=torch.float32, pin_memory=pin)
        self._t  = torch.zeros(cap,                          dtype=torch.int64,   pin_memory=pin)
        self._tm = torch.zeros(cap,                          dtype=torch.float32, pin_memory=pin)

        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    @property
    def gamma_n(self) -> float:
        return self._gamma_n

    def push(self, sg: np.ndarray, sv: np.ndarray, a: int, r: float,
             ng: np.ndarray, nv: np.ndarray, done: bool,
             teacher: Optional[int] = None) -> None:
        """Push a single transition with max priority."""
        i = self._ptr
        r = float(np.clip(r, -60.0, 60.0))
        self._sg[i] = torch.from_numpy(sg).to(torch.float16)
        self._sv[i] = torch.from_numpy(sv)
        self._a[i]  = a
        self._r[i]  = r
        self._ng[i] = torch.from_numpy(ng).to(torch.float16)
        self._nv[i] = torch.from_numpy(nv)
        self._d[i]  = float(done)
        self._t[i]  = int(teacher if teacher is not None else 0)
        self._tm[i] = 1.0 if teacher is not None else 0.0

        # New transitions get max priority (guaranteed to be sampled)
        priority = min(self._max_priority, self.MAX_PRIORITY_CAP) ** self.alpha
        self.tree.add(priority)

        self._ptr  = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def push_batch(self, transitions: list) -> None:
        """Push multiple transitions at once."""
        for sg, sv, a, r, ng, nv, done, teacher in transitions:
            self.push(sg, sv, a, r, ng, nv, done, teacher)

    def sample(self, n: int, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, np.ndarray,
    ]:
        """Sample a prioritized batch with importance sampling weights.

        Returns the usual 9 tensors plus:
          - weights: IS correction weights (n,)
          - tree_indices: for priority update after loss computation
        """
        total = max(self.tree.total, self._eps)
        segment = total / n

        # Stratified sampling: one sample per segment
        lo = np.arange(n, dtype=np.float64) * segment
        hi = lo + segment
        values = np.random.uniform(lo, hi)

        tree_indices, priorities, data_indices = self.tree.batch_get(values)

        # Clamp data indices to valid range
        data_indices = np.clip(data_indices, 0, self._size - 1)
        priorities = np.maximum(priorities, self._eps)

        # Importance sampling weights
        probs = priorities / total
        weights = (self._size * probs) ** (-self.beta)
        weights /= max(weights.max(), self._eps)  # normalize to [0, 1]

        idx = torch.from_numpy(data_indices.astype(np.int64)).long()
        w   = torch.from_numpy(weights.astype(np.float32)).to(device, non_blocking=True)

        nb = {"non_blocking": True}
        return (
            self._sg[idx].float().to(device, **nb),
            self._sv[idx]        .to(device, **nb),
            self._a[idx].unsqueeze(1).to(device, **nb),
            self._r[idx]         .to(device, **nb),
            self._ng[idx].float().to(device, **nb),
            self._nv[idx]        .to(device, **nb),
            self._d[idx]         .to(device, **nb),
            self._t[idx]         .to(device, **nb),
            self._tm[idx]        .to(device, **nb),
            w,
            tree_indices,
        )

    def update_priorities(self, tree_indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        """Update priorities after computing TD errors.

        TD errors are clamped to [-100, 100] to prevent runaway feedback loops
        where high prediction error → high priority → more sampling →
        even higher prediction error → divergence.
        """
        # CRITICAL: clamp TD errors to prevent priority explosion
        clamped = np.clip(np.abs(td_errors), 0.0, 100.0)
        priorities = (clamped + self._eps) ** self.alpha

        # Batch update (faster than individual updates)
        self.tree.batch_update(tree_indices, priorities)

        # Update max priority with cap
        max_td = float(clamped.max())
        self._max_priority = min(max_td + self._eps, self.MAX_PRIORITY_CAP)

    def anneal_beta(self, progress: float) -> None:
        """Anneal beta from beta_start toward 1.0 based on training progress."""
        self.beta = min(1.0, 0.4 + progress * 0.6)


# ── Legacy alias for backward compatibility ───────────────────────────────────
ReplayBuffer = PrioritizedReplayBuffer
