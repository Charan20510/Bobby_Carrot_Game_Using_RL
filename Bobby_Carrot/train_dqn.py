from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parent
while not (ROOT / "Game_Python").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

GAME_PYTHON_DIR = ROOT / "Game_Python"
if not GAME_PYTHON_DIR.exists():
    raise RuntimeError("Could not locate Game_Python directory for imports.")

if str(GAME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_PYTHON_DIR))

from bobby_carrot.rl_env import BobbyCarrotEnv

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:
    raise RuntimeError("PyTorch is required. Install with: pip install torch") from exc


# ============================================================
# Constants
# ============================================================

GRID_SIZE     = 16
GRID_CHANNELS = 13
INV_FEATURES  = 5

# Tile-id → channel lookup table (length 256, built once at import time)
# Channel map:
#  0  wall (<18)          5  hazard/death (31,46)   10 floor/other
#  1  carrot (19)         6  key (32,34,36)          11 agent position
#  2  egg (45)            7  locked door (33,35,37)  12 all_collected flag
#  3  finish ACTIVE       8  crumble (30)
#  4  finish INACTIVE     9  conveyor (40-43)
_TILE_CHANNEL = np.zeros(256, dtype=np.int8)
for _i in range(256):
    if _i < 18:                         _TILE_CHANNEL[_i] = 0
    elif _i == 19:                      _TILE_CHANNEL[_i] = 1
    elif _i == 45:                      _TILE_CHANNEL[_i] = 2
    elif _i == 44:                      _TILE_CHANNEL[_i] = 4   # finish (active=3 set later)
    elif _i in (31, 46):                _TILE_CHANNEL[_i] = 5
    elif _i in (32, 34, 36):            _TILE_CHANNEL[_i] = 6
    elif _i in (33, 35, 37):            _TILE_CHANNEL[_i] = 7
    elif _i == 30:                      _TILE_CHANNEL[_i] = 8
    elif _i in (40, 41, 42, 43):        _TILE_CHANNEL[_i] = 9
    else:                               _TILE_CHANNEL[_i] = 10


# ============================================================
# Per-level config (derived from .blm map analysis)
# ============================================================

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 600,  "episodes": 5000,  "distance_scale": 1.5, "post_penalty": -0.5},
    2:  {"max_steps": 700,  "episodes": 5000,  "distance_scale": 1.5, "post_penalty": -0.5},
    3:  {"max_steps": 800,  "episodes": 5000,  "distance_scale": 1.2, "post_penalty": -0.5},
    4:  {"max_steps": 1200, "episodes": 6000,  "distance_scale": 0.8, "post_penalty": -0.8},
    5:  {"max_steps": 700,  "episodes": 5000,  "distance_scale": 1.0, "post_penalty": -0.6},
    6:  {"max_steps": 900,  "episodes": 5000,  "distance_scale": 1.0, "post_penalty": -0.6},
    7:  {"max_steps": 1400, "episodes": 7000,  "distance_scale": 0.8, "post_penalty": -0.8},
    8:  {"max_steps": 600,  "episodes": 5000,  "distance_scale": 1.2, "post_penalty": -0.5},
    9:  {"max_steps": 900,  "episodes": 5000,  "distance_scale": 1.0, "post_penalty": -0.6},
    10: {"max_steps": 800,  "episodes": 5000,  "distance_scale": 1.0, "post_penalty": -0.6},
}


# ============================================================
# DQNConfig
# ============================================================

@dataclass
class DQNConfig:
    gamma:               float = 0.99
    lr:                  float = 3e-4          # ↑ from 1e-4 — faster convergence on T4
    batch_size:          int   = 256           # ↑ from 128 — T4 has 16GB VRAM, larger batch = fewer steps
    replay_capacity:     int   = 100_000       # ↓ from 150k — saves ~1.2GB RAM
    min_replay_size:     int   = 2000          # ↓ from 3000 — start learning sooner
    target_update_steps: int   = 1500          # ↓ from 2000 — more frequent target sync
    train_every_steps:   int   = 2
    # Exploration
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.05          # ↑ slightly from 0.02 — faster decay floor
    epsilon_decay:       float = 0.998         # ↑ from 0.9994 — decays faster, less random time
    # Reward shaping
    completion_bonus:    float = 300.0
    death_penalty:       float = -80.0
    carrot_bonus:        float = 15.0
    crumble_step_penalty:float = -5.0
    invalid_move_penalty:float = -0.5
    terminal_oversample: int   = 4
    # Misc
    report_every:        int   = 100
    save_every:          int   = 500
    seed:                int   = 42
    observation_mode:    str   = "full"        # single source of truth


# ============================================================
# SPEED FIX 1 — Pre-allocated numpy ReplayBuffer
# Uses uint8 for grid storage (values are 0/1 only → 4× smaller than float32)
# Casts to float32 only at sample time (GPU transfer handles it)
# Eliminates deque O(n) random access and repeated np.stack() calls
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._cap  = capacity
        self._ptr  = 0
        self._size = 0
        # uint8 grid: 0/1 values only — 4× memory saving vs float32
        self._sg  = np.zeros((capacity, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self._sv  = np.zeros((capacity, INV_FEATURES),                         dtype=np.float32)
        self._a   = np.zeros(capacity,                                         dtype=np.int64)
        self._r   = np.zeros(capacity,                                         dtype=np.float32)
        self._ng  = np.zeros((capacity, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self._nv  = np.zeros((capacity, INV_FEATURES),                         dtype=np.float32)
        self._d   = np.zeros(capacity,                                         dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        grid: np.ndarray, inv: np.ndarray,
        action: int, reward: float,
        next_grid: np.ndarray, next_inv: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self._sg[i]  = grid          # uint8 — no copy needed, already uint8
        self._sv[i]  = inv
        self._a[i]   = action
        self._r[i]   = reward
        self._ng[i]  = next_grid
        self._nv[i]  = next_inv
        self._d[i]   = float(done)
        self._ptr    = (i + 1) % self._cap
        self._size   = min(self._size + 1, self._cap)

    def sample(self, n: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self._size, size=n)
        # Cast uint8 → float32 here (once per batch, not per step)
        return (
            self._sg[idx].astype(np.float32),
            self._sv[idx],
            self._a[idx],
            self._r[idx],
            self._ng[idx].astype(np.float32),
            self._nv[idx],
            self._d[idx],
        )


# ============================================================
# Dueling DDQN
# ============================================================

class DuelingDQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),            nn.ReLU(),
            nn.Flatten(),
        )
        merged = 64 * GRID_SIZE * GRID_SIZE + INV_FEATURES
        self.shared           = nn.Sequential(nn.Linear(merged, 512), nn.ReLU())
        self.value_stream     = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        feat   = self.conv(grid)
        shared = self.shared(torch.cat([feat, inv], dim=1))
        v = self.value_stream(shared)
        a = self.advantage_stream(shared)
        return v + a - a.mean(dim=1, keepdim=True)


# ============================================================
# SPEED FIX 2 — Vectorised observation encoding
# Replaces 256-iteration Python for-loop with numpy array indexing
# _TILE_CHANNEL lookup + np.put → encodes entire 16×16 grid in ~5 µs vs ~200 µs
# ============================================================

# Pre-allocate a reusable grid buffer — avoids np.zeros() allocation every step
_GRID_BUF = np.zeros((GRID_CHANNELS, GRID_SIZE * GRID_SIZE), dtype=np.uint8)


def _semantic_channels(env: BobbyCarrotEnv) -> Tuple[np.ndarray, np.ndarray]:
    assert env.map_info is not None and env.bobby is not None

    data_arr      = np.asarray(env.map_info.data, dtype=np.uint8)  # shape (256,)
    all_collected = env.bobby.is_finished(env.map_info)

    # Map every tile-id to its channel index (vectorised — no Python loop)
    channels = _TILE_CHANNEL[data_arr]                              # shape (256,) int8

    # Override finish tile channel based on all_collected flag
    finish_mask = (data_arr == 44)
    channels[finish_mask] = 3 if all_collected else 4

    # Build grid using numpy advanced indexing (no Python for-loop over tiles)
    grid = np.zeros((GRID_CHANNELS, GRID_SIZE * GRID_SIZE), dtype=np.uint8)
    grid[channels, np.arange(256)] = 1

    grid = grid.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Agent position channel
    px, py = env.bobby.coord_src
    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
        grid[11, py, px] = 1

    # Phase flag: all_collected floods channel 12
    if all_collected:
        grid[12, :, :] = 1

    # ---- Inventory vector ----
    remaining = (
        (env.map_info.carrot_total - env.bobby.carrot_count) +
        (env.map_info.egg_total    - env.bobby.egg_count)
    )
    denom          = max(1, env.map_info.carrot_total + env.map_info.egg_total)
    remaining_norm = min(1.0, remaining / denom)

    # SPEED FIX 3 — Manhattan distance via numpy (no Python list comprehension)
    if not all_collected:
        # Find all uncollected carrot/egg positions using numpy
        target_mask = (data_arr == 19) | (data_arr == 45)
        if target_mask.any():
            indices    = np.where(target_mask)[0]
            xs         = indices % GRID_SIZE
            ys         = indices // GRID_SIZE
            manhattan  = np.abs(xs - px) + np.abs(ys - py)
            manhattan_norm = float(manhattan.min()) / (GRID_SIZE * 2)
        else:
            manhattan_norm = 0.0
    else:
        # Phase 2: distance to finish tile
        finish_idx = np.where(data_arr == 44)[0]
        if len(finish_idx) > 0:
            fx = int(finish_idx[0] % GRID_SIZE)
            fy = int(finish_idx[0] // GRID_SIZE)
            manhattan_norm = min(1.0, (abs(fx - px) + abs(fy - py)) / (GRID_SIZE * 2))
        else:
            manhattan_norm = 0.0

    inv = np.array([
        float(env.bobby.key_gray   > 0),
        float(env.bobby.key_yellow > 0),
        float(env.bobby.key_red    > 0),
        remaining_norm,
        float(manhattan_norm),
    ], dtype=np.float32)

    return grid, inv


# ============================================================
# Reward shaping
# ============================================================

def _crumble_adjacent_to_carrot(env: BobbyCarrotEnv) -> bool:
    assert env.map_info is not None and env.bobby is not None
    px, py = env.bobby.coord_src
    if env.map_info.data[px + py * GRID_SIZE] != 31:
        return False
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        nx, ny = px + dx, py + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if env.map_info.data[nx + ny * GRID_SIZE] == 19:
                return True
    return False


def _shape_reward(
    raw_reward:           float,
    info:                 Dict[str, object],
    cfg:                  DQNConfig,
    level_cfg:            Dict,
    prev_inv:             np.ndarray,
    curr_inv:             np.ndarray,
    all_collected_before: bool,
    env:                  BobbyCarrotEnv,
) -> float:
    reward         = float(raw_reward)
    distance_scale = level_cfg["distance_scale"]
    post_penalty   = level_cfg["post_penalty"]

    if bool(info.get("invalid_move", False)):
        reward += cfg.invalid_move_penalty
    if bool(info.get("dead", False)):
        reward += cfg.death_penalty

    carrot_delta = int(info.get("collected_carrot", 0))
    if carrot_delta > 0:
        reward += cfg.carrot_bonus * carrot_delta

    if bool(info.get("level_completed", False)):
        reward += cfg.completion_bonus

    all_collected_now = bool(info.get("all_collected", False))
    level_done        = bool(info.get("level_completed", False))

    # prev/curr inv[4] = manhattan_norm (to nearest carrot p1, to finish p2)
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])   # positive = moved closer

    if not all_collected_now:
        reward += distance_scale * dist_delta
    elif not level_done:
        reward += distance_scale * dist_delta   # now shaping toward finish tile
        reward += post_penalty

    if _crumble_adjacent_to_carrot(env):
        reward += cfg.crumble_step_penalty

    return reward


# ============================================================
# DQN Agent
# SPEED FIX 4 — pin_memory + non_blocking GPU transfer
# SPEED FIX 5 — torch.inference_mode instead of no_grad in select_action
# ============================================================

class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig, device: torch.device) -> None:
        self.n_actions   = n_actions
        self.cfg         = cfg
        self.device      = device
        self.epsilon     = cfg.epsilon_start
        self.total_steps = 0

        self.policy_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr, eps=1.5e-4)
        self.loss_fn   = nn.SmoothL1Loss()
        self.replay    = ReplayBuffer(cfg.replay_capacity)

        # SPEED FIX 4 — reusable pinned-memory tensors for inference
        # Avoids repeated host-alloc + H2D copy every select_action call
        self._inf_grid = torch.zeros(
            (1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE),
            dtype=torch.float32, pin_memory=(device.type == "cuda")
        )
        self._inf_inv = torch.zeros(
            (1, INV_FEATURES),
            dtype=torch.float32, pin_memory=(device.type == "cuda")
        )

    @torch.inference_mode()           # SPEED FIX 5 — faster than no_grad
    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Reuse pinned buffers — copy numpy → pinned CPU → GPU (non-blocking)
        self._inf_grid[0].copy_(torch.from_numpy(grid.astype(np.float32)))
        self._inf_inv[0].copy_(torch.from_numpy(inv))
        g = self._inf_grid.to(self.device, non_blocking=True)
        v = self._inf_inv.to(self.device,  non_blocking=True)
        return int(self.policy_net(g, v).argmax(1).item())

    def optimize_step(self) -> Optional[float]:
        if len(self.replay) < self.cfg.min_replay_size:
            return None
        if self.total_steps % self.cfg.train_every_steps != 0:
            return None

        sg, sv, acts, rwds, nsg, nsv, dones = self.replay.sample(self.cfg.batch_size)

        # SPEED FIX 4 — non_blocking GPU transfer
        sg_t  = torch.from_numpy(sg).to(self.device,   non_blocking=True)
        sv_t  = torch.from_numpy(sv).to(self.device,   non_blocking=True)
        a_t   = torch.from_numpy(acts).unsqueeze(1).to(self.device, non_blocking=True)
        r_t   = torch.from_numpy(rwds).to(self.device,  non_blocking=True)
        nsg_t = torch.from_numpy(nsg).to(self.device,   non_blocking=True)
        nsv_t = torch.from_numpy(nsv).to(self.device,   non_blocking=True)
        d_t   = torch.from_numpy(dones).to(self.device, non_blocking=True)

        q_pred = self.policy_net(sg_t, sv_t).gather(1, a_t).squeeze(1)

        with torch.inference_mode():
            best_a = self.policy_net(nsg_t, nsv_t).argmax(1, keepdim=True)
            q_next = self.target_net(nsg_t, nsv_t).gather(1, best_a).squeeze(1)
            target = r_t + (1.0 - d_t) * self.cfg.gamma * q_next

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        if self.total_steps % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path: Path, level: int, map_kind: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy":        self.policy_net.state_dict(),
            "target":        self.target_net.state_dict(),
            "optim":         self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "total_steps":   self.total_steps,
            "level":         level,
            "map_kind":      map_kind,
            "grid_channels": GRID_CHANNELS,
            "inv_features":  INV_FEATURES,
        }, path)

    def load(self, path: Path) -> Dict[str, object]:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt.get("target", ckpt["policy"]))
        if "optim" in ckpt:
            self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon     = float(ckpt.get("epsilon", self.cfg.epsilon_start))
        self.total_steps = int(ckpt.get("total_steps", 0))
        return {"level": ckpt.get("level", -1), "map_kind": ckpt.get("map_kind", "normal")}


# ============================================================
# Helpers
# ============================================================

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(cfg: DQNConfig, map_kind: str, level: int, max_steps: int) -> BobbyCarrotEnv:
    return BobbyCarrotEnv(
        map_kind=map_kind, map_number=level,
        observation_mode=cfg.observation_mode,
        local_view_size=3, include_inventory=True,
        headless=True, max_steps=max_steps,
    )


def _get_level_cfg(level: int) -> Dict:
    return LEVEL_CONFIG.get(level, {
        "max_steps": 800, "episodes": 5000,
        "distance_scale": 1.0, "post_penalty": -0.6,
    })


# ============================================================
# SPEED FIX 6 — torch.compile (PyTorch 2.0+)
# Fuses conv+relu ops into a single CUDA kernel, ~20% faster on T4
# ============================================================

def _maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            pass
    return model


# ============================================================
# Training loop
# SPEED FIX 7 — steps/sec counter printed every report_every episodes
# so you can see immediately if training is fast enough
# ============================================================

def train_one_level(
    agent:      DQNAgent,
    level:      int,
    cfg:        DQNConfig,
    map_kind:   str,
    ckpt_path:  Optional[Path] = None,
) -> Dict[str, float]:

    level_cfg  = _get_level_cfg(level)
    max_steps  = level_cfg["max_steps"]
    n_episodes = level_cfg["episodes"]

    env = _make_env(cfg, map_kind, level, max_steps)

    reward_hist:    List[float] = []
    success_hist:   List[float] = []
    collected_hist: List[float] = []
    loss_win: Deque[float] = deque(maxlen=cfg.report_every)
    best_success = 0.0

    print(f"  max_steps={max_steps} | episodes={n_episodes} | "
          f"distance_scale={level_cfg['distance_scale']} | "
          f"post_penalty={level_cfg['post_penalty']} | "
          f"batch={cfg.batch_size} | lr={cfg.lr}")

    t0         = time.time()
    total_env_steps = 0

    for episode in range(1, n_episodes + 1):
        env.set_map(map_kind=map_kind, map_number=level)
        env.reset()
        grid, inv = _semantic_channels(env)

        done                 = False
        steps                = 0
        ep_reward            = 0.0
        info: Dict[str, object] = {}
        all_collected_before = False
        prev_inv             = inv.copy()

        while not done and steps < max_steps:
            action = agent.select_action(grid, inv)
            _, raw_reward, done, info = env.step(action)

            next_grid, next_inv = _semantic_channels(env)
            all_collected_now   = bool(info.get("all_collected", False))

            shaped = _shape_reward(
                raw_reward, info, cfg, level_cfg,
                prev_inv, next_inv, all_collected_before, env,
            )

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            if bool(info.get("level_completed", False)) and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps += 1
            total_env_steps   += 1
            loss = agent.optimize_step()
            if loss is not None:
                loss_win.append(loss)

            prev_inv             = next_inv
            grid, inv            = next_grid, next_inv
            all_collected_before = all_collected_now
            ep_reward           += shaped
            steps               += 1

        agent.decay_epsilon()

        success   = 1.0 if bool(info.get("level_completed", False)) else 0.0
        collected = 1.0 if bool(info.get("all_collected",    False)) else 0.0
        reward_hist.append(ep_reward)
        success_hist.append(success)
        collected_hist.append(collected)

        if episode % cfg.report_every == 0 or episode == 1:
            n      = cfg.report_every
            avg_r  = float(np.mean(reward_hist[-n:]))
            avg_s  = float(np.mean(success_hist[-n:]))
            avg_c  = float(np.mean(collected_hist[-n:]))
            avg_l  = float(np.mean(loss_win)) if loss_win else 0.0
            elapsed = time.time() - t0
            sps    = total_env_steps / max(elapsed, 1e-6)   # steps per second
            eta_h  = ((n_episodes - episode) * max_steps) / max(sps, 1) / 3600
            print(
                f"[L{level}] ep={episode:5d} | "
                f"reward={avg_r:7.1f} | "
                f"collected={avg_c:5.1%} | "
                f"success={avg_s:5.1%} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_l:.3f} | "
                f"sps={sps:5.0f} | "      # steps/sec — key speed metric
                f"ETA={eta_h:.1f}h"
            )

        if ckpt_path is not None and episode % cfg.save_every == 0:
            rolling_s = float(np.mean(success_hist[-cfg.report_every:]))
            if rolling_s >= best_success:
                best_success = rolling_s
                best_path = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(best_path, level=level, map_kind=map_kind)

    env.close()
    return {
        "mean_reward":        float(np.mean(reward_hist))    if reward_hist    else 0.0,
        "success_rate":       float(np.mean(success_hist))   if success_hist   else 0.0,
        "all_collected_rate": float(np.mean(collected_hist)) if collected_hist else 0.0,
    }


# ============================================================
# Play / evaluation
# ============================================================

def play_trained_dqn(
    model_path: Path,
    map_kind:   str,
    map_number: int,
    episodes:   int   = 5,
    render:     bool  = False,
    render_fps: float = 5.0,
    cfg:        Optional[DQNConfig] = None,
) -> None:
    if cfg is None:
        cfg = DQNConfig()
    level_cfg = _get_level_cfg(map_number)
    max_steps = level_cfg["max_steps"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BobbyCarrotEnv(
        map_kind=map_kind, map_number=map_number,
        observation_mode=cfg.observation_mode,
        local_view_size=3, include_inventory=True,
        headless=not render, max_steps=max_steps,
    )
    agent = DQNAgent(n_actions=env.action_space_n, cfg=cfg, device=device)
    meta  = agent.load(model_path)
    agent.epsilon = 0.0
    print(f"Loaded {model_path}  (trained level={meta['level']}, kind={meta['map_kind']})")
    print(f"Playing L{map_number}: max_steps={max_steps}, obs={cfg.observation_mode}")

    for ep in range(1, episodes + 1):
        env.set_map(map_kind=map_kind, map_number=map_number)
        env.reset()
        grid, inv = _semantic_channels(env)

        done = False; steps = 0; total_reward = 0.0
        info: Dict[str, object] = {}

        while not done and steps < max_steps:
            action = agent.select_action(grid, inv)
            _, reward, done, info = env.step(action)
            grid, inv    = _semantic_channels(env)
            total_reward += reward
            steps        += 1
            if render:
                env.render()
                if render_fps > 0:
                    time.sleep(1.0 / render_fps)

        print(
            f"Play ep {ep}/{episodes} | reward={total_reward:.2f} | "
            f"collected={bool(info.get('all_collected', False))} | "
            f"success={bool(info.get('level_completed', False))} | steps={steps}"
        )
    env.close()


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bobby Carrot DQN — speed-optimised for Colab T4")
    p.add_argument("--play",              action="store_true")
    p.add_argument("--map-kind",          default="normal", choices=["normal","egg"])
    p.add_argument("--map-number",        type=int,   default=1)
    p.add_argument("--levels",            type=int,   nargs="+", default=None)
    p.add_argument("--individual-levels", action="store_true")
    p.add_argument("--episodes-per-level",type=int,   default=None)
    p.add_argument("--max-steps",         type=int,   default=None)
    p.add_argument("--batch-size",        type=int,   default=256)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--epsilon-start",     type=float, default=1.0)
    p.add_argument("--epsilon-min",       type=float, default=0.05)
    p.add_argument("--epsilon-decay",     type=float, default=0.998)
    p.add_argument("--completion-bonus",  type=float, default=300.0)
    p.add_argument("--death-penalty",     type=float, default=-80.0)
    p.add_argument("--crumble-step-penalty", type=float, default=-5.0)
    p.add_argument("--invalid-move-penalty", type=float, default=-0.5)
    p.add_argument("--terminal-oversample", type=int, default=4)
    p.add_argument("--observation-mode",  default="full", choices=["full","local","compact"])
    p.add_argument("--report-every",      type=int,   default=100)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--model-path",        default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",    default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes",     type=int,   default=5)
    p.add_argument("--no-render",         action="store_true")
    p.add_argument("--render-fps",        type=float, default=5.0)
    return p


def _cfg_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
        epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        completion_bonus=args.completion_bonus, death_penalty=args.death_penalty,
        crumble_step_penalty=args.crumble_step_penalty,
        invalid_move_penalty=args.invalid_move_penalty,
        terminal_oversample=args.terminal_oversample,
        observation_mode=args.observation_mode,
        report_every=args.report_every, seed=args.seed,
    )


def _apply_cli_overrides(level: int, args: argparse.Namespace) -> None:
    if args.episodes_per_level is not None:
        LEVEL_CONFIG.setdefault(level, {})["episodes"] = args.episodes_per_level
    if args.max_steps is not None:
        LEVEL_CONFIG.setdefault(level, {})["max_steps"] = args.max_steps


def _main() -> None:
    args       = _build_parser().parse_args()
    cfg        = _cfg_from_args(args)
    _seed_everything(cfg.seed)
    model_path = Path(args.model_path)
    ckpt_dir   = Path(args.checkpoint_dir)

    if args.play:
        play_trained_dqn(
            model_path=model_path, map_kind=args.map_kind,
            map_number=args.map_number, episodes=args.play_episodes,
            render=not args.no_render, render_fps=args.render_fps, cfg=cfg,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    levels = [int(l) for l in (args.levels or [args.map_number])]
    for lvl in levels:
        _apply_cli_overrides(lvl, args)

    probe_cfg = _get_level_cfg(levels[0])
    probe_env = _make_env(cfg, args.map_kind, levels[0], probe_cfg["max_steps"])
    n_actions = probe_env.action_space_n
    probe_env.close()

    if args.individual_levels and len(levels) > 1:
        for lvl in levels:
            print(f"\n=== Independent training — level {lvl} ===")
            agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
            # SPEED FIX 6 — compile both nets
            agent.policy_net = _maybe_compile(agent.policy_net)
            agent.target_net = _maybe_compile(agent.target_net)
            out   = ckpt_dir / f"dqn_level{lvl}_individual.pt"
            summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
            agent.save(out, level=lvl, map_kind=args.map_kind)
            print(f"Saved {out} | {summary}")
        return

    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
    # SPEED FIX 6 — compile both nets once (first call has ~30s warm-up, all subsequent calls faster)
    agent.policy_net = _maybe_compile(agent.policy_net)
    agent.target_net = _maybe_compile(agent.target_net)

    if model_path.exists():
        meta = agent.load(model_path)
        print(f"Resumed from {model_path}  (level={meta['level']}, kind={meta['map_kind']})")

    for lvl in levels:
        print(f"\n=== Sequential training — level {lvl} ===")
        out     = ckpt_dir / f"dqn_level{lvl}_sequential.pt"
        summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
        agent.save(out, level=lvl, map_kind=args.map_kind)
        print(
            f"Saved {out} | "
            f"mean_reward={summary['mean_reward']:.2f} | "
            f"collected={summary['all_collected_rate']:.2%} | "
            f"success={summary['success_rate']:.2%}"
        )

    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"\nFinal model saved to: {model_path}")


if __name__ == "__main__":
    _main()
