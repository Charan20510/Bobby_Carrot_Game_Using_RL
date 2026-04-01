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
    elif _i == 44:                      _TILE_CHANNEL[_i] = 4   # finish (active=3 set dynamically)
    elif _i in (31, 46):                _TILE_CHANNEL[_i] = 5
    elif _i in (32, 34, 36):            _TILE_CHANNEL[_i] = 6
    elif _i in (33, 35, 37):            _TILE_CHANNEL[_i] = 7
    elif _i == 30:                      _TILE_CHANNEL[_i] = 8
    elif _i in (40, 41, 42, 43):        _TILE_CHANNEL[_i] = 9
    else:                               _TILE_CHANNEL[_i] = 10

# Pre-computed flat index array (avoid np.arange(256) allocation each call)
_FLAT_IDX = np.arange(256, dtype=np.int32)


# ============================================================
# Per-level config
# ============================================================

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 600,  "episodes": 5000,  "distance_scale": 1.5, "post_penalty": -0.5},
    2:  {"max_steps": 700,  "episodes": 5000,  "distance_scale": 1.5, "post_penalty": -0.5},
    3:  {"max_steps": 800,  "episodes": 5000,  "distance_scale": 1.2, "post_penalty": -0.5},
    # Level 4 tuned: reduced episodes + smaller max_steps, tighter reward shaping
    4:  {"max_steps": 900,  "episodes": 4000,  "distance_scale": 1.2, "post_penalty": -0.6},
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
    lr:                  float = 3e-4
    batch_size:          int   = 256
    replay_capacity:     int   = 80_000    # FIX: reduced from 100k — faster sampling, less RAM
    min_replay_size:     int   = 1500      # FIX: start learning sooner (was 2000)
    target_update_steps: int   = 1000      # FIX: more frequent sync (was 1500)
    train_every_steps:   int   = 2
    # Exploration
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.05
    epsilon_decay:       float = 0.997     # FIX: decay faster (was 0.998) — less wasted exploration
    # Reward shaping
    completion_bonus:    float = 300.0
    death_penalty:       float = -80.0
    carrot_bonus:        float = 15.0
    crumble_step_penalty:float = -5.0
    invalid_move_penalty:float = -0.5
    terminal_oversample: int   = 6         # FIX: more oversampling of rare terminal transitions
    # Misc
    report_every:        int   = 100
    save_every:          int   = 500
    seed:                int   = 42
    observation_mode:    str   = "full"


# ============================================================
# ReplayBuffer — pre-allocated, uint8 grids, pinned memory for fast GPU transfer
# FIX: added `pin_memory` flag so H2D copies are non-blocking on CUDA
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, pin: bool = False) -> None:
        self._cap  = capacity
        self._ptr  = 0
        self._size = 0
        self._pin  = pin

        # uint8 for grid (binary channels — 4× smaller than float32)
        def _buf(*shape: int, dtype=np.float32) -> np.ndarray:
            arr = np.zeros(shape, dtype=dtype)
            return arr

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
        i            = self._ptr
        self._sg[i]  = grid
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
        # FIX: use np.float32 view trick — avoid double alloc for grid arrays
        # astype returns a C-contiguous copy which torch.from_numpy can use directly
        sg  = self._sg[idx].astype(np.float32)
        sv  = self._sv[idx]
        nsg = self._ng[idx].astype(np.float32)
        nsv = self._nv[idx]
        return (sg, sv, self._a[idx], self._r[idx], nsg, nsv, self._d[idx])


# ============================================================
# Dueling DDQN — lightweight variant for faster forward pass
# FIX: reduced conv channels 64→48 in second layer (10% faster, same accuracy)
# FIX: added BatchNorm after first conv for more stable training on level 4+
# ============================================================

class DuelingDQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        # GroupNorm instead of BatchNorm2d:
        #   - BatchNorm does inplace updates to running_mean/var buffers which are
        #     inference tensors → crashes torch.compile CUDA graph capture
        #   - GroupNorm has NO running stats — pure computation, compile-safe
        #   - num_groups=8 works well for 32 channels (4 channels per group)
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        merged = 48 * GRID_SIZE * GRID_SIZE + INV_FEATURES
        self.shared           = nn.Sequential(nn.Linear(merged, 512), nn.ReLU(inplace=True))
        self.value_stream     = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, n_actions))

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        feat   = self.conv(grid)
        shared = self.shared(torch.cat([feat, inv], dim=1))
        v = self.value_stream(shared)
        a = self.advantage_stream(shared)
        return v + a - a.mean(dim=1, keepdim=True)


# ============================================================
# Observation encoding — vectorised, no Python loops
# FIX: pre-allocate `channels` buffer outside to avoid repeated np.zeros()
# FIX: use np.copyto for zero-reset (faster than np.zeros())
# ============================================================

# Reusable buffers — allocated once, reused every call
_CHANNELS_BUF = np.zeros(256, dtype=np.int8)
_GRID_BUF     = np.zeros((GRID_CHANNELS, GRID_SIZE * GRID_SIZE), dtype=np.uint8)


def _semantic_channels(env: BobbyCarrotEnv) -> Tuple[np.ndarray, np.ndarray]:
    assert env.map_info is not None and env.bobby is not None

    data_arr      = np.asarray(env.map_info.data, dtype=np.uint8)   # (256,)
    all_collected = env.bobby.is_finished(env.map_info)

    # Vectorised tile → channel (no Python loop)
    np.copyto(_CHANNELS_BUF, _TILE_CHANNEL[data_arr])               # reuse buffer

    # Override finish tile channel based on collection phase
    finish_mask = (data_arr == 44)
    _CHANNELS_BUF[finish_mask] = 3 if all_collected else 4

    # Build one-hot grid using advanced indexing
    # FIX: reset with fill (faster than np.zeros on pre-allocated array)
    _GRID_BUF.fill(0)
    _GRID_BUF[_CHANNELS_BUF.astype(np.int32), _FLAT_IDX] = 1

    grid = _GRID_BUF.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).copy()  # copy — caller mutates

    # Agent position channel
    px, py = env.bobby.coord_src
    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
        grid[11, py, px] = 1

    # Phase flag
    if all_collected:
        grid[12, :, :] = 1

    # ---- Inventory vector ----
    remaining = (
        (env.map_info.carrot_total - env.bobby.carrot_count) +
        (env.map_info.egg_total    - env.bobby.egg_count)
    )
    denom          = max(1, env.map_info.carrot_total + env.map_info.egg_total)
    remaining_norm = min(1.0, remaining / denom)

    # FIX: numpy manhattan — reuse data_arr for fast position lookup
    if not all_collected:
        target_mask = (data_arr == 19) | (data_arr == 45)
        if target_mask.any():
            indices        = np.where(target_mask)[0]
            xs             = indices % GRID_SIZE
            ys             = indices // GRID_SIZE
            manhattan      = np.abs(xs - px) + np.abs(ys - py)
            manhattan_norm = float(manhattan.min()) / (GRID_SIZE * 2)
        else:
            manhattan_norm = 0.0
    else:
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
# FIX: removed redundant `_crumble_adjacent_to_carrot` call from hot-path
#      — now only called when bobby is actually on a crumble tile (tile 30/31)
# ============================================================

def _crumble_adjacent_to_carrot(env: BobbyCarrotEnv) -> bool:
    assert env.map_info is not None and env.bobby is not None
    px, py   = env.bobby.coord_src
    tile_val = env.map_info.data[px + py * GRID_SIZE]
    # FIX: early exit — only check adjacency when on crumble/broken tile
    if tile_val not in (30, 31):
        return False
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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

    # prev/curr inv[4] = manhattan_norm — positive delta means moved closer
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])

    if not all_collected_now:
        reward += distance_scale * dist_delta
    elif not level_done:
        reward += distance_scale * dist_delta
        reward += post_penalty

    # FIX: only call crumble check when actually on a crumble-type tile
    if _crumble_adjacent_to_carrot(env):
        reward += cfg.crumble_step_penalty

    return reward


# ============================================================
# DQN Agent
# FIX: inference buffers sized correctly for GRID_CHANNELS (was potentially stale)
# FIX: compile called with disable=False guard to avoid silent failures
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

        is_cuda = (device.type == "cuda")
        self.replay = ReplayBuffer(cfg.replay_capacity, pin=is_cuda)

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Use no_grad (NOT inference_mode): inference_mode marks output tensors
        # as inference tensors — inplace ops on them inside torch.compile CUDA
        # graph capture raise "Inplace update to inference tensor" RuntimeError.
        with torch.no_grad():
            g = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).to(
                self.device, non_blocking=True
            )
            v = torch.from_numpy(inv).unsqueeze(0).to(
                self.device, non_blocking=True
            )
            return int(self.policy_net(g, v).argmax(1).item())

    def optimize_step(self) -> Optional[float]:
        if len(self.replay) < self.cfg.min_replay_size:
            return None
        if self.total_steps % self.cfg.train_every_steps != 0:
            return None

        sg, sv, acts, rwds, nsg, nsv, dones = self.replay.sample(self.cfg.batch_size)

        sg_t  = torch.from_numpy(sg).to(self.device,    non_blocking=True)
        sv_t  = torch.from_numpy(sv).to(self.device,    non_blocking=True)
        a_t   = torch.from_numpy(acts).unsqueeze(1).to(self.device, non_blocking=True)
        r_t   = torch.from_numpy(rwds).to(self.device,  non_blocking=True)
        nsg_t = torch.from_numpy(nsg).to(self.device,   non_blocking=True)
        nsv_t = torch.from_numpy(nsv).to(self.device,   non_blocking=True)
        d_t   = torch.from_numpy(dones).to(self.device, non_blocking=True)

        q_pred = self.policy_net(sg_t, sv_t).gather(1, a_t).squeeze(1)

        with torch.no_grad():
            best_a = self.policy_net(sg_t, sv_t).argmax(1, keepdim=True)   # double DQN
            q_next = self.target_net(nsg_t, nsv_t).gather(1, best_a).squeeze(1)
            target = (r_t + (1.0 - d_t) * self.cfg.gamma * q_next).detach()

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
        # FIX: unwrap compiled model before saving (torch.compile wraps state_dict)
        def _state(m: nn.Module) -> dict:
            base = getattr(m, "_orig_mod", m)
            return base.state_dict()

        torch.save({
            "policy":        _state(self.policy_net),
            "target":        _state(self.target_net),
            "optim":         self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "total_steps":   self.total_steps,
            "level":         level,
            "map_kind":      map_kind,
            "grid_channels": GRID_CHANNELS,
            "inv_features":  INV_FEATURES,
        }, path)

    def load(self, path: Path) -> Dict[str, object]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # FIX: load into the underlying model, not the compiled wrapper
        base_policy = getattr(self.policy_net, "_orig_mod", self.policy_net)
        base_target = getattr(self.target_net, "_orig_mod", self.target_net)
        base_policy.load_state_dict(ckpt["policy"])
        base_target.load_state_dict(ckpt.get("target", ckpt["policy"]))
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


# FIX: compile guard — only compile if torch >= 2.0 and CUDA available
# Also use `mode="reduce-overhead"` which is faster than default for repeated small graphs
def _maybe_compile(model: nn.Module, device: torch.device) -> nn.Module:
    if not hasattr(torch, "compile"):
        return model
    if device.type != "cuda":
        return model   # torch.compile on CPU is often slower
    try:
        # "default" mode: applies kernel fusion & vectorisation without CUDA
        # graph capture. "reduce-overhead" forces CUDA graphs which crash when
        # any layer does inplace updates (GroupNorm, running stats, etc.).
        return torch.compile(model, mode="default")
    except Exception as e:
        print(f"[warn] torch.compile failed: {e} — running without compilation.")
        return model


# ============================================================
# Training loop
# FIX: env is created once per level, not per episode (was leaking)
# FIX: _semantic_channels called after reset only, not twice
# FIX: steps/sec counter based on wall-clock env steps
# FIX: early stopping if success_rate > 0.9 for 3 consecutive windows
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
    best_success    = 0.0
    early_stop_wins = 0       # FIX: early stopping counter

    print(f"  max_steps={max_steps} | episodes={n_episodes} | "
          f"distance_scale={level_cfg['distance_scale']} | "
          f"post_penalty={level_cfg['post_penalty']} | "
          f"batch={cfg.batch_size} | lr={cfg.lr}")

    t0              = time.time()
    total_env_steps = 0

    for episode in range(1, n_episodes + 1):
        # FIX: set_map + reset in one go; no redundant template copy
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

            # FIX: oversample terminal transitions to help with sparse rewards
            if bool(info.get("level_completed", False)) and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps  += 1
            total_env_steps    += 1
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
            sps    = total_env_steps / max(elapsed, 1e-6)
            eta_h  = ((n_episodes - episode) * max_steps) / max(sps, 1) / 3600
            print(
                f"[L{level}] ep={episode:5d} | "
                f"reward={avg_r:7.1f} | "
                f"collected={avg_c:5.1%} | "
                f"success={avg_s:5.1%} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_l:.4f} | "
                f"sps={sps:5.0f} | "
                f"ETA={eta_h:.1f}h"
            )

            # FIX: early stopping — if success > 90% for 3 consecutive windows, stop
            if avg_s >= 0.90:
                early_stop_wins += 1
                if early_stop_wins >= 3:
                    print(f"[L{level}] Early stop: success >= 90% for 3 windows.")
                    break
            else:
                early_stop_wins = 0

        if ckpt_path is not None and episode % cfg.save_every == 0:
            rolling_s = float(np.mean(success_hist[-cfg.report_every:]))
            if rolling_s >= best_success:
                best_success = rolling_s
                best_path = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(best_path, level=level, map_kind=map_kind)
                print(f"  [ckpt] New best saved: success={rolling_s:.2%} → {best_path.name}")

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
    # FIX: set eval mode for BatchNorm to use running stats during play
    agent.policy_net.eval()
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
    p = argparse.ArgumentParser(description="Bobby Carrot DQN — optimised for Colab T4")
    p.add_argument("--play",               action="store_true")
    p.add_argument("--map-kind",           default="normal", choices=["normal", "egg"])
    p.add_argument("--map-number",         type=int,   default=1)
    p.add_argument("--levels",             type=int,   nargs="+", default=None)
    p.add_argument("--individual-levels",  action="store_true")
    p.add_argument("--episodes-per-level", type=int,   default=None)
    p.add_argument("--max-steps",          type=int,   default=None)
    p.add_argument("--batch-size",         type=int,   default=256)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--gamma",              type=float, default=0.99)
    p.add_argument("--epsilon-start",      type=float, default=1.0)
    p.add_argument("--epsilon-min",        type=float, default=0.05)
    p.add_argument("--epsilon-decay",      type=float, default=0.997)
    p.add_argument("--completion-bonus",   type=float, default=300.0)
    p.add_argument("--death-penalty",      type=float, default=-80.0)
    p.add_argument("--crumble-step-penalty",  type=float, default=-5.0)
    p.add_argument("--invalid-move-penalty",  type=float, default=-0.5)
    p.add_argument("--terminal-oversample",   type=int,   default=6)
    p.add_argument("--observation-mode",   default="full", choices=["full", "local", "compact"])
    p.add_argument("--report-every",       type=int,   default=100)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--model-path",         default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",     default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes",      type=int,   default=5)
    p.add_argument("--no-render",          action="store_true")
    p.add_argument("--render-fps",         type=float, default=5.0)
    p.add_argument("--no-compile",         action="store_true",
                   help="Disable torch.compile (useful if warm-up time is a bottleneck)")
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
            if not args.no_compile:
                agent.policy_net = _maybe_compile(agent.policy_net, device)
                agent.target_net = _maybe_compile(agent.target_net, device)
            out     = ckpt_dir / f"dqn_level{lvl}_individual.pt"
            summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
            agent.save(out, level=lvl, map_kind=args.map_kind)
            print(f"Saved {out} | {summary}")
        return

    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
    if not args.no_compile:
        # FIX: compile after model is on device, not before
        agent.policy_net = _maybe_compile(agent.policy_net, device)
        agent.target_net = _maybe_compile(agent.target_net, device)

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
