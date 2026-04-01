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

# Import game primitives directly — we bypass the slow rl_env step loop
from bobby_carrot.game import (
    Bobby, Map, MapInfo, State,
    FRAMES_PER_STEP, WIDTH_POINTS, HEIGHT_POINTS,
)
from bobby_carrot.rl_env import (
    BobbyCarrotEnv, RewardConfig,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:
    raise RuntimeError("PyTorch is required. Install with: pip install torch") from exc


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

GRID_SIZE     = 16
GRID_CHANNELS = 13
INV_FEATURES  = 5

# Tile-id → channel index (built once at import time)
# 0=wall  1=carrot  2=egg  3=finish-active  4=finish-inactive
# 5=hazard  6=key  7=locked-door  8=crumble  9=conveyor  10=floor  11=agent  12=phase
_TILE_CH = np.zeros(256, dtype=np.int8)
for _i in range(256):
    if   _i < 18:                _TILE_CH[_i] = 0
    elif _i == 19:               _TILE_CH[_i] = 1
    elif _i == 45:               _TILE_CH[_i] = 2
    elif _i == 44:               _TILE_CH[_i] = 4   # active=3 set dynamically
    elif _i in (31, 46):         _TILE_CH[_i] = 5
    elif _i in (32, 34, 36):     _TILE_CH[_i] = 6
    elif _i in (33, 35, 37):     _TILE_CH[_i] = 7
    elif _i == 30:               _TILE_CH[_i] = 8
    elif _i in (40, 41, 42, 43): _TILE_CH[_i] = 9
    else:                        _TILE_CH[_i] = 10

# Pre-allocated reusable buffers (one allocation total — reused every step)
_FLAT_IDX = np.arange(256, dtype=np.int32)
_CH_BUF   = np.zeros(256, dtype=np.int8)
_GRID_BUF = np.zeros((GRID_CHANNELS, 256), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Per-level training configuration
# ─────────────────────────────────────────────────────────────

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 400,  "episodes": 3000, "distance_scale": 2.5, "post_penalty": -0.2},
    2:  {"max_steps": 500,  "episodes": 3000, "distance_scale": 2.5, "post_penalty": -0.2},
    3:  {"max_steps": 600,  "episodes": 3000, "distance_scale": 2.0, "post_penalty": -0.3},
    4:  {"max_steps": 600,  "episodes": 3000, "distance_scale": 2.0, "post_penalty": -0.3},
    5:  {"max_steps": 500,  "episodes": 3500, "distance_scale": 1.8, "post_penalty": -0.3},
    6:  {"max_steps": 700,  "episodes": 3500, "distance_scale": 1.5, "post_penalty": -0.4},
    7:  {"max_steps": 900,  "episodes": 5000, "distance_scale": 1.2, "post_penalty": -0.5},
    8:  {"max_steps": 400,  "episodes": 3000, "distance_scale": 2.0, "post_penalty": -0.2},
    9:  {"max_steps": 700,  "episodes": 3500, "distance_scale": 1.5, "post_penalty": -0.4},
    10: {"max_steps": 600,  "episodes": 3500, "distance_scale": 1.8, "post_penalty": -0.3},
}

def _get_level_cfg(level: int) -> Dict:
    return LEVEL_CONFIG.get(level, {
        "max_steps": 600, "episodes": 3500,
        "distance_scale": 1.8, "post_penalty": -0.3,
    })


# ─────────────────────────────────────────────────────────────
# DQNConfig
# ─────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    gamma:               float = 0.99
    lr:                  float = 5e-4          # higher LR → faster convergence on sparse signal
    batch_size:          int   = 128           # smaller batch → more updates per env-step
    replay_capacity:     int   = 50_000        # small buffer → stays fresh
    min_replay_size:     int   = 500           # start learning very early
    target_update_steps: int   = 600
    train_every_steps:   int   = 1             # train after EVERY env step
    # Exploration — slow decay so agent has time to find carrots before going greedy
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.10
    epsilon_decay:       float = 0.9985        # hits 0.10 around ep ~1500
    # Reward — carrot signal must dominate all penalties
    completion_bonus:    float = 200.0
    death_penalty:       float = -30.0         # soft: don't overshadow carrot signal
    carrot_bonus:        float = 40.0          # LARGE: unmistakably positive
    crumble_penalty:     float = -2.0
    invalid_move_penalty:float = -0.2          # soft: don't punish exploration
    step_penalty:        float = -0.01         # near-zero: don't drown signal in step cost
    terminal_oversample: int   = 10            # store winning transitions many times
    # Misc
    report_every:        int   = 100
    save_every:          int   = 250
    seed:                int   = 42
    observation_mode:    str   = "full"


# ─────────────────────────────────────────────────────────────
# FastBobbyEnv — direct single-step game logic, no frame loop
#
# ROOT CAUSE OF sps=84:
#   BobbyCarrotEnv._advance_until_transition() loops up to
#   8 * FRAMES_PER_STEP * 3 = 48 Python calls to update_texture_position()
#   per RL step. That function has large if/elif chains for sprite animation.
#   48 calls × ~0.25ms each = ~12ms per step → sps ≈ 84.
#
# FIX: implement only the tile-transition logic (the step==8 branch of
#   update_texture_position) with no animation bookkeeping at all.
#   One step = one call. sps → 800–1500.
# ─────────────────────────────────────────────────────────────

class FastBobbyEnv:
    """Lightweight env — applies exactly one game-logic step per RL step."""

    ACTIONS = [State.Left, State.Right, State.Up, State.Down]

    def __init__(self, map_kind: str, map_number: int, max_steps: int) -> None:
        self.map_kind       = map_kind
        self.map_number     = map_number
        self.max_steps      = max_steps
        self._map_obj       = Map(map_kind, map_number)
        self._fresh:  Optional[MapInfo] = None
        self.map_info:Optional[MapInfo] = None
        self.bobby:   Optional[Bobby]   = None
        self.step_count     = 0
        self.episode_done   = False
        self.action_space_n = 4

    # ── public API ───────────────────────────────────────────

    def set_map(self, map_kind: str, map_number: int) -> None:
        if map_kind != self.map_kind or map_number != self.map_number:
            self.map_kind   = map_kind
            self.map_number = map_number
            self._map_obj   = Map(map_kind, map_number)
            self._fresh     = None   # force reload on next reset

    def reset(self) -> None:
        if self._fresh is None:
            self._fresh = self._map_obj.load_map_info()
        mi = self._fresh
        self.map_info = MapInfo(
            data=mi.data.copy(),
            coord_start=mi.coord_start,
            carrot_total=mi.carrot_total,
            egg_total=mi.egg_total,
        )
        self.bobby = Bobby(
            start_frame=0, start_time=0,
            coord_src=self.map_info.coord_start,
        )
        self.bobby.state      = State.Down
        self.bobby.coord_dest = self.bobby.coord_src
        self.step_count   = 0
        self.episode_done = False

    def step(self, action: int) -> Tuple[float, bool, Dict[str, object]]:
        """Apply one action. Returns (raw_reward=0, done, info).
        All reward shaping is done externally in _shape_reward().
        """
        assert self.bobby is not None and self.map_info is not None
        b  = self.bobby
        md = self.map_info.data

        before_carrot = b.carrot_count
        before_egg    = b.egg_count
        before_pos    = b.coord_src

        # ── apply movement ───────────────────────────────────
        desired = self.ACTIONS[action]
        b.state      = desired
        b.coord_dest = b.coord_src   # reset dest before update_dest
        b.update_dest(md)
        invalid_move = (b.coord_dest == b.coord_src)  # dest unchanged = blocked

        # ── tile transitions (only when actually moving) ─────
        moved = b.coord_src != b.coord_dest
        if moved:
            old_pos  = b.coord_src[0]  + b.coord_src[1]  * GRID_SIZE
            new_pos  = b.coord_dest[0] + b.coord_dest[1] * GRID_SIZE

            # Tile underfoot (leaving)
            ot = md[old_pos]
            if   ot == 24: md[old_pos] = 25
            elif ot == 25: md[old_pos] = 26
            elif ot == 26: md[old_pos] = 27
            elif ot == 27: md[old_pos] = 24
            elif ot == 28: md[old_pos] = 29
            elif ot == 29: md[old_pos] = 28
            elif ot == 30: md[old_pos] = 31          # crumble → broken
            elif ot == 45:                           # egg step-on
                md[old_pos] = 46
                b.egg_count += 1

            # Destination tile (arriving)
            nt = md[new_pos]
            if   nt == 19:                           # carrot
                md[new_pos] = 20
                b.carrot_count += 1
            elif nt == 22: _apply_switch(md, 22)     # red switch
            elif nt == 32: md[new_pos] = 18; b.key_gray   += 1
            elif nt == 33 and b.key_gray   > 0: md[new_pos] = 18; b.key_gray   -= 1
            elif nt == 34: md[new_pos] = 18; b.key_yellow += 1
            elif nt == 35 and b.key_yellow > 0: md[new_pos] = 18; b.key_yellow -= 1
            elif nt == 36: md[new_pos] = 18; b.key_red    += 1
            elif nt == 37 and b.key_red    > 0: md[new_pos] = 18; b.key_red    -= 1
            elif nt == 38: _apply_switch(md, 38)     # blue switch
            elif nt == 40: b.next_state = State.Left
            elif nt == 41: b.next_state = State.Right
            elif nt == 42: b.next_state = State.Up
            elif nt == 43: b.next_state = State.Down
            elif nt == 31: b.dead = True             # spike

            b.coord_src = b.coord_dest

        # Death check on current tile after move
        cur_pos  = b.coord_src[0] + b.coord_src[1] * GRID_SIZE
        cur_tile = md[cur_pos]
        if cur_tile == 31:
            b.dead = True

        all_collected = self._is_finished()
        on_finish     = (cur_tile == 44) and all_collected and not b.dead

        carrot_delta = b.carrot_count - before_carrot
        egg_delta    = b.egg_count    - before_egg

        done = b.dead or on_finish
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.episode_done = done

        info: Dict[str, object] = {
            "collected_carrot": carrot_delta,
            "collected_egg":    egg_delta,
            "all_collected":    all_collected,
            "invalid_move":     invalid_move,
            "dead":             b.dead,
            "level_completed":  on_finish,
            "position":         b.coord_src,
            "moved":            moved,
        }
        return 0.0, done, info   # raw reward=0; shaping done externally

    def _is_finished(self) -> bool:
        b  = self.bobby
        mi = self.map_info
        assert b is not None and mi is not None
        if mi.carrot_total > 0:
            return b.carrot_count >= mi.carrot_total
        return b.egg_count >= mi.egg_total

    def close(self) -> None:
        pass


def _apply_switch(md: List[int], tile: int) -> None:
    """Toggle tiles for red (22) or blue (38) switch."""
    pairs = (
        {22:23, 23:22, 24:25, 25:26, 26:27, 27:24, 28:29, 29:28}
        if tile == 22 else
        {38:39, 39:38, 40:41, 41:40, 42:43, 43:42}
    )
    for i in range(256):
        v = md[i]
        if v in pairs:
            md[i] = pairs[v]


# ─────────────────────────────────────────────────────────────
# Observation encoding — vectorised, zero Python loops
# ─────────────────────────────────────────────────────────────

def _semantic_channels(env: FastBobbyEnv) -> Tuple[np.ndarray, np.ndarray]:
    b  = env.bobby
    mi = env.map_info
    assert b is not None and mi is not None

    data_arr      = np.asarray(mi.data, dtype=np.uint8)
    all_collected = env._is_finished()

    # Vectorised tile → channel index (no Python loop)
    np.copyto(_CH_BUF, _TILE_CH[data_arr])
    _CH_BUF[data_arr == 44] = 3 if all_collected else 4

    # Build one-hot grid (reuse buffer — fill(0) is faster than np.zeros())
    _GRID_BUF.fill(0)
    _GRID_BUF[_CH_BUF.astype(np.int32), _FLAT_IDX] = 1
    grid = _GRID_BUF.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).copy()

    px, py = b.coord_src
    grid[11, py, px] = 1        # agent position channel
    if all_collected:
        grid[12, :, :] = 1      # phase flag channel

    # ── inventory vector ─────────────────────────────────────
    remaining = (mi.carrot_total - b.carrot_count) + (mi.egg_total - b.egg_count)
    denom     = max(1, mi.carrot_total + mi.egg_total)
    rem_norm  = min(1.0, remaining / denom)

    if not all_collected:
        mask = (data_arr == 19) | (data_arr == 45)
        if mask.any():
            idx     = np.where(mask)[0]
            xs, ys  = idx % GRID_SIZE, idx // GRID_SIZE
            md_norm = float(np.min(np.abs(xs - px) + np.abs(ys - py))) / (GRID_SIZE * 2)
        else:
            md_norm = 0.0
    else:
        fidx = np.where(data_arr == 44)[0]
        if len(fidx):
            fx, fy  = int(fidx[0] % GRID_SIZE), int(fidx[0] // GRID_SIZE)
            md_norm = min(1.0, (abs(fx - px) + abs(fy - py)) / (GRID_SIZE * 2))
        else:
            md_norm = 0.0

    inv = np.array([
        float(b.key_gray   > 0),
        float(b.key_yellow > 0),
        float(b.key_red    > 0),
        rem_norm,
        md_norm,
    ], dtype=np.float32)

    return grid, inv


# ─────────────────────────────────────────────────────────────
# Reward shaping — all shaping lives here, env returns raw 0
#
# ROOT CAUSE OF collected=0%:
#   The original BobbyCarrotEnv.RewardConfig applies step=-0.05 and
#   no_progress_penalty=-0.2/-0.4 continuously. Combined with carrot_bonus=15,
#   the agent can never collect a carrot early enough before the penalty flood
#   overwhelms the signal. With step=-0.01 and carrot_bonus=40 the carrot
#   reward is 4000x the per-step cost — the agent always "profits" by seeking.
# ─────────────────────────────────────────────────────────────

def _shape_reward(
    info:       Dict[str, object],
    cfg:        DQNConfig,
    level_cfg:  Dict,
    prev_inv:   np.ndarray,
    curr_inv:   np.ndarray,
    env:        FastBobbyEnv,
) -> float:
    r = cfg.step_penalty

    if info["invalid_move"]:
        r += cfg.invalid_move_penalty
    if info["dead"]:
        r += cfg.death_penalty

    carrot_delta = int(info["collected_carrot"])
    egg_delta    = int(info["collected_egg"])
    if carrot_delta > 0:
        r += cfg.carrot_bonus * carrot_delta
    if egg_delta > 0:
        r += cfg.carrot_bonus * egg_delta   # treat eggs same as carrots

    if info["level_completed"]:
        r += cfg.completion_bonus

    all_collected = bool(info["all_collected"])
    level_done    = bool(info["level_completed"])

    # Distance shaping: prev_inv[4] = Manhattan norm to nearest target/finish
    # Positive delta = moved closer = positive reward
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])
    r += level_cfg["distance_scale"] * dist_delta

    # Post-collection step penalty: collected everything but not yet on finish tile
    if all_collected and not level_done:
        r += level_cfg["post_penalty"]

    # Crumble adjacency penalty
    if env.bobby is not None and env.map_info is not None:
        b  = env.bobby
        md = env.map_info.data
        px, py = b.coord_src
        if md[px + py * GRID_SIZE] in (30, 31):
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = px+dx, py+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if md[nx + ny * GRID_SIZE] == 19:
                        r += cfg.crumble_penalty
                        break

    return r


# ─────────────────────────────────────────────────────────────
# Replay buffer — pre-allocated, uint8 grids, recency-biased sampling
#
# recency bias: 40% of each batch comes from the newest 25% of stored
# transitions. Without this, early all-failure episodes dilute the buffer
# and stall learning even after the agent starts finding carrots.
# ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._cap  = capacity
        self._ptr  = 0
        self._size = 0
        self._sg   = np.zeros((capacity, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self._sv   = np.zeros((capacity, INV_FEATURES),                         dtype=np.float32)
        self._a    = np.zeros(capacity,                                         dtype=np.int64)
        self._r    = np.zeros(capacity,                                         dtype=np.float32)
        self._ng   = np.zeros((capacity, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self._nv   = np.zeros((capacity, INV_FEATURES),                         dtype=np.float32)
        self._d    = np.zeros(capacity,                                         dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        grid: np.ndarray, inv: np.ndarray,
        action: int, reward: float,
        next_grid: np.ndarray, next_inv: np.ndarray,
        done: bool,
    ) -> None:
        i           = self._ptr
        self._sg[i] = grid
        self._sv[i] = inv
        self._a[i]  = action
        self._r[i]  = reward
        self._ng[i] = next_grid
        self._nv[i] = next_inv
        self._d[i]  = float(done)
        self._ptr   = (i + 1) % self._cap
        self._size  = min(self._size + 1, self._cap)

    def sample(self, n: int, recent_frac: float = 0.4) -> Tuple[np.ndarray, ...]:
        n_recent  = int(n * recent_frac)
        n_uniform = n - n_recent
        idx_uni   = np.random.randint(0, self._size, size=n_uniform)

        win = max(n, self._size // 4)
        start = (self._ptr - win) % self._cap
        if self._size >= win:
            if start + win <= self._cap:
                pool = np.arange(start, start + win)
            else:
                pool = np.concatenate([
                    np.arange(start, self._cap),
                    np.arange(0, (start + win) % self._cap),
                ])
            idx_rec = pool[np.random.randint(0, len(pool), size=n_recent)]
        else:
            idx_rec = np.random.randint(0, self._size, size=n_recent)

        idx = np.concatenate([idx_uni, idx_rec])
        np.random.shuffle(idx)

        return (
            self._sg[idx].astype(np.float32),
            self._sv[idx],
            self._a[idx],
            self._r[idx],
            self._ng[idx].astype(np.float32),
            self._nv[idx],
            self._d[idx],
        )


# ─────────────────────────────────────────────────────────────
# Dueling DDQN — GroupNorm (compile-safe, no inplace buffer updates)
# ─────────────────────────────────────────────────────────────

class DuelingDQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        # GroupNorm NOT BatchNorm:
        #   BatchNorm updates running_mean/var INPLACE → crashes torch.compile
        #   CUDA graph capture ("Inplace update to inference tensor").
        #   GroupNorm has zero running stats — pure computation, compile-safe.
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        merged = 64 * GRID_SIZE * GRID_SIZE + INV_FEATURES
        self.shared     = nn.Sequential(nn.Linear(merged, 512), nn.ReLU(inplace=True))
        self.value_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.adv_head   = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, n_actions))

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        shared = self.shared(torch.cat([self.conv(grid), inv], dim=1))
        v = self.value_head(shared)
        a = self.adv_head(shared)
        return v + a - a.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────

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

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # torch.no_grad() NOT inference_mode():
        #   inference_mode marks output tensors as inference tensors.
        #   Inplace ops on them inside torch.compile CUDA graph capture crash
        #   with "Inplace update to inference tensor outside InferenceMode".
        with torch.no_grad():
            g = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).to(
                self.device, non_blocking=True)
            v = torch.from_numpy(inv).unsqueeze(0).to(
                self.device, non_blocking=True)
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
            # Double DQN: policy selects action, target evaluates it
            best_a = self.policy_net(sg_t, sv_t).argmax(1, keepdim=True)
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
        def _sd(m: nn.Module) -> dict:
            return getattr(m, "_orig_mod", m).state_dict()
        torch.save({
            "policy":        _sd(self.policy_net),
            "target":        _sd(self.target_net),
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
        def _base(m: nn.Module) -> nn.Module:
            return getattr(m, "_orig_mod", m)
        _base(self.policy_net).load_state_dict(ckpt["policy"])
        _base(self.target_net).load_state_dict(ckpt.get("target", ckpt["policy"]))
        if "optim" in ckpt:
            self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon     = float(ckpt.get("epsilon",     self.cfg.epsilon_start))
        self.total_steps = int  (ckpt.get("total_steps", 0))
        return {"level": ckpt.get("level", -1), "map_kind": ckpt.get("map_kind", "normal")}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(map_kind: str, level: int, max_steps: int) -> FastBobbyEnv:
    return FastBobbyEnv(map_kind=map_kind, map_number=level, max_steps=max_steps)


# "default" mode — safe with GroupNorm.
# "reduce-overhead" forces CUDA graphs → crash with inplace ops (BatchNorm etc).
def _maybe_compile(model: nn.Module, device: torch.device) -> nn.Module:
    if not hasattr(torch, "compile") or device.type != "cuda":
        return model
    try:
        return torch.compile(model, mode="default")
    except Exception as e:
        print(f"[warn] torch.compile failed: {e}")
        return model


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train_one_level(
    agent:     DQNAgent,
    level:     int,
    cfg:       DQNConfig,
    map_kind:  str,
    ckpt_path: Optional[Path] = None,
) -> Dict[str, float]:

    level_cfg  = _get_level_cfg(level)
    max_steps  = level_cfg["max_steps"]
    n_episodes = level_cfg["episodes"]
    env        = _make_env(map_kind, level, max_steps)

    reward_hist:    List[float] = []
    success_hist:   List[float] = []
    collected_hist: List[float] = []
    loss_win: Deque[float]      = deque(maxlen=cfg.report_every)
    best_success    = 0.0
    early_stop_wins = 0

    print(f"  max_steps={max_steps} | episodes={n_episodes} | "
          f"dist_scale={level_cfg['distance_scale']} | "
          f"post_pen={level_cfg['post_penalty']} | "
          f"batch={cfg.batch_size} | lr={cfg.lr} | "
          f"carrot={cfg.carrot_bonus} | eps_decay={cfg.epsilon_decay}")

    t0 = time.time()
    total_env_steps = 0

    for episode in range(1, n_episodes + 1):
        env.set_map(map_kind=map_kind, map_number=level)
        env.reset()
        grid, inv = _semantic_channels(env)

        done     = False
        steps    = 0
        ep_r     = 0.0
        info: Dict[str, object] = {}
        prev_inv = inv.copy()

        while not done and steps < max_steps:
            action         = agent.select_action(grid, inv)
            _, done, info  = env.step(action)
            next_grid, next_inv = _semantic_channels(env)

            shaped = _shape_reward(info, cfg, level_cfg, prev_inv, next_inv, env)

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            # Oversample rare winning transitions
            if info.get("level_completed") and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps  += 1
            total_env_steps    += 1
            loss = agent.optimize_step()
            if loss is not None:
                loss_win.append(loss)

            prev_inv  = next_inv
            grid, inv = next_grid, next_inv
            ep_r  += shaped
            steps += 1

        agent.decay_epsilon()

        success   = 1.0 if info.get("level_completed") else 0.0
        collected = 1.0 if info.get("all_collected")   else 0.0
        reward_hist.append(ep_r)
        success_hist.append(success)
        collected_hist.append(collected)

        if episode % cfg.report_every == 0 or episode == 1:
            n       = cfg.report_every
            avg_r   = float(np.mean(reward_hist[-n:]))
            avg_s   = float(np.mean(success_hist[-n:]))
            avg_c   = float(np.mean(collected_hist[-n:]))
            avg_l   = float(np.mean(loss_win)) if loss_win else 0.0
            elapsed = time.time() - t0
            sps     = total_env_steps / max(elapsed, 1e-6)
            eta_h   = ((n_episodes - episode) * max_steps) / max(sps, 1) / 3600
            print(
                f"[L{level}] ep={episode:5d} | "
                f"reward={avg_r:8.1f} | "
                f"collected={avg_c:5.1%} | "
                f"success={avg_s:5.1%} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_l:.4f} | "
                f"sps={sps:6.0f} | "
                f"ETA={eta_h:.1f}h"
            )

            # Early stop: 90%+ success for 3 consecutive windows
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
                best_path    = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(best_path, level=level, map_kind=map_kind)
                print(f"  [ckpt] New best: success={rolling_s:.1%} → {best_path.name}")

    env.close()
    return {
        "mean_reward":        float(np.mean(reward_hist))    if reward_hist    else 0.0,
        "success_rate":       float(np.mean(success_hist))   if success_hist   else 0.0,
        "all_collected_rate": float(np.mean(collected_hist)) if collected_hist else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Play / evaluation  (uses BobbyCarrotEnv for full game fidelity)
# ─────────────────────────────────────────────────────────────

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

    # Silence all env-internal reward so we measure true task performance
    silent_rc = RewardConfig(
        step=0.0, carrot=0.0, egg=0.0, finish=0.0, death=0.0,
        invalid_move=0.0, distance_delta_scale=0.0,
        new_best_target_distance_scale=0.0,
        new_best_finish_distance_scale=0.0,
        post_collection_step_penalty=0.0,
        no_progress_penalty_after=999999, no_progress_penalty=0.0,
        no_progress_penalty_hard_after=999999, no_progress_penalty_hard=0.0,
        all_collected_bonus=0.0,
    )
    full_env = BobbyCarrotEnv(
        map_kind=map_kind, map_number=map_number,
        observation_mode="full",
        local_view_size=3, include_inventory=True,
        headless=not render, max_steps=max_steps,
        reward_config=silent_rc,
    )

    agent = DQNAgent(n_actions=full_env.action_space_n, cfg=cfg, device=device)
    meta  = agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    print(f"Loaded {model_path}  (trained level={meta['level']}, kind={meta['map_kind']})")

    # We use a FastBobbyEnv proxy to compute observations identically to training
    proxy = FastBobbyEnv(map_kind, map_number, max_steps)

    for ep in range(1, episodes + 1):
        full_env.set_map(map_kind=map_kind, map_number=map_number)
        full_env.reset()
        proxy.map_info = full_env.map_info
        proxy.bobby    = full_env.bobby
        grid, inv = _semantic_channels(proxy)

        done = False; steps = 0; total_r = 0.0
        info: Dict[str, object] = {}

        while not done and steps < max_steps:
            action             = agent.select_action(grid, inv)
            _, raw_r, done, info = full_env.step(action)
            proxy.map_info     = full_env.map_info
            proxy.bobby        = full_env.bobby
            grid, inv          = _semantic_channels(proxy)
            total_r           += raw_r
            steps             += 1
            if render:
                full_env.render()
                if render_fps > 0:
                    time.sleep(1.0 / render_fps)

        print(
            f"Play ep {ep}/{episodes} | "
            f"collected={bool(info.get('all_collected'))} | "
            f"success={bool(info.get('level_completed'))} | steps={steps}"
        )
    full_env.close()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bobby Carrot DQN — fast + stable training")
    p.add_argument("--play",               action="store_true")
    p.add_argument("--map-kind",           default="normal", choices=["normal", "egg"])
    p.add_argument("--map-number",         type=int,   default=1)
    p.add_argument("--levels",             type=int,   nargs="+", default=None)
    p.add_argument("--individual-levels",  action="store_true")
    p.add_argument("--episodes-per-level", type=int,   default=None)
    p.add_argument("--max-steps",          type=int,   default=None)
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--gamma",              type=float, default=0.99)
    p.add_argument("--epsilon-start",      type=float, default=1.0)
    p.add_argument("--epsilon-min",        type=float, default=0.10)
    p.add_argument("--epsilon-decay",      type=float, default=0.9985)
    p.add_argument("--completion-bonus",   type=float, default=200.0)
    p.add_argument("--death-penalty",      type=float, default=-30.0)
    p.add_argument("--carrot-bonus",       type=float, default=40.0)
    p.add_argument("--crumble-penalty",    type=float, default=-2.0)
    p.add_argument("--invalid-move-penalty", type=float, default=-0.2)
    p.add_argument("--step-penalty",       type=float, default=-0.01)
    p.add_argument("--terminal-oversample", type=int,  default=10)
    p.add_argument("--observation-mode",   default="full", choices=["full","local","compact"])
    p.add_argument("--report-every",       type=int,   default=100)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--model-path",
                   default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",
                   default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes",      type=int,   default=5)
    p.add_argument("--no-render",          action="store_true")
    p.add_argument("--render-fps",         type=float, default=5.0)
    p.add_argument("--no-compile",         action="store_true",
                   help="Disable torch.compile (skips ~30s warm-up on T4)")
    return p


def _cfg_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
        epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        completion_bonus=args.completion_bonus, death_penalty=args.death_penalty,
        carrot_bonus=args.carrot_bonus, crumble_penalty=args.crumble_penalty,
        invalid_move_penalty=args.invalid_move_penalty,
        step_penalty=args.step_penalty,
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

    probe_env = _make_env(args.map_kind, levels[0], 10)
    probe_env.reset()
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
