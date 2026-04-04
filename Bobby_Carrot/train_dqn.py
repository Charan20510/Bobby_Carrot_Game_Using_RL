from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
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

from bobby_carrot.game import (
    Bobby, Map, MapInfo, State,
    WIDTH_POINTS, HEIGHT_POINTS,
)
from bobby_carrot.rl_env import BobbyCarrotEnv, RewardConfig

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

# Tile-id → channel index (built once)
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

_FLAT_IDX = np.arange(256, dtype=np.int32)
_CH_BUF   = np.zeros(256, dtype=np.int8)
_GRID_BUF = np.zeros((GRID_CHANNELS, 256), dtype=np.uint8)

# Switch LUTs — vectorised tile toggle (faster than dict loop)
_SWITCH_RED  = np.arange(256, dtype=np.uint8)
for _s, _d in {22:23, 23:22, 24:25, 25:26, 26:27, 27:24, 28:29, 29:28}.items():
    _SWITCH_RED[_s] = _d
_SWITCH_BLUE = np.arange(256, dtype=np.uint8)
for _s, _d in {38:39, 39:38, 40:41, 41:40, 42:43, 43:42}.items():
    _SWITCH_BLUE[_s] = _d

# Pre-computed numpy array view of switch LUTs (avoids repeated asarray calls)
_SWITCH_RED_NP  = _SWITCH_RED.copy()
_SWITCH_BLUE_NP = _SWITCH_BLUE.copy()


# ─────────────────────────────────────────────────────────────
# BFS reachability — detects unreachable carrots after crumble
# ─────────────────────────────────────────────────────────────

def _bfs_reachable_carrot_info(md, sx, sy, grid_size=GRID_SIZE):
    """Direction-aware BFS from (sx,sy).

    Return (reachable_carrot_count, nearest_bfs_dist, finish_reachable, finish_dist).

    Respects directional constraints:
      - Arrows 24-27: rotating tiles with entry/exit restrictions
      - Conveyors 40-43: one-way tiles
      - Rails 28-29: horizontal/vertical only
    Spike (31) and broken-egg spike (46) are impassable.
    O(256) on a 16x16 grid — negligible overhead.
    """
    total = grid_size * grid_size
    visited = bytearray(total)
    q = deque()
    start_idx = sx + sy * grid_size
    q.append((sx, sy, 0))
    visited[start_idx] = 1

    reachable_count = 0
    nearest_dist = -1
    finish_reachable = False
    finish_dist = -1

    # Direction indices: 0=Left(-1,0), 1=Right(+1,0), 2=Up(0,-1), 3=Down(0,+1)
    _DIRS = ((-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3))

    while q:
        x, y, dist = q.popleft()
        tile = md[x + y * grid_size]

        if tile == 19 or tile == 45:  # uncollected carrot / egg
            reachable_count += 1
            if nearest_dist < 0 or dist < nearest_dist:
                nearest_dist = dist
        elif tile == 44:
            finish_reachable = True
            if finish_dist < 0 or dist < finish_dist:
                finish_dist = dist

        for ddx, ddy, d_idx in _DIRS:
            # Check if we can EXIT current tile in this direction
            if not _can_exit_tile(tile, d_idx):
                continue

            nx, ny = x + ddx, y + ddy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                ni = nx + ny * grid_size
                if not visited[ni]:
                    nt = md[ni]
                    if nt >= 18 and nt != 31 and nt != 46:
                        # Check if we can ENTER neighbour tile from this direction
                        if _can_enter_tile(nt, d_idx):
                            visited[ni] = 1
                            q.append((nx, ny, dist + 1))

    return reachable_count, nearest_dist, finish_reachable, finish_dist


def _can_exit_tile(tile: int, direction: int) -> bool:
    """Check if we can leave `tile` in `direction`.

    direction: 0=Left, 1=Right, 2=Up, 3=Down

    Derived from game.py update_dest old_item checks:
      old_item==24 forbids {Left,Up}   → can exit Right(1), Down(3)
      old_item==25 forbids {Right,Up}  → can exit Left(0),  Down(3)
      old_item==26 forbids {Right,Down}→ can exit Left(0),  Up(2)
      old_item==27 forbids {Left,Down} → can exit Right(1), Up(2)
    """
    # Arrow tiles: restrict exit directions
    # 24: forbids Left,Up → can exit Right or Down
    if tile == 24:
        return direction in (1, 3)
    # 25: forbids Right,Up → can exit Left or Down
    if tile == 25:
        return direction in (0, 3)
    # 26: forbids Right,Down → can exit Left or Up
    if tile == 26:
        return direction in (0, 2)
    # 27: forbids Left,Down → can exit Right or Up
    if tile == 27:
        return direction in (1, 2)
    # Horizontal rail 28: Left or Right only (forbids Up,Down)
    if tile == 28:
        return direction in (0, 1)
    # Vertical rail 29: Up or Down only (forbids Left,Right)
    if tile == 29:
        return direction in (2, 3)
    # Conveyor Left 40: forbids Up,Down,Right → can only exit Left
    if tile == 40:
        return direction == 0
    # Conveyor Right 41: forbids Up,Down,Left → can only exit Right
    if tile == 41:
        return direction == 1
    # Conveyor Up 42: forbids Left,Right,Down → can only exit Up
    if tile == 42:
        return direction == 2
    # Conveyor Down 43: forbids Left,Right,Up → can only exit Down
    if tile == 43:
        return direction == 3
    # All other walkable tiles: no exit restriction
    return True


def _can_enter_tile(tile: int, direction: int) -> bool:
    """Check if we can enter `tile` coming FROM `direction`.

    direction: 0=Left, 1=Right, 2=Up, 3=Down
    (the direction we are MOVING, not the side we enter from)
    """
    # Arrow tile entry restrictions (same as game.py update_dest)
    # 24: cannot enter from Right or Down
    if tile == 24:
        return direction not in (1, 3)
    # 25: cannot enter from Left or Down
    if tile == 25:
        return direction not in (0, 3)
    # 26: cannot enter from Left or Up
    if tile == 26:
        return direction not in (0, 2)
    # 27: cannot enter from Right or Up
    if tile == 27:
        return direction not in (1, 2)
    # Horizontal rail 28: cannot enter from Up or Down
    if tile == 28:
        return direction in (0, 1)
    # Vertical rail 29: cannot enter from Left or Right
    if tile == 29:
        return direction in (2, 3)
    # Conveyor Left 40: cannot enter going Right, Up, or Down
    if tile == 40:
        return direction == 0
    # Conveyor Right 41: cannot enter going Left, Up, or Down
    if tile == 41:
        return direction == 1
    # Conveyor Up 42: cannot enter going Down, Left, or Right
    if tile == 42:
        return direction == 2
    # Conveyor Down 43: cannot enter going Up, Left, or Right
    if tile == 43:
        return direction == 3
    # Locked doors: BFS assumes optimistic (keys may be found)
    # All other tiles: no entry restriction
    return True


# ─────────────────────────────────────────────────────────────
# Per-level config
# ─────────────────────────────────────────────────────────────

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 400,  "episodes": 2000,  "distance_scale": 2.5, "post_penalty": -0.2},
    2:  {"max_steps": 500,  "episodes": 2000,  "distance_scale": 2.5, "post_penalty": -0.2},
    3:  {"max_steps": 600,  "episodes": 2200,  "distance_scale": 2.0, "post_penalty": -0.3},
    4:  {"max_steps": 800,  "episodes": 5000,  "distance_scale": 2.0, "post_penalty": -0.3},
    5:  {"max_steps": 1000, "episodes": 8000,  "distance_scale": 2.0, "post_penalty": -0.3},
    6:  {"max_steps": 1200, "episodes": 10000, "distance_scale": 2.0, "post_penalty": -0.3},
    7:  {"max_steps": 1500, "episodes": 15000, "distance_scale": 1.5, "post_penalty": -0.3},
    8:  {"max_steps": 800,  "episodes": 5000,  "distance_scale": 2.0, "post_penalty": -0.3},
    9:  {"max_steps": 1000, "episodes": 6000,  "distance_scale": 2.0, "post_penalty": -0.3},
    10: {"max_steps": 1000, "episodes": 8000,  "distance_scale": 2.0, "post_penalty": -0.3},
}

def _get_level_cfg(level: int) -> Dict:
    return LEVEL_CONFIG.get(level, {
        "max_steps": 600, "episodes": 2500,
        "distance_scale": 1.8, "post_penalty": -0.3,
    })


@lru_cache(maxsize=64)
def _load_map_stats(map_kind: str, level: int) -> Dict[str, int]:
    """Cache map tile counts — called once per level."""
    mi  = Map(map_kind, level).load_map_info()
    arr = np.asarray(mi.data, dtype=np.uint8)
    return {
        "carrots":   int(np.count_nonzero(arr == 19)),
        "eggs":      int(np.count_nonzero(arr == 45)),
        "crumbles":  int(np.count_nonzero(arr == 30)),
        "conveyors": int(np.count_nonzero((arr >= 40) & (arr <= 43))),
        # FIX: also cache the finish tile position — used to verify finish detection
        "has_finish": int(np.any(arr == 44)),
    }


# ─────────────────────────────────────────────────────────────
# DQNConfig
# ─────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    gamma:               float = 0.99
    lr:                  float = 3e-4       # lower LR for more stable convergence on hard levels
    batch_size:          int   = 256
    replay_capacity:     int   = 200_000    # larger buffer for crumble-heavy levels
    min_replay_size:     int   = 512
    target_update_steps: int   = 500
    train_every_steps:   int   = 2          # every 2 steps (not 4 — too few updates)
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.05       # lower floor — exploit more once learning kicks in
    epsilon_decay:       float = 0.9996     # slower decay gives more exploration for harder levels
    completion_bonus:    float = 1200.0
    death_penalty:       float = -25.0
    carrot_bonus:        float = 18.0
    egg_bonus:           float = 18.0
    all_collected_bonus: float = 220.0
    safe_crumble_bonus:  float = 5.0
    late_collection_bonus_scale: float = 6.0
    incomplete_penalty:  float = -80.0      # softer — allows more exploration before converging
    crumble_penalty:     float = -2.0
    reachability_loss_penalty: float = -50.0 # per carrot made unreachable by crumble (episode continues)
    invalid_move_penalty:float = -0.15
    step_penalty:        float = -0.005     # near-zero: don't drown the carrot signal
    revisit_penalty:     float = -0.04
    terminal_oversample: int   = 48         # heavily replay rare successful endings
    all_collected_oversample: int = 10      # reinforce trajectories reaching final phase
    warmup_episodes:     int   = 20         # random-policy episodes before learning
    report_every:        int   = 100
    save_every:          int   = 250
    seed:                int   = 42
    observation_mode:    str   = "full"
    use_amp:             bool  = True
    pre_success_epsilon_floor: float = 0.25 # lower floor before first success


# ─────────────────────────────────────────────────────────────
# _apply_switch — pure Python loop (FAST)
#
# BUG in previous version: converted list→numpy→list for every switch call.
# That's 256 numpy allocations per episode on switch-heavy maps → sps=219.
# Pure Python if/elif over 256 ints is faster when called occasionally.
# ─────────────────────────────────────────────────────────────

# Pre-built Python dicts for switch toggling (avoids dict construction per call)
_RED_MAP  = {22:23, 23:22, 24:25, 25:26, 26:27, 27:24, 28:29, 29:28}
_BLUE_MAP = {38:39, 39:38, 40:41, 41:40, 42:43, 43:42}

def _apply_switch(md: List[int], tile: int) -> None:
    """Toggle map tiles for red (22) or blue (38) switch.

    Pure Python loop — faster than numpy round-trip for a 256-element list
    that is only occasionally mutated (most tiles don't match any switch key).
    """
    lut = _RED_MAP if tile == 22 else _BLUE_MAP
    for i in range(256):
        v = md[i]
        if v in lut:
            md[i] = lut[v]


# ─────────────────────────────────────────────────────────────
# FastBobbyEnv — one game-logic step per RL step, zero frame loop
# ─────────────────────────────────────────────────────────────

class FastBobbyEnv:
    """Applies exactly one game-logic step per RL step.

    Bypasses BobbyCarrotEnv._advance_until_transition() which loops
    up to 48 Python calls to update_texture_position() per step.
    """

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

        # FIX: cache expected totals from the ORIGINAL map, not live map_info.
        # map_info.data is mutated as items are collected, but carrot_total/egg_total
        # are set once at load time — safe to read from there.
        self._expected_carrots: int = 0
        self._expected_eggs:    int = 0

    def set_map(self, map_kind: str, map_number: int) -> None:
        if map_kind != self.map_kind or map_number != self.map_number:
            self.map_kind   = map_kind
            self.map_number = map_number
            self._map_obj   = Map(map_kind, map_number)
            self._fresh     = None

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
        # FIX: store expected totals once per episode from the clean map
        self._expected_carrots = mi.carrot_total
        self._expected_eggs    = mi.egg_total

        self.bobby = Bobby(
            start_frame=0, start_time=0,
            coord_src=mi.coord_start,
        )
        self.bobby.state      = State.Down
        self.bobby.coord_dest = self.bobby.coord_src
        self.step_count   = 0
        self.episode_done = False

    def step(self, action: int) -> Tuple[float, bool, Dict[str, object]]:
        """One RL step. Returns (0.0, done, info). Reward shaped externally."""
        assert self.bobby is not None and self.map_info is not None
        b  = self.bobby
        md = self.map_info.data

        before_carrot = b.carrot_count
        before_egg    = b.egg_count

        # ── movement ─────────────────────────────────────────
        desired = self.ACTIONS[action]
        b.state      = desired
        b.coord_dest = b.coord_src
        b.update_dest(md)
        invalid_move = (b.coord_dest == b.coord_src)

        moved = (b.coord_src != b.coord_dest)
        if moved:
            old_pos = b.coord_src[0]  + b.coord_src[1]  * GRID_SIZE
            new_pos = b.coord_dest[0] + b.coord_dest[1] * GRID_SIZE

            # ── leaving tile ──────────────────────────────────
            ot = md[old_pos]
            if   ot == 24: md[old_pos] = 25
            elif ot == 25: md[old_pos] = 26
            elif ot == 26: md[old_pos] = 27
            elif ot == 27: md[old_pos] = 24
            elif ot == 28: md[old_pos] = 29
            elif ot == 29: md[old_pos] = 28
            elif ot == 30: md[old_pos] = 31          # crumble → spike
            elif ot == 45:                           # egg (standing on it collects it)
                md[old_pos] = 46
                b.egg_count += 1

            # ── arriving tile ─────────────────────────────────
            nt = md[new_pos]
            if   nt == 19:                           # carrot
                md[new_pos] = 20
                b.carrot_count += 1
            elif nt == 22: _apply_switch(md, 22)
            elif nt == 32: md[new_pos] = 18; b.key_gray   += 1
            elif nt == 33 and b.key_gray   > 0: md[new_pos] = 18; b.key_gray   -= 1
            elif nt == 34: md[new_pos] = 18; b.key_yellow += 1
            elif nt == 35 and b.key_yellow > 0: md[new_pos] = 18; b.key_yellow -= 1
            elif nt == 36: md[new_pos] = 18; b.key_red    += 1
            elif nt == 37 and b.key_red    > 0: md[new_pos] = 18; b.key_red    -= 1
            elif nt == 38: _apply_switch(md, 38)
            elif nt == 40: b.next_state = State.Left
            elif nt == 41: b.next_state = State.Right
            elif nt == 42: b.next_state = State.Up
            elif nt == 43: b.next_state = State.Down
            elif nt == 31: b.dead = True             # spike kills on entry

            b.coord_src = b.coord_dest

        # ── death check on current tile ───────────────────────
        cur_pos  = b.coord_src[0] + b.coord_src[1] * GRID_SIZE
        cur_tile = md[cur_pos]
        if cur_tile == 31:
            b.dead = True

        # ── FIX: robust finish detection ─────────────────────
        # all_collected checks actual counts against ORIGINAL totals (not live map).
        # Previous code used self.map_info.carrot_total which is correct, but
        # _is_finished() had a subtle bug: if carrot_total==0 it only checked eggs.
        # A map with BOTH carrots and eggs would incorrectly finish on eggs alone.
        all_collected = self._is_finished()

        # on_finish: agent must be on tile 44 AND all collected AND alive
        on_finish = (cur_tile == 44) and all_collected and not b.dead

        carrot_delta = b.carrot_count - before_carrot
        egg_delta    = b.egg_count    - before_egg

        # Evaluate reachability loss if a crumble was destroyed
        destroyed_crumble = moved and ot == 30
        carrots_lost = 0
        finish_lost = False

        if destroyed_crumble:
            # We just left a crumble, converting it to a spike (31)
            px, py = b.coord_src
            reachable_count, _, finish_reachable, _ = _bfs_reachable_carrot_info(md, px, py)
            
            if not all_collected:
                remaining = (self._expected_carrots - b.carrot_count) + (self._expected_eggs - b.egg_count)
                if reachable_count < remaining:
                    carrots_lost = remaining - reachable_count
            
            # Check finish reachability
            original_has_finish = (44 in self._fresh.data) if self._fresh else False
            if original_has_finish and not finish_reachable:
                finish_lost = True

        self.step_count += 1
        # FIX: Do NOT terminate on crumble reachability loss — let the agent
        # continue and learn from the penalty. Terminating here killed 30%+ of
        # episodes instantly on crumble-heavy levels (5,6,7), preventing learning.
        done = b.dead or on_finish or (self.step_count >= self.max_steps)
        self.episode_done = done

        return 0.0, done, {
            "collected_carrot": carrot_delta,
            "collected_egg":    egg_delta,
            "all_collected":    all_collected,
            "invalid_move":     invalid_move,
            "dead":             b.dead,
            "level_completed":  on_finish,
            "position":         b.coord_src,
            "moved":            moved,
            "carrots_lost":     carrots_lost,
            "finish_lost":      finish_lost,
            "destroyed_crumble": destroyed_crumble,
            "destroyed_crumble_safe": destroyed_crumble and carrots_lost == 0 and not finish_lost,
        }

    def _is_finished(self) -> bool:
        """FIX: check BOTH carrots AND eggs against original totals.

        Original bug: if carrot_total > 0, only checked carrots (ignored eggs).
        If carrot_total == 0, only checked eggs.
        Maps with both types would never finish if eggs weren't also counted.

        Correct rule (matching game.py): if carrot_total > 0, need all carrots.
        If carrot_total == 0 and egg_total > 0, need all eggs.
        This matches the original game logic exactly.
        """
        b = self.bobby
        assert b is not None
        # Use stored expected totals (immune to map_info mutation)
        if self._expected_carrots > 0:
            return b.carrot_count >= self._expected_carrots
        if self._expected_eggs > 0:
            return b.egg_count >= self._expected_eggs
        # Map has no collectibles — finished immediately
        return True

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────
# Observation encoding
# ─────────────────────────────────────────────────────────────

def _semantic_channels(env: FastBobbyEnv) -> Tuple[np.ndarray, np.ndarray]:
    b  = env.bobby
    mi = env.map_info
    assert b is not None and mi is not None

    data_arr      = np.asarray(mi.data, dtype=np.uint8)
    all_collected = env._is_finished()

    np.copyto(_CH_BUF, _TILE_CH[data_arr])
    _CH_BUF[data_arr == 44] = 3 if all_collected else 4

    _GRID_BUF.fill(0)
    _GRID_BUF[_CH_BUF.astype(np.int32), _FLAT_IDX] = 1
    grid = _GRID_BUF.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).copy()

    px, py = b.coord_src
    grid[11, py, px] = 1
    if all_collected:
        grid[12, :, :] = 1

    # Inventory vector
    remaining = (mi.carrot_total - b.carrot_count) + (mi.egg_total - b.egg_count)
    remaining  = max(0, remaining)   # clamp — counts can't go negative but guard it
    denom      = max(1, mi.carrot_total + mi.egg_total)
    rem_norm   = min(1.0, remaining / denom)

    if not all_collected:
        mask = (data_arr == 19) | (data_arr == 45)
        if mask.any():
            _, bfs_dist, _, _ = _bfs_reachable_carrot_info(data_arr, px, py)
            
            if bfs_dist >= 0:
                md_norm = float(bfs_dist) / (GRID_SIZE * 2)
            else:
                # Fallback to Manhattan if BFS couldn't find a path (unreachable)
                idx     = np.where(mask)[0]
                xs, ys  = idx % GRID_SIZE, idx // GRID_SIZE
                md_norm = float(np.min(np.abs(xs - px) + np.abs(ys - py))) / (GRID_SIZE * 2)
        else:
            md_norm = 0.0
    else:
        # Use BFS distance to finish if reachable, otherwise Manhattan fallback
        _, _, fin_reach, fin_dist = _bfs_reachable_carrot_info(data_arr, px, py)
        if fin_reach and fin_dist >= 0:
            md_norm = float(fin_dist) / (GRID_SIZE * 2)
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
# Reward shaping
# ─────────────────────────────────────────────────────────────

def _shape_reward(
    info:         Dict[str, object],
    cfg:          DQNConfig,
    level_cfg:    Dict,
    prev_inv:     np.ndarray,
    curr_inv:     np.ndarray,
    env:          FastBobbyEnv,
    visit_counts: Optional[np.ndarray] = None,
    prev_collected: int = 0,
) -> float:
    r = cfg.step_penalty
    progress_frac = 0.0
    rq_carrots = 0
    if env.bobby is not None and env.map_info is not None:
        total_items = env.map_info.carrot_total + env.map_info.egg_total
        done_items = env.bobby.carrot_count + env.bobby.egg_count
        rq_carrots = total_items - done_items
        progress_frac = float(done_items) / float(max(1, total_items))

    if info["invalid_move"]:
        r += cfg.invalid_move_penalty
    if info["dead"]:
        r += cfg.death_penalty

    carrot_delta = int(info["collected_carrot"])
    egg_delta    = int(info["collected_egg"])
    if carrot_delta > 0:
        r += cfg.carrot_bonus * carrot_delta
    if egg_delta > 0:
        r += cfg.egg_bonus * egg_delta

    # Make late collectibles increasingly valuable to avoid partial-progress local optima.
    if (carrot_delta > 0 or egg_delta > 0) and progress_frac > 0:
        r += cfg.late_collection_bonus_scale * progress_frac * float(carrot_delta + egg_delta)

    if info["level_completed"]:
        r += cfg.completion_bonus

    all_collected = bool(info["all_collected"])
    level_done    = bool(info["level_completed"])

    # FIX: all-collected bonus — fire once when transitioning to all-collected state
    if all_collected and (carrot_delta > 0 or egg_delta > 0):
        # Just became all-collected this step
        if env.bobby is not None and env.map_info is not None:
            now = env.bobby.carrot_count + env.bobby.egg_count
            tot = env.map_info.carrot_total + env.map_info.egg_total
            if now >= tot:
                r += cfg.all_collected_bonus

    # Distance shaping toward nearest target / finish
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])
    r += level_cfg["distance_scale"] * dist_delta

    if all_collected and not level_done:
        r += level_cfg["post_penalty"]

    if info.get("destroyed_crumble_safe", False):
        r += cfg.safe_crumble_bonus

    # Reachability loss penalty
    carrots_lost = info.get("carrots_lost", 0)
    if carrots_lost > 0:
        r += cfg.reachability_loss_penalty * carrots_lost

    if info.get("finish_lost", False):
        # Heavy penalty for making finish tile unreachable
        r += cfg.reachability_loss_penalty * 2.0

    # Anti-oscillation revisit penalty
    if visit_counts is not None and env.bobby is not None:
        px, py = env.bobby.coord_src
        visits = int(visit_counts[px + py * GRID_SIZE])
        if visits > 2:
            revisit_scale = max(0.05, 1.0 - (1.25 * progress_frac))
            r += cfg.revisit_penalty * revisit_scale * min(visits - 2, 5)   # cap at 5x

    return r


# ─────────────────────────────────────────────────────────────
# Replay buffer — recency-biased sampling
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

    def push(self,
             grid: np.ndarray, inv: np.ndarray, action: int, reward: float,
             next_grid: np.ndarray, next_inv: np.ndarray, done: bool) -> None:
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
        """Sample with recency bias — 40% from newest 25% of buffer."""
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
# Dueling DDQN — GroupNorm (compile-safe)
# ─────────────────────────────────────────────────────────────

class DuelingDQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
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

        self.use_amp = cfg.use_amp and (device.type == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # FIX: use torch.no_grad() NOT inference_mode — inference_mode marks
        # output tensors as inference tensors which crash torch.compile CUDA
        # graph capture with "Inplace update to inference tensor" error.
        # Also: allocate fresh tensors each call (no pre-allocated buffers) —
        # inplace writes to pre-allocated buffers inside no_grad can still
        # interfere with the computation graph on some torch versions.
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

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                q_pred = self.policy_net(sg_t, sv_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    best_a = self.policy_net(nsg_t, nsv_t).argmax(1, keepdim=True)
                    q_next = self.target_net(nsg_t, nsv_t).gather(1, best_a).squeeze(1)
                    target = r_t + (1.0 - d_t) * self.cfg.gamma * q_next
                loss = self.loss_fn(q_pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_pred = self.policy_net(sg_t, sv_t).gather(1, a_t).squeeze(1)
            with torch.no_grad():
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
        def _sd(m: nn.Module) -> dict:
            return getattr(m, "_orig_mod", m).state_dict()
        torch.save({
            "policy": _sd(self.policy_net), "target": _sd(self.target_net),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon, "total_steps": self.total_steps,
            "level": level, "map_kind": map_kind,
            "grid_channels": GRID_CHANNELS, "inv_features": INV_FEATURES,
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
        torch.backends.cudnn.benchmark   = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32  = True


def _make_env(map_kind: str, level: int, max_steps: int) -> FastBobbyEnv:
    return FastBobbyEnv(map_kind=map_kind, map_number=level, max_steps=max_steps)


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
    stats      = _load_map_stats(map_kind, level)
    env        = _make_env(map_kind, level, max_steps)

    reward_hist:    List[float] = []
    success_hist:   List[float] = []
    collected_hist: List[float] = []
    progress_hist:  List[float] = []
    loss_win: Deque[float]      = deque(maxlen=cfg.report_every)
    best_success    = -1.0
    early_stop_wins = 0
    stagnation_ctr  = 0
    best_progress_seen = 0.0
    plateau_windows = 0
    ever_succeeded = False

    print(f"  max_steps={max_steps} | episodes={n_episodes} | "
          f"dist_scale={level_cfg['distance_scale']} | "
          f"post_pen={level_cfg['post_penalty']} | "
          f"carrots={stats['carrots']} eggs={stats['eggs']} "
          f"crumbles={stats['crumbles']} conveyors={stats['conveyors']} | "
          f"batch={cfg.batch_size} | lr={cfg.lr} | "
          f"carrot_bonus={cfg.carrot_bonus} | eps_decay={cfg.epsilon_decay} | "
          f"train_every={cfg.train_every_steps} | amp={cfg.use_amp}")

    t0              = time.time()
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

        visit_counts = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.int16)
        if env.bobby is not None:
            sx, sy = env.bobby.coord_src
            visit_counts[sx + sy * GRID_SIZE] = 1

        prev_total_collected = 0

        while not done and steps < max_steps:
            # Replay warmup: first few episodes are fully random to seed useful transitions.
            if episode <= cfg.warmup_episodes:
                action = random.randint(0, agent.n_actions - 1)
            else:
                action = agent.select_action(grid, inv)
            _, done, info = env.step(action)
            next_grid, next_inv = _semantic_channels(env)

            if env.bobby is not None:
                px, py = env.bobby.coord_src
                visit_counts[px + py * GRID_SIZE] += 1

            shaped = _shape_reward(
                info, cfg, level_cfg, prev_inv, next_inv, env,
                visit_counts, prev_total_collected,
            )

            # Terminal failure penalty (apply once, not per-step):
            # if episode ends without level completion, penalize remaining collectibles.
            # Special case: if we terminated early due to reachability loss, the penalty is mostly applied
            # by reachability_loss_penalty, but we still apply incomplete_penalty.
            if done and not info.get("level_completed") and env.bobby is not None and env.map_info is not None:
                total = env.map_info.carrot_total + env.map_info.egg_total
                done_now = env.bobby.carrot_count + env.bobby.egg_count
                rem_frac = 1.0 - (float(done_now) / float(max(1, total)))
                if rem_frac > 0.0:
                    shaped += cfg.incomplete_penalty * rem_frac

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            if info.get("level_completed") and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)
            elif info.get("all_collected") and not info.get("level_completed") and cfg.all_collected_oversample > 0:
                for _ in range(cfg.all_collected_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps  += 1
            total_env_steps    += 1
            loss = agent.optimize_step()
            if loss is not None:
                loss_win.append(loss)

            if env.bobby is not None:
                prev_total_collected = env.bobby.carrot_count + env.bobby.egg_count

            prev_inv  = next_inv
            grid, inv = next_grid, next_inv
            ep_r  += shaped
            steps += 1

        agent.decay_epsilon()

        success   = 1.0 if info.get("level_completed") else 0.0
        if success > 0.0:
            ever_succeeded = True
        collected = 1.0 if info.get("all_collected")   else 0.0

        total_col = (env.map_info.carrot_total + env.map_info.egg_total
                     if env.map_info else 1)
        done_col  = (env.bobby.carrot_count + env.bobby.egg_count
                     if env.bobby else 0)
        progress  = float(done_col) / float(max(1, total_col))
        if progress > best_progress_seen:
            best_progress_seen = progress

        # Keep exploration alive until first real success.
        if not ever_succeeded:
            agent.epsilon = max(agent.epsilon, cfg.pre_success_epsilon_floor)

        reward_hist.append(ep_r)
        success_hist.append(success)
        collected_hist.append(collected)
        progress_hist.append(progress)

        if episode % cfg.report_every == 0 or episode == 1:
            n       = cfg.report_every
            avg_r   = float(np.mean(reward_hist[-n:]))
            avg_s   = float(np.mean(success_hist[-n:]))
            avg_c   = float(np.mean(collected_hist[-n:]))
            avg_p   = float(np.mean(progress_hist[-n:]))
            avg_l   = float(np.mean(loss_win)) if loss_win else 0.0
            elapsed = time.time() - t0
            sps     = total_env_steps / max(elapsed, 1e-6)
            eta_h   = ((n_episodes - episode) * max_steps) / max(sps, 1) / 3600
            print(
                f"[L{level}] ep={episode:5d} | "
                f"reward={avg_r:8.1f} | "
                f"progress={avg_p:5.1%} | "
                f"collected={avg_c:5.1%} | "
                f"success={avg_s:5.1%} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_l:.4f} | "
                f"sps={sps:6.0f} | "
                f"ETA={eta_h:.1f}h"
            )

            # Stagnation rescue: reset epsilon if making zero progress for 4 windows
            if avg_s == 0.0 and avg_p < 0.05 and episode >= 4 * cfg.report_every:
                stagnation_ctr += 1
                if stagnation_ctr >= 4:
                    old = agent.epsilon
                    agent.epsilon = max(agent.epsilon, 0.85)
                    print(f"[L{level}] Stagnation rescue: eps {old:.3f} → {agent.epsilon:.3f}")
                    stagnation_ctr = 0
            else:
                stagnation_ctr = 0

            # Plateau rescue: progress exists but never reaches completion.
            if avg_s == 0.0 and 0.30 <= avg_p < 0.95 and episode >= 6 * cfg.report_every:
                recent_best = float(np.max(progress_hist[-n:])) if progress_hist else 0.0
                if recent_best <= (best_progress_seen - 0.02):
                    plateau_windows += 1
                else:
                    plateau_windows = 0
                if plateau_windows >= 2:
                    old = agent.epsilon
                    agent.epsilon = max(agent.epsilon, 0.55)
                    print(f"[L{level}] Plateau rescue: eps {old:.3f} → {agent.epsilon:.3f}")
                    plateau_windows = 0

            if avg_s >= 0.90:
                early_stop_wins += 1
                if early_stop_wins >= 2:
                    print(f"[L{level}] Early stop: success >= 90% for 2 windows.")
                    break
            else:
                early_stop_wins = 0

        if ckpt_path is not None and episode % cfg.save_every == 0:
            n = min(cfg.report_every, len(success_hist))
            rs = float(np.mean(success_hist[-n:]))
            if rs > best_success:
                best_success = rs
                best_path    = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(best_path, level=level, map_kind=map_kind)
                print(f"  [ckpt] New best: success={rs:.1%} → {best_path.name}")

    env.close()
    return {
        "mean_reward":        float(np.mean(reward_hist))    if reward_hist    else 0.0,
        "success_rate":       float(np.mean(success_hist))   if success_hist   else 0.0,
        "all_collected_rate": float(np.mean(collected_hist)) if collected_hist else 0.0,
        "progress_rate":      float(np.mean(progress_hist))  if progress_hist  else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Play / evaluation
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
        observation_mode="full", local_view_size=3, include_inventory=True,
        headless=not render, max_steps=max_steps, reward_config=silent_rc,
    )

    agent = DQNAgent(n_actions=full_env.action_space_n, cfg=cfg, device=device)
    meta  = agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    print(f"Loaded {model_path}  (trained level={meta['level']}, kind={meta['map_kind']})")

    proxy = FastBobbyEnv(map_kind, map_number, max_steps)

    for ep in range(1, episodes + 1):
        full_env.set_map(map_kind=map_kind, map_number=map_number)
        full_env.reset()
        proxy.map_info = full_env.map_info
        proxy.bobby    = full_env.bobby
        if proxy.bobby is not None:
            proxy._expected_carrots = full_env.map_info.carrot_total if full_env.map_info else 0
            proxy._expected_eggs    = full_env.map_info.egg_total    if full_env.map_info else 0
        grid, inv = _semantic_channels(proxy)

        done = False; steps = 0; total_r = 0.0
        info: Dict[str, object] = {}

        while not done and steps < max_steps:
            action               = agent.select_action(grid, inv)
            _, raw_r, done, info = full_env.step(action)
            proxy.map_info       = full_env.map_info
            proxy.bobby          = full_env.bobby
            grid, inv            = _semantic_channels(proxy)
            total_r             += raw_r
            steps               += 1
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
    p = argparse.ArgumentParser(description="Bobby Carrot DQN")
    p.add_argument("--play",                action="store_true")
    p.add_argument("--map-kind",            default="normal", choices=["normal","egg"])
    p.add_argument("--map-number",          type=int,   default=1)
    p.add_argument("--levels",              type=int,   nargs="+", default=None)
    p.add_argument("--individual-levels",   action="store_true")
    p.add_argument("--episodes-per-level",  type=int,   default=None)
    p.add_argument("--max-steps",           type=int,   default=None)
    p.add_argument("--batch-size",          type=int,   default=256)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--epsilon-start",       type=float, default=1.0)
    p.add_argument("--epsilon-min",         type=float, default=0.05)
    p.add_argument("--epsilon-decay",       type=float, default=0.9996)
    p.add_argument("--completion-bonus",    type=float, default=1200.0)
    p.add_argument("--death-penalty",       type=float, default=-25.0)
    p.add_argument("--carrot-bonus",        type=float, default=18.0)
    p.add_argument("--egg-bonus",           type=float, default=18.0)
    p.add_argument("--all-collected-bonus", type=float, default=220.0)
    p.add_argument("--late-collection-bonus-scale", type=float, default=6.0)
    p.add_argument("--incomplete-penalty", type=float, default=-80.0)
    p.add_argument("--crumble-penalty",     type=float, default=-2.0)
    p.add_argument("--invalid-move-penalty",type=float, default=-0.15)
    p.add_argument("--step-penalty",        type=float, default=-0.005)
    p.add_argument("--revisit-penalty",     type=float, default=-0.04)
    p.add_argument("--terminal-oversample", type=int,   default=48)
    p.add_argument("--all-collected-oversample", type=int, default=10)
    p.add_argument("--warmup-episodes",     type=int,   default=20,
                   help="Random-policy warmup episodes per level before epsilon-greedy training")
    p.add_argument("--train-every",         type=int,   default=2)
    p.add_argument("--observation-mode",    default="full", choices=["full","local","compact"])
    p.add_argument("--report-every",        type=int,   default=100)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--model-path",
                   default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",
                   default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes",       type=int,   default=5)
    p.add_argument("--no-render",           action="store_true")
    p.add_argument("--render-fps",          type=float, default=5.0)
    p.add_argument("--pre-success-epsilon-floor", type=float, default=0.25)
    p.add_argument("--resume",              action="store_true",
                   help="Resume from --model-path checkpoint if it exists")
    p.add_argument("--no-compile",          action="store_true")
    p.add_argument("--no-amp",              action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
        epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        completion_bonus=args.completion_bonus, death_penalty=args.death_penalty,
        carrot_bonus=args.carrot_bonus, egg_bonus=args.egg_bonus,
        all_collected_bonus=args.all_collected_bonus,
        late_collection_bonus_scale=args.late_collection_bonus_scale,
        incomplete_penalty=args.incomplete_penalty,
        crumble_penalty=args.crumble_penalty,
        invalid_move_penalty=args.invalid_move_penalty,
        step_penalty=args.step_penalty, revisit_penalty=args.revisit_penalty,
        terminal_oversample=args.terminal_oversample,
        all_collected_oversample=args.all_collected_oversample,
        warmup_episodes=args.warmup_episodes,
        train_every_steps=args.train_every,
        observation_mode=args.observation_mode,
        report_every=args.report_every, seed=args.seed,
        pre_success_epsilon_floor=args.pre_success_epsilon_floor,
        use_amp=not args.no_amp,
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
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | AMP: {cfg.use_amp} | train_every: {cfg.train_every_steps}")

    levels = [int(l) for l in (args.levels or [args.map_number])]
    for lvl in levels:
        _apply_cli_overrides(lvl, args)

    probe_env = _make_env(args.map_kind, levels[0], 10)
    probe_env.reset()
    n_actions = probe_env.action_space_n
    probe_env.close()

    if args.individual_levels:
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

    if args.resume and model_path.exists():
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
            f"progress={summary['progress_rate']:.2%} | "
            f"all_collected={summary['all_collected_rate']:.2%} | "
            f"success={summary['success_rate']:.2%}"
        )

    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"\nFinal model saved to: {model_path}")


if __name__ == "__main__":
    _main()
