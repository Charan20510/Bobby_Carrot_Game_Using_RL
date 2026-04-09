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

from bobby_carrot.game import Bobby, Map, MapInfo, State, WIDTH_POINTS, HEIGHT_POINTS
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
GRID_CHANNELS = 14   # +1 channel: crumble-adjacency map
INV_FEATURES  = 7    # +2: new_area_dist_norm, crumble_bonus_available

_TILE_CH = np.zeros(256, dtype=np.int8)
for _i in range(256):
    if   _i < 18:                _TILE_CH[_i] = 0   # wall
    elif _i == 18:               _TILE_CH[_i] = 1   # floor
    elif _i == 19:               _TILE_CH[_i] = 2   # carrot
    elif _i == 45:               _TILE_CH[_i] = 3   # egg
    elif _i == 44:               _TILE_CH[_i] = 4   # finish (open) / 5 (locked)
    elif _i in (31, 46):         _TILE_CH[_i] = 6   # spike / used egg
    elif _i in (32, 34, 36):     _TILE_CH[_i] = 7   # key pickup
    elif _i in (33, 35, 37):     _TILE_CH[_i] = 8   # locked door
    elif _i == 30:               _TILE_CH[_i] = 9   # crumble ← explicit channel
    elif _i in (40, 41, 42, 43): _TILE_CH[_i] = 10  # conveyor
    elif _i == 22:               _TILE_CH[_i] = 11  # red switch
    elif _i == 38:               _TILE_CH[_i] = 11  # blue switch
    else:                        _TILE_CH[_i] = 12  # other

_FLAT_IDX = np.arange(256, dtype=np.int32)
_CH_BUF   = np.zeros(256, dtype=np.int8)
_GRID_BUF = np.zeros((GRID_CHANNELS, 256), dtype=np.uint8)
_RED_MAP  = {22:23, 23:22, 24:25, 25:26, 26:27, 27:24, 28:29, 29:28}
_BLUE_MAP = {38:39, 39:38, 40:41, 41:40, 42:43, 43:42}


# ─────────────────────────────────────────────────────────────
# BFS utilities — wall-aware, spike-aware
# ─────────────────────────────────────────────────────────────

def _bfs_full(md: List[int], sx: int, sy: int, spikes: Optional[set] = None) -> Dict[Tuple[int,int], int]:
    """BFS returning distance-dict from (sx,sy). Treats spikes (tile 31) as walls.
    Crumbles (tile 30) are PASSABLE (stepping them is a move).
    Returns {(x,y): dist} for all reachable walkable tiles.
    """
    if spikes is None:
        spikes = set()
    dist_map: Dict[Tuple[int,int], int] = {}
    q: deque = deque()
    q.append((sx, sy, 0))
    dist_map[(sx, sy)] = 0
    while q:
        x, y, d = q.popleft()
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < 16 and 0 <= ny < 16 and (nx,ny) not in dist_map:
                if (nx,ny) in spikes:
                    continue
                nt = md[nx+ny*16]
                if nt >= 18:   # passable: floor, crumble, carrot, finish, etc.
                    dist_map[(nx,ny)] = d+1
                    q.append((nx, ny, d+1))
    return dist_map


def _bfs_reachability(md: List[int], sx: int, sy: int) -> Tuple[int, bool, int]:
    """Reachability BFS: returns (reachable_targets, finish_reachable, finish_dist).
    Used for crumble safety checks. Spikes treated as walls, crumbles passable.
    """
    dist_map = _bfs_full(md, sx, sy)
    rc = 0; fr = False; fd = 9999
    for (x,y), d in dist_map.items():
        t = md[x+y*16]
        if t == 19 or t == 45:
            rc += 1
        elif t == 44:
            fr = True
            fd = min(fd, d)
    return rc, fr, (fd if fr else -1)


def _bfs_nearest_target(md: List[int], sx: int, sy: int, all_collected: bool) -> int:
    """BFS distance to nearest uncollected carrot/egg (or finish if all_collected).
    Fully wall-aware and spike-aware. Returns 32 if unreachable.
    """
    dist_map = _bfs_full(md, sx, sy)
    best = 32
    for (x,y), d in dist_map.items():
        t = md[x+y*16]
        if all_collected:
            if t == 44:
                best = min(best, d)
        else:
            if t == 19 or t == 45:
                best = min(best, d)
    return best


def _count_reachable_targets(md: List[int], sx: int, sy: int) -> int:
    """Count reachable uncollected carrots/eggs via BFS."""
    dist_map = _bfs_full(md, sx, sy)
    count = 0
    for (x,y) in dist_map:
        t = md[x+y*16]
        if t == 19 or t == 45:
            count += 1
    return count


def _build_crumble_adjacency(md: List[int]) -> np.ndarray:
    """Build a 16x16 binary map: 1 where stepping that floor tile puts you
    adjacent to an unstepped crumble. Helps agent identify crumble opportunities.
    """
    arr = np.zeros(256, dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            t = md[x+y*16]
            if t < 18:
                continue
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < 16 and 0 <= ny < 16:
                    if md[nx+ny*16] == 30:   # crumble neighbour
                        arr[x+y*16] = 1
                        break
    return arr


# ─────────────────────────────────────────────────────────────
# Per-level config
# ─────────────────────────────────────────────────────────────

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 250,  "episodes": 2000,  "distance_scale": 2.0, "post_penalty": -0.5},
    2:  {"max_steps": 280,  "episodes": 2000,  "distance_scale": 2.0, "post_penalty": -0.5},
    3:  {"max_steps": 320,  "episodes": 2500,  "distance_scale": 2.0, "post_penalty": -0.5},
    4:  {"max_steps": 400,  "episodes": 3000,  "distance_scale": 2.0, "post_penalty": -0.5},
    # ── Level 5: 7 isolated components, all gated by crumbles ──────────────────
    # Correct path requires stepping (7,9) horizontally (not vertically!),
    # then (7,7), then (7,5) for finish.  Wings via (1,8)+(1,4) and (13,8)+(13,4).
    # Needs far more episodes and a high crumble_area_bonus to reward bridge steps.
    5:  {"max_steps": 600,  "episodes": 15000, "distance_scale": 2.0, "post_penalty": -0.1,
         "crumble_area_bonus": 25.0,   # big reward per newly reachable carrot after crumble step
         "stall_steps": 350,           # longer patience for multi-crumble paths
         "new_tile_bonus": 0.8},       # stronger exploration bonus
    # ── Level 6: 6 components, complex crumble graph ────────────────────────────
    6:  {"max_steps": 550,  "episodes": 12000, "distance_scale": 2.0, "post_penalty": -0.1,
         "crumble_area_bonus": 25.0,
         "stall_steps": 350,
         "new_tile_bonus": 0.8},
    7:  {"max_steps": 650,  "episodes": 10000, "distance_scale": 2.0, "post_penalty": -0.3,
         "crumble_area_bonus": 20.0, "stall_steps": 300, "new_tile_bonus": 0.5},
    8:  {"max_steps": 400,  "episodes": 3000,  "distance_scale": 2.0, "post_penalty": -0.5},
    9:  {"max_steps": 450,  "episodes": 4000,  "distance_scale": 2.0, "post_penalty": -0.5},
    10: {"max_steps": 500,  "episodes": 5000,  "distance_scale": 2.0, "post_penalty": -0.5},
}

def _get_level_cfg(level: int) -> Dict:
    base = {
        "max_steps": 350, "episodes": 3000,
        "distance_scale": 2.0, "post_penalty": -0.5,
        "crumble_area_bonus": 15.0, "stall_steps": 200, "new_tile_bonus": 0.3,
    }
    cfg = dict(base)
    cfg.update(LEVEL_CONFIG.get(level, {}))
    return cfg


@lru_cache(maxsize=64)
def _load_map_stats(map_kind: str, level: int) -> Dict[str, int]:
    mi  = Map(map_kind, level).load_map_info()
    arr = np.asarray(mi.data, dtype=np.uint8)
    return {
        "carrots":    int(np.count_nonzero(arr == 19)),
        "eggs":       int(np.count_nonzero(arr == 45)),
        "crumbles":   int(np.count_nonzero(arr == 30)),
        "conveyors":  int(np.count_nonzero((arr >= 40) & (arr <= 43))),
        "has_finish": int(np.any(arr == 44)),
    }


# ─────────────────────────────────────────────────────────────
# DQNConfig
# ─────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    gamma:               float = 0.99
    lr:                  float = 3e-4
    batch_size:          int   = 256
    replay_capacity:     int   = 120_000
    min_replay_size:     int   = 1024
    target_update_steps: int   = 500
    train_every_steps:   int   = 2
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.03
    epsilon_decay:       float = 0.9997   # slower decay → more exploration for complex levels
    death_penalty:       float = -150.0
    carrot_bonus:        float = 30.0
    egg_bonus:           float = 30.0
    completion_bonus:    float = 500.0
    all_collected_bonus: float = 100.0
    # ── crumble-area reward: fired when a crumble step increases reachable targets ──
    # This is the KEY missing signal in the original code.
    crumble_area_bonus:  float = 25.0    # per newly reachable carrot
    safe_crumble_bonus:  float = 5.0     # small bonus when crumble is safe AND opens area
    reachability_loss_penalty: float = -80.0
    finish_loss_penalty: float = -160.0
    incomplete_penalty:  float = -30.0
    invalid_move_penalty:float = -0.1
    step_penalty:        float = -0.01
    revisit_penalty:     float = -0.03
    new_tile_bonus:      float = 0.3
    terminal_oversample: int   = 20
    all_collected_oversample: int = 5
    warmup_episodes:     int   = 10
    stall_steps:         int   = 200     # overridden per level
    report_every:        int   = 100
    save_every:          int   = 500
    seed:                int   = 42
    use_amp:             bool  = True
    pre_success_eps_floor: float = 0.0   # disabled


# ─────────────────────────────────────────────────────────────
# _apply_switch
# ─────────────────────────────────────────────────────────────

def _apply_switch(md: List[int], tile: int) -> None:
    lut = _RED_MAP if tile == 22 else _BLUE_MAP
    for i in range(256):
        v = md[i]
        if v in lut:
            md[i] = lut[v]


# ─────────────────────────────────────────────────────────────
# FastBobbyEnv
# ─────────────────────────────────────────────────────────────

class FastBobbyEnv:
    """Single-step game-logic environment. Death on spikes = real game."""

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
        self._expected_carrots: int = 0
        self._expected_eggs:    int = 0
        self._bfs_cache: Dict[Tuple[int, ...], Tuple] = {}
        self._map_state_id: int = 0
        # Track visited tiles for new-tile bonus
        self._visited_tiles: set = set()

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
        self._expected_carrots = mi.carrot_total
        self._expected_eggs    = mi.egg_total
        self.bobby = Bobby(start_frame=0, start_time=0, coord_src=mi.coord_start)
        self.bobby.state      = State.Down
        self.bobby.coord_dest = self.bobby.coord_src
        self.step_count   = 0
        self.episode_done = False
        self._bfs_cache.clear()
        self._map_state_id = 0
        self._visited_tiles = {mi.coord_start}

    def step(self, action: int) -> Tuple[float, bool, Dict[str, object]]:
        assert self.bobby is not None and self.map_info is not None
        b  = self.bobby
        md = self.map_info.data

        before_carrot = b.carrot_count
        before_egg    = b.egg_count

        # Conveyor override
        if b.next_state in (State.Left, State.Right, State.Up, State.Down):
            desired    = b.next_state
            b.next_state = None
        else:
            desired = self.ACTIONS[action]

        b.state      = desired
        b.coord_dest = b.coord_src
        b.update_dest(md)

        invalid_move = (b.coord_dest == b.coord_src)
        moved        = (b.coord_src  != b.coord_dest)
        map_mutated  = False
        ot           = 0
        destroyed_crumble = False
        reachable_before  = 0
        finish_reachable_before = False

        if moved:
            old_pos = b.coord_src[0]  + b.coord_src[1]  * GRID_SIZE
            new_pos = b.coord_dest[0] + b.coord_dest[1] * GRID_SIZE
            ot = md[old_pos]

            # ── Measure reachability BEFORE any crumble is destroyed ──────────
            if ot == 30:
                destroyed_crumble = True
                px, py = b.coord_src
                reachable_before = _count_reachable_targets(md, px, py)
                _, finish_reachable_before, _ = _bfs_reachability(md, px, py)

            # Leaving tile
            if   ot == 24: md[old_pos] = 25; map_mutated = True
            elif ot == 25: md[old_pos] = 26; map_mutated = True
            elif ot == 26: md[old_pos] = 27; map_mutated = True
            elif ot == 27: md[old_pos] = 24; map_mutated = True
            elif ot == 28: md[old_pos] = 29; map_mutated = True
            elif ot == 29: md[old_pos] = 28; map_mutated = True
            elif ot == 30: md[old_pos] = 31; map_mutated = True   # crumble → spike
            elif ot == 45: md[old_pos] = 46; b.egg_count += 1; map_mutated = True

            # Arriving tile
            nt = md[new_pos]
            if   nt == 19: md[new_pos] = 20; b.carrot_count += 1; map_mutated = True
            elif nt == 22: _apply_switch(md, 22); map_mutated = True
            elif nt == 32: md[new_pos] = 18; b.key_gray   += 1
            elif nt == 33 and b.key_gray   > 0: md[new_pos] = 18; b.key_gray   -= 1
            elif nt == 34: md[new_pos] = 18; b.key_yellow += 1
            elif nt == 35 and b.key_yellow > 0: md[new_pos] = 18; b.key_yellow -= 1
            elif nt == 36: md[new_pos] = 18; b.key_red    += 1
            elif nt == 37 and b.key_red    > 0: md[new_pos] = 18; b.key_red    -= 1
            elif nt == 38: _apply_switch(md, 38); map_mutated = True
            elif nt == 40: b.next_state = State.Left
            elif nt == 41: b.next_state = State.Right
            elif nt == 42: b.next_state = State.Up
            elif nt == 43: b.next_state = State.Down
            elif nt == 31: b.dead = True

            b.coord_src = b.coord_dest

        if map_mutated:
            self._map_state_id += 1
            self._bfs_cache.clear()

        cur_pos  = b.coord_src[0] + b.coord_src[1] * GRID_SIZE
        cur_tile = md[cur_pos]
        if cur_tile == 31:
            b.dead = True

        # ── New tile tracking ──────────────────────────────────────────────────
        is_new_tile = b.coord_src not in self._visited_tiles
        if moved:
            self._visited_tiles.add(b.coord_src)

        all_collected = self._is_finished()
        on_finish     = (cur_tile == 44) and all_collected and not b.dead
        carrot_delta  = b.carrot_count - before_carrot
        egg_delta     = b.egg_count    - before_egg

        # ── Crumble-area reward: newly reachable carrots AFTER stepping crumble ─
        # This is the KEY signal that was missing. The agent must be rewarded
        # for stepping crumbles that open new carrot clusters, not just penalised
        # for stepping ones that close them off.
        newly_reachable = 0
        finish_newly_lost = False
        carrots_lost = 0

        if destroyed_crumble and not b.dead:
            px, py = b.coord_src
            reachable_after = _count_reachable_targets(md, px, py)
            _, finish_reachable_after, _ = _bfs_reachability(md, px, py)

            # Positive: opened new areas
            if reachable_after > reachable_before:
                newly_reachable = reachable_after - reachable_before
            # Negative: closed off carrots
            if reachable_after < reachable_before:
                carrots_lost = reachable_before - reachable_after
            # Finish loss
            if finish_reachable_before and not finish_reachable_after:
                finish_newly_lost = True

        self.step_count += 1
        done = b.dead or on_finish or (self.step_count >= self.max_steps)
        self.episode_done = done

        return 0.0, done, {
            "collected_carrot":       carrot_delta,
            "collected_egg":          egg_delta,
            "all_collected":          all_collected,
            "invalid_move":           invalid_move,
            "dead":                   b.dead,
            "level_completed":        on_finish,
            "position":               b.coord_src,
            "moved":                  moved,
            "carrots_lost":           carrots_lost,
            "finish_lost":            finish_newly_lost,
            "destroyed_crumble":      destroyed_crumble,
            "newly_reachable":        newly_reachable,  # ← NEW: bonus signal
            "is_new_tile":            is_new_tile,      # ← NEW: for stall counter
        }

    def _is_finished(self) -> bool:
        b = self.bobby
        assert b is not None
        if self._expected_carrots > 0:
            return b.carrot_count >= self._expected_carrots
        if self._expected_eggs > 0:
            return b.egg_count >= self._expected_eggs
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

    data_arr      = np.array(mi.data, dtype=np.uint8)
    all_collected = env._is_finished()

    np.copyto(_CH_BUF, _TILE_CH[data_arr])
    # finish channel: 4 = locked, 5 = open
    _CH_BUF[data_arr == 44] = 5 if all_collected else 4

    _GRID_BUF.fill(0)
    for ch in range(13):
        mask = (_CH_BUF == ch)
        _GRID_BUF[ch][mask] = 1

    # Channel 13: crumble-adjacency (tiles next to an unstepped crumble)
    crumble_adj = _build_crumble_adjacency(mi.data)
    _GRID_BUF[13] = crumble_adj

    grid = _GRID_BUF.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).copy()

    # Agent position marker (overlay on channel 1 = floor)
    px, py = b.coord_src
    grid[1, py, px] = 2   # distinguishable value

    if all_collected:
        grid[5, :, :] = 1   # broadcast "finish open" hint

    remaining = max(0, (mi.carrot_total - b.carrot_count) + (mi.egg_total - b.egg_count))
    denom     = max(1, mi.carrot_total + mi.egg_total)
    rem_norm  = min(1.0, remaining / denom)

    # BFS distance — fully wall-aware, spike-aware
    bfs_dist  = _bfs_nearest_target(mi.data, px, py, all_collected)
    dist_norm = min(1.0, bfs_dist / (GRID_SIZE * 2))

    # Newly reachable carrot count if we step the nearest crumble
    # (gives the agent a forward-looking signal about crumble value)
    crumble_bonus_signal = _crumble_forward_value(mi.data, px, py)
    crumble_bonus_norm   = min(1.0, crumble_bonus_signal / max(1, remaining))

    # Distance to nearest crumble
    crumble_dist = _bfs_nearest_crumble(mi.data, px, py)
    crumble_dist_norm = min(1.0, crumble_dist / (GRID_SIZE * 2))

    inv = np.array([
        float(b.key_gray   > 0),
        float(b.key_yellow > 0),
        float(b.key_red    > 0),
        rem_norm,
        dist_norm,
        crumble_bonus_norm,   # ← NEW: how many new carrots would nearest crumble open
        crumble_dist_norm,    # ← NEW: how far to nearest crumble
    ], dtype=np.float32)

    return grid, inv


def _crumble_forward_value(md: List[int], px: int, py: int) -> int:
    """Estimate how many new carrots the NEAREST crumble (if stepped) would unlock.
    Fast heuristic: find nearest crumble via BFS, simulate stepping it, count new reachable.
    """
    dist_map = _bfs_full(md, px, py)
    best_crumble = None
    best_dist = 9999
    for (x,y), d in dist_map.items():
        if md[x+y*16] == 30 and d < best_dist:
            best_dist = d
            best_crumble = (x,y)
    if best_crumble is None:
        return 0
    before = _count_reachable_targets(md, px, py)
    # Simulate: step that crumble (it becomes spike)
    md2 = md.copy()
    md2[best_crumble[0] + best_crumble[1]*16] = 31
    # Agent would be on the other side
    after = _count_reachable_targets(md2, px, py)
    return max(0, after - before)


def _bfs_nearest_crumble(md: List[int], px: int, py: int) -> int:
    """BFS distance to nearest unstepped crumble tile."""
    dist_map = _bfs_full(md, px, py)
    best = 32
    for (x,y), d in dist_map.items():
        if md[x+y*16] == 30:
            best = min(best, d)
    return best


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
) -> float:
    r = cfg.step_penalty

    if info["invalid_move"]: r += cfg.invalid_move_penalty
    if info["dead"]:         r += cfg.death_penalty

    carrot_delta = int(info["collected_carrot"])
    egg_delta    = int(info["collected_egg"])
    b            = env.bobby
    mi           = env.map_info

    progress_frac = 0.0
    if b is not None and mi is not None:
        total = mi.carrot_total + mi.egg_total
        done  = b.carrot_count + b.egg_count
        progress_frac = float(done) / float(max(1, total))

    if carrot_delta > 0:
        r += cfg.carrot_bonus * carrot_delta
        r += 5.0 * progress_frac * carrot_delta   # late carrots worth more
    if egg_delta > 0:
        r += cfg.egg_bonus * egg_delta
        r += 5.0 * progress_frac * egg_delta

    if info["level_completed"]:
        r += cfg.completion_bonus

    all_collected = bool(info["all_collected"])
    level_done    = bool(info["level_completed"])

    # All-collected bonus (fires once on the collection step)
    if all_collected and (carrot_delta > 0 or egg_delta > 0):
        if b is not None and mi is not None:
            if b.carrot_count + b.egg_count >= mi.carrot_total + mi.egg_total:
                r += cfg.all_collected_bonus

    # ── Distance shaping using BFS-corrected distances ────────────────────────
    # inv[4] = BFS dist_norm. Decrease = moving closer = positive reward.
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])
    r += level_cfg["distance_scale"] * dist_delta

    # ── Crumble-area reward: KEY FIX ─────────────────────────────────────────
    # Reward agent for stepping crumbles that open new carrot clusters.
    # This was the missing signal causing the 70-78% plateau in level 5.
    newly_reachable = int(info.get("newly_reachable", 0))
    if newly_reachable > 0:
        crumble_area_bonus = level_cfg.get("crumble_area_bonus", cfg.crumble_area_bonus)
        r += crumble_area_bonus * newly_reachable
        r += cfg.safe_crumble_bonus   # small extra for the safe step itself

    # Penalty for destroying access to carrots/finish
    carrots_lost = int(info.get("carrots_lost", 0))
    if carrots_lost > 0:
        r += cfg.reachability_loss_penalty * carrots_lost
    if info.get("finish_lost"):
        r += cfg.finish_loss_penalty

    # Post-collection: penalise wandering after all collected but not at finish
    if all_collected and not level_done:
        r += level_cfg["post_penalty"]

    # Revisit penalty + new-tile bonus — using visit_counts array
    if visit_counts is not None and b is not None:
        vx, vy = b.coord_src
        visits = int(visit_counts[vx + vy * GRID_SIZE])
        if visits > 2:
            scale = max(0.05, 1.0 - 1.2 * progress_frac)
            r += cfg.revisit_penalty * scale * min(visits - 2, 5)

    # New-tile bonus from info flag (simpler than visit count for new areas)
    if bool(info.get("is_new_tile")):
        new_tile_bonus = level_cfg.get("new_tile_bonus", cfg.new_tile_bonus)
        r += new_tile_bonus

    return r


# ─────────────────────────────────────────────────────────────
# Replay buffer — CPU numpy, recency-biased sampling
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

    def push(self, grid, inv, action, reward, next_grid, next_inv, done) -> None:
        i           = self._ptr
        self._sg[i] = grid;      self._sv[i] = inv
        self._a[i]  = action;    self._r[i]  = reward
        self._ng[i] = next_grid; self._nv[i] = next_inv
        self._d[i]  = float(done)
        self._ptr   = (i + 1) % self._cap
        self._size  = min(self._size + 1, self._cap)

    def sample(self, n: int, recent_frac: float = 0.4) -> Tuple[np.ndarray, ...]:
        n_recent  = int(n * recent_frac)
        n_uniform = n - n_recent
        idx_uni   = np.random.randint(0, self._size, size=n_uniform)
        win   = max(n, self._size // 4)
        start = (self._ptr - win) % self._cap
        if self._size >= win:
            if start + win <= self._cap:
                pool = np.arange(start, start + win)
            else:
                pool = np.concatenate([np.arange(start, self._cap),
                                       np.arange(0, (start + win) % self._cap)])
            idx_rec = pool[np.random.randint(0, len(pool), size=n_recent)]
        else:
            idx_rec = np.random.randint(0, self._size, size=n_recent)
        idx = np.concatenate([idx_uni, idx_rec])
        np.random.shuffle(idx)
        return (
            self._sg[idx].astype(np.float32), self._sv[idx],
            self._a[idx], self._r[idx],
            self._ng[idx].astype(np.float32), self._nv[idx],
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
        self.current_episode = 0
        self.policy_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr, eps=1.5e-4)
        self.loss_fn   = nn.SmoothL1Loss()
        self.replay    = ReplayBuffer(cfg.replay_capacity)
        self.use_amp   = cfg.use_amp and (device.type == "cuda")
        self.scaler    = torch.amp.GradScaler("cuda") if self.use_amp else None

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            g = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).to(self.device, non_blocking=True)
            v = torch.from_numpy(inv).unsqueeze(0).to(self.device, non_blocking=True)
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
        def _sd(m): return getattr(m, "_orig_mod", m).state_dict()
        torch.save({
            "policy": _sd(self.policy_net), "target": _sd(self.target_net),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon, "total_steps": self.total_steps,
            "current_episode": getattr(self, "current_episode", 0),
            "level": level, "map_kind": map_kind,
            "grid_channels": GRID_CHANNELS, "inv_features": INV_FEATURES,
        }, path)

    def load(self, path: Path) -> Dict[str, object]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        def _base(m): return getattr(m, "_orig_mod", m)
        _base(self.policy_net).load_state_dict(ckpt["policy"])
        _base(self.target_net).load_state_dict(ckpt.get("target", ckpt["policy"]))
        if "optim" in ckpt: self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon     = float(ckpt.get("epsilon",     self.cfg.epsilon_start))
        self.total_steps = int  (ckpt.get("total_steps", 0))
        self.current_episode = int(ckpt.get("current_episode", 0))
        return {"level": ckpt.get("level", -1), "map_kind": ckpt.get("map_kind", "normal")}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

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
    agent:               DQNAgent,
    level:               int,
    cfg:                 DQNConfig,
    map_kind:            str,
    target_collect_rate: float = 0.90,
    target_success_rate: float = 0.90,
    ckpt_path:           Optional[Path] = None,
) -> Dict[str, float]:
    level_cfg  = _get_level_cfg(level)
    max_steps  = level_cfg["max_steps"]
    n_episodes = level_cfg["episodes"]
    # Per-level stall override
    stall_steps = level_cfg.get("stall_steps", cfg.stall_steps)
    stats      = _load_map_stats(map_kind, level)
    env        = _make_env(map_kind, level, max_steps)

    reward_hist:   List[float] = []
    success_hist:  List[float] = []
    collected_hist:List[float] = []
    progress_hist: List[float] = []
    loss_win: Deque[float]     = deque(maxlen=cfg.report_every)
    best_success   = -1.0
    early_stop_wins = 0
    stagnation_ctr  = 0
    ever_succeeded  = False
    best_progress   = 0.0

    print(f"  max_steps={max_steps} | episodes={n_episodes} | "
          f"dist={level_cfg['distance_scale']} | post={level_cfg['post_penalty']} | "
          f"crumble_area_bonus={level_cfg.get('crumble_area_bonus', cfg.crumble_area_bonus)} | "
          f"stall={stall_steps} | "
          f"carrots={stats['carrots']} crumbles={stats['crumbles']} | "
          f"death_penalty={cfg.death_penalty} | carrot_bonus={cfg.carrot_bonus} | "
          f"batch={cfg.batch_size} | lr={cfg.lr} | "
          f"train_every={cfg.train_every_steps} | amp={cfg.use_amp}")

    t0 = time.time(); total_env_steps = 0
    start_episode = agent.current_episode + 1
    end_episode = start_episode + n_episodes

    for episode in range(start_episode, end_episode):
        agent.current_episode = episode
        env.set_map(map_kind=map_kind, map_number=level)
        env.reset()
        grid, inv = _semantic_channels(env)
        done      = False; steps = 0; ep_r = 0.0
        info: Dict[str, object] = {}
        prev_inv = inv.copy()
        visit_counts = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.int16)
        if env.bobby is not None:
            sx, sy = env.bobby.coord_src
            visit_counts[sx + sy * GRID_SIZE] = 1

        # ── Stall detection: reset on new tile OR new carrot ──────────────────
        # OLD code only reset on carrot collection → agent trapped after center
        # carrots but before discovering crumble bridges.
        # NEW: reset also on any new tile visited (exploring new area).
        last_total_collected = 0
        last_visited_count   = len(env._visited_tiles)
        stall_counter        = 0

        while not done and steps < max_steps:
            if episode <= cfg.warmup_episodes and agent.total_steps == 0:
                action = random.randint(0, agent.n_actions - 1)
            else:
                action = agent.select_action(grid, inv)

            _, done, info = env.step(action)
            next_grid, next_inv = _semantic_channels(env)

            if env.bobby is not None:
                px, py = env.bobby.coord_src
                visit_counts[px + py * GRID_SIZE] += 1

            shaped = _shape_reward(info, cfg, level_cfg, prev_inv, next_inv, env, visit_counts)

            # ── Stall termination ─────────────────────────────────────────────
            now_col     = 0
            now_visited = len(env._visited_tiles)
            if env.bobby is not None:
                now_col = env.bobby.carrot_count + env.bobby.egg_count
            # Reset stall if: collected new carrot OR visited new tile
            if now_col > last_total_collected or now_visited > last_visited_count:
                last_total_collected = now_col
                last_visited_count   = now_visited
                stall_counter        = 0
            else:
                stall_counter += 1

            if not done and stall_counter >= stall_steps:
                done = True
                info["stall_terminated"] = True

            # Terminal failure penalty
            if done and not info.get("level_completed"):
                if env.bobby is not None and env.map_info is not None:
                    total   = env.map_info.carrot_total + env.map_info.egg_total
                    done_n  = env.bobby.carrot_count + env.bobby.egg_count
                    rem_frac = 1.0 - float(done_n) / float(max(1, total))
                    if rem_frac > 0:
                        shaped += cfg.incomplete_penalty * rem_frac

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            # Oversample important transitions
            n_os = (cfg.terminal_oversample
                    if info.get("level_completed") else
                    cfg.all_collected_oversample
                    if (info.get("all_collected") and not info.get("level_completed")) else
                    # Also oversample crumble-area reward transitions (key signal)
                    3 if int(info.get("newly_reachable", 0)) > 0 else 0)
            for _ in range(n_os):
                agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps += 1; total_env_steps += 1
            loss = agent.optimize_step()
            if loss is not None: loss_win.append(loss)

            prev_inv = next_inv; grid, inv = next_grid, next_inv
            ep_r += shaped; steps += 1

        agent.decay_epsilon()

        success   = 1.0 if info.get("level_completed") else 0.0
        if success > 0: ever_succeeded = True
        collected = 1.0 if info.get("all_collected") else 0.0
        total_col = (env.map_info.carrot_total + env.map_info.egg_total) if env.map_info else 1
        done_col  = (env.bobby.carrot_count + env.bobby.egg_count)       if env.bobby   else 0
        progress  = float(done_col) / float(max(1, total_col))
        if progress > best_progress: best_progress = progress

        reward_hist.append(ep_r); success_hist.append(success)
        collected_hist.append(collected); progress_hist.append(progress)

        if episode % cfg.report_every == 0 or episode == 1:
            n     = cfg.report_every
            avg_r = float(np.mean(reward_hist[-n:]))
            avg_s = float(np.mean(success_hist[-n:]))
            avg_c = float(np.mean(collected_hist[-n:]))
            avg_p = float(np.mean(progress_hist[-n:]))
            avg_l = float(np.mean(loss_win)) if loss_win else 0.0
            elapsed = time.time() - t0
            sps     = total_env_steps / max(elapsed, 1e-6)
            eta_h   = ((end_episode - episode) * max_steps) / max(sps, 1) / 3600
            print(f"[L{level}] ep={episode:5d} | reward={avg_r:8.1f} | "
                  f"progress={avg_p:5.1%} | collected={avg_c:5.1%} | success={avg_s:5.1%} | "
                  f"eps={agent.epsilon:.3f} | loss={avg_l:.4f} | sps={sps:6.0f} | ETA={eta_h:.1f}h")

            # Stagnation rescue
            if avg_s == 0.0 and avg_p < 0.05 and episode >= 4 * cfg.report_every:
                stagnation_ctr += 1
                if stagnation_ctr >= 4:
                    old = agent.epsilon
                    agent.epsilon = max(agent.epsilon, 0.80)
                    print(f"[L{level}] Stagnation rescue: eps {old:.3f} → {agent.epsilon:.3f}")
                    stagnation_ctr = 0
            else:
                stagnation_ctr = 0

            # Finish-phase exploit
            if avg_c >= target_collect_rate and avg_s < target_success_rate and episode >= 2 * cfg.report_every:
                old = agent.epsilon
                agent.epsilon = max(cfg.epsilon_min, min(agent.epsilon, 0.08))
                if agent.epsilon != old:
                    print(f"[L{level}] Finish-phase exploit: eps {old:.3f} → {agent.epsilon:.3f}")

            if avg_s >= target_success_rate and avg_c >= target_collect_rate:
                early_stop_wins += 1
                if early_stop_wins >= 2:
                    print(f"[L{level}] Early stop: success>={target_success_rate:.0%} for 2 windows.")
                    break
            else:
                early_stop_wins = 0

        if ckpt_path is not None and episode % cfg.save_every == 0:
            n  = min(cfg.report_every, len(success_hist))
            rs = float(np.mean(success_hist[-n:]))
            if rs > best_success:
                best_success = rs
                bp = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(bp, level=level, map_kind=map_kind)
                print(f"  [ckpt] New best: success={rs:.1%} → {bp.name}")

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
    model_path: Path, map_kind: str, map_number: int,
    episodes: int = 5, render: bool = False, render_fps: float = 5.0,
    cfg: Optional[DQNConfig] = None,
) -> None:
    if cfg is None: cfg = DQNConfig()
    level_cfg = _get_level_cfg(map_number)
    max_steps = level_cfg["max_steps"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    silent_rc = RewardConfig(
        step=0.0, carrot=0.0, egg=0.0, finish=0.0, death=0.0,
        invalid_move=0.0, distance_delta_scale=0.0,
        new_best_target_distance_scale=0.0, new_best_finish_distance_scale=0.0,
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
    agent.epsilon = 0.0; agent.policy_net.eval()
    print(f"Loaded {model_path}  (level={meta['level']}, kind={meta['map_kind']})")

    proxy = FastBobbyEnv(map_kind, map_number, max_steps)
    for ep in range(1, episodes + 1):
        full_env.set_map(map_kind=map_kind, map_number=map_number); full_env.reset()
        proxy.map_info          = full_env.map_info
        proxy.bobby             = full_env.bobby
        proxy._expected_carrots = full_env.map_info.carrot_total if full_env.map_info else 0
        proxy._expected_eggs    = full_env.map_info.egg_total    if full_env.map_info else 0
        proxy._visited_tiles    = set()
        grid, inv = _semantic_channels(proxy)
        done = False; steps = 0; total_r = 0.0
        info: Dict[str, object] = {}
        while not done and steps < max_steps:
            action               = agent.select_action(grid, inv)
            _, raw_r, done, info = full_env.step(action)
            proxy.map_info = full_env.map_info; proxy.bobby = full_env.bobby
            grid, inv = _semantic_channels(proxy); total_r += raw_r; steps += 1
            if render:
                full_env.render()
                if render_fps > 0: time.sleep(1.0 / render_fps)
        print(f"Play ep {ep}/{episodes} | collected={bool(info.get('all_collected'))} | "
              f"success={bool(info.get('level_completed'))} | steps={steps}")
    full_env.close()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bobby Carrot DQN")
    p.add_argument("--play",                 action="store_true")
    p.add_argument("--map-kind",             default="normal", choices=["normal","egg"])
    p.add_argument("--map-number",           type=int,   default=1)
    p.add_argument("--levels",               type=int,   nargs="+", default=None)
    p.add_argument("--individual-levels",    action="store_true")
    p.add_argument("--episodes-per-level",   type=int,   default=None)
    p.add_argument("--max-steps",            type=int,   default=None)
    p.add_argument("--batch-size",           type=int,   default=256)
    p.add_argument("--lr",                   type=float, default=3e-4)
    p.add_argument("--gamma",                type=float, default=0.99)
    p.add_argument("--epsilon-start",        type=float, default=1.0)
    p.add_argument("--epsilon-min",          type=float, default=0.03)
    p.add_argument("--epsilon-decay",        type=float, default=0.9997)
    p.add_argument("--completion-bonus",     type=float, default=500.0)
    p.add_argument("--death-penalty",        type=float, default=-150.0)
    p.add_argument("--carrot-bonus",         type=float, default=30.0)
    p.add_argument("--egg-bonus",            type=float, default=30.0)
    p.add_argument("--all-collected-bonus",  type=float, default=100.0)
    p.add_argument("--crumble-area-bonus",   type=float, default=25.0)
    p.add_argument("--reachability-loss-penalty", type=float, default=-80.0)
    p.add_argument("--incomplete-penalty",   type=float, default=-30.0)
    p.add_argument("--invalid-move-penalty", type=float, default=-0.1)
    p.add_argument("--step-penalty",         type=float, default=-0.01)
    p.add_argument("--revisit-penalty",      type=float, default=-0.03)
    p.add_argument("--new-tile-bonus",       type=float, default=0.3)
    p.add_argument("--terminal-oversample",  type=int,   default=20)
    p.add_argument("--all-collected-oversample", type=int, default=5)
    p.add_argument("--warmup-episodes",      type=int,   default=10)
    p.add_argument("--stall-steps",          type=int,   default=None,
                   help="Stall threshold (default: per-level config)")
    p.add_argument("--train-every",          type=int,   default=2)
    p.add_argument("--report-every",         type=int,   default=100)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--model-path",
                   default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",
                   default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes",        type=int,   default=5)
    p.add_argument("--no-render",            action="store_true")
    p.add_argument("--render-fps",           type=float, default=5.0)
    p.add_argument("--target-collect-rate",  type=float, default=0.90)
    p.add_argument("--target-success-rate",  type=float, default=0.90)
    p.add_argument("--resume",               action="store_true")
    p.add_argument("--no-compile",           action="store_true")
    p.add_argument("--no-amp",               action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
        epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        completion_bonus=args.completion_bonus, death_penalty=args.death_penalty,
        carrot_bonus=args.carrot_bonus, egg_bonus=args.egg_bonus,
        all_collected_bonus=args.all_collected_bonus,
        crumble_area_bonus=args.crumble_area_bonus,
        reachability_loss_penalty=args.reachability_loss_penalty,
        incomplete_penalty=args.incomplete_penalty,
        invalid_move_penalty=args.invalid_move_penalty,
        step_penalty=args.step_penalty, revisit_penalty=args.revisit_penalty,
        new_tile_bonus=args.new_tile_bonus,
        terminal_oversample=args.terminal_oversample,
        all_collected_oversample=args.all_collected_oversample,
        warmup_episodes=args.warmup_episodes,
        stall_steps=args.stall_steps if args.stall_steps is not None else 200,
        train_every_steps=args.train_every,
        report_every=args.report_every, seed=args.seed, use_amp=not args.no_amp,
    )


def _apply_cli_overrides(level: int, args: argparse.Namespace) -> None:
    if args.episodes_per_level is not None:
        LEVEL_CONFIG.setdefault(level, {})["episodes"] = args.episodes_per_level
    if args.max_steps is not None:
        LEVEL_CONFIG.setdefault(level, {})["max_steps"] = args.max_steps
    if args.stall_steps is not None:
        LEVEL_CONFIG.setdefault(level, {})["stall_steps"] = args.stall_steps


def _main() -> None:
    args = _build_parser().parse_args()
    cfg  = _cfg_from_args(args)
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
    for lvl in levels: _apply_cli_overrides(lvl, args)

    probe = _make_env(args.map_kind, levels[0], 10); probe.reset()
    n_actions = probe.action_space_n; probe.close()

    if args.individual_levels:
        for lvl in levels:
            print(f"\n=== Independent training — level {lvl} ===")
            agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
            if not args.no_compile:
                agent.policy_net = _maybe_compile(agent.policy_net, device)
                agent.target_net = _maybe_compile(agent.target_net, device)
            out     = ckpt_dir / f"dqn_level{lvl}_individual.pt"
            summary = train_one_level(agent, lvl, cfg, args.map_kind,
                                      args.target_collect_rate, args.target_success_rate, out)
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
        summary = train_one_level(agent, lvl, cfg, args.map_kind,
                                  args.target_collect_rate, args.target_success_rate, out)
        agent.save(out, level=lvl, map_kind=args.map_kind)
        print(f"Saved {out} | "
              f"mean_reward={summary['mean_reward']:.2f} | "
              f"progress={summary['progress_rate']:.2%} | "
              f"all_collected={summary['all_collected_rate']:.2%} | "
              f"success={summary['success_rate']:.2%}")

    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"\nFinal model saved to: {model_path}")


if __name__ == "__main__":
    _main()
