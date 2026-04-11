from __future__ import annotations
import argparse, random, sys, time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Set
import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

_HERE = Path(__file__).resolve()
ROOT  = _HERE.parent
while not (ROOT / "Game_Python").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
GAME_PYTHON_DIR = ROOT / "Game_Python"
if not GAME_PYTHON_DIR.exists():
    raise RuntimeError("Could not locate Game_Python directory.")
if str(GAME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_PYTHON_DIR))

from bobby_carrot.game import Bobby, Map, MapInfo, State
from bobby_carrot.rl_env import BobbyCarrotEnv, RewardConfig

try:
    import torch, torch.nn as nn, torch.optim as optim
except Exception as exc:
    raise RuntimeError("PyTorch required") from exc

# ── constants ────────────────────────────────────────────────────────────────
GRID_SIZE     = 16
GRID_CHANNELS = 15   # ch14 = BFS distance gradient (directional guide for the CNN)
INV_FEATURES  = 7

# Precomputed neighbour table [i] → tuple of valid flat indices
NEIGHBORS: List[Tuple[int, ...]] = []
for _i in range(256):
    _x, _y = _i % 16, _i // 16
    _n: List[int] = []
    if _x > 0:  _n.append(_i - 1)
    if _x < 15: _n.append(_i + 1)
    if _y > 0:  _n.append(_i - 16)
    if _y < 15: _n.append(_i + 16)
    NEIGHBORS.append(tuple(_n))

_TILE_CH = np.zeros(256, dtype=np.int8)
for _i in range(256):
    if   _i < 18:                _TILE_CH[_i] = 0
    elif _i == 18:               _TILE_CH[_i] = 1
    elif _i == 19:               _TILE_CH[_i] = 2
    elif _i == 45:               _TILE_CH[_i] = 3
    elif _i == 44:               _TILE_CH[_i] = 4
    elif _i in (31, 46):         _TILE_CH[_i] = 6
    elif _i in (32, 34, 36):     _TILE_CH[_i] = 7
    elif _i in (33, 35, 37):     _TILE_CH[_i] = 8
    elif _i == 30:               _TILE_CH[_i] = 9
    elif _i in (40,41,42,43):    _TILE_CH[_i] = 10
    elif _i in (22,23,38,39):    _TILE_CH[_i] = 11
    else:                        _TILE_CH[_i] = 12

_RED_MAP  = {22:23,23:22,24:25,25:26,26:27,27:24,28:29,29:28}
_BLUE_MAP = {38:39,39:38,40:41,41:40,42:43,43:42}

# direction delta → (delta, left-bound-check-mask, right-bound-check-mask)
_DELTAS = ((-1, True, False), (1, False, True), (-16, False, False), (16, False, False))

_DIST_BUF = np.full(256, -1, dtype=np.int16)
_CH_BUF   = np.zeros(256, dtype=np.int8)
_GRID_BUF = np.zeros((GRID_CHANNELS, 256), dtype=np.uint8)
_GRAD_BUF = np.zeros(256, dtype=np.float32)  # BFS distance as float for gradient channel

# ── conveyor-aware BFS ───────────────────────────────────────────────────────
# THE key fix: phantom paths through wrong-direction conveyors are blocked.
# Rule: do NOT enter conveyor tile from the direction it would bounce you back.
#   tile 40 (force LEFT ):  block entry from left  (delta=+1 coming from left)
#   tile 41 (force RIGHT):  block entry from right (delta=-1 coming from right)
#   tile 42 (force UP   ):  block entry from above (delta=+16 coming from above)
#   tile 43 (force DOWN ):  block entry from below (delta=-16 coming from below)

def _get_bfs_func():
    def _bfs_impl(passable: np.ndarray, start: int, out: np.ndarray, md: np.ndarray) -> None:
        out[start] = 0
        q = np.empty(512, dtype=np.int32)
        head = 0
        tail = 0
        q[tail] = start
        tail += 1
        
        while head < tail:
            i = q[head]
            head += 1
            d1 = out[i] + 1
            ti = md[i]
            
            for delta in (-1, 1, -16, 16):
                if ti == 40 and delta != -1: continue
                if ti == 41 and delta != 1: continue
                if ti == 42 and delta != -16: continue
                if ti == 43 and delta != 16: continue
                
                ni = i + delta
                if delta == -1 and i % 16 == 0: continue
                if delta == 1 and i % 16 == 15: continue
                if ni < 0 or ni > 255: continue
                if out[ni] != -1: continue
                if not passable[ni]: continue
                
                t_ni = md[ni]
                blocked = False
                if t_ni == 40 and (delta == 1 or delta == 16 or delta == -16): blocked = True
                elif t_ni == 41 and (delta == -1 or delta == 16 or delta == -16): blocked = True
                elif t_ni == 42 and (delta == 16 or delta == 1 or delta == -1): blocked = True
                elif t_ni == 43 and (delta == -16 or delta == 1 or delta == -1): blocked = True
                if blocked: continue
                
                out[ni] = d1
                q[tail] = ni
                tail += 1
    return _bfs_impl

if "HAS_NUMBA" in globals() and HAS_NUMBA:
    _bfs = numba.njit(cache=True)(_get_bfs_func())
else:
    _bfs = _get_bfs_func()

def _make_passable(md: np.ndarray) -> np.ndarray:
    return md >= 18

def _make_passable_nc(md: np.ndarray) -> np.ndarray:
    return (md >= 18) & (md != 30)

def _nearest_target(dist: np.ndarray, target_mask: np.ndarray) -> int:
    hits = dist[target_mask]
    valid = hits[hits >= 0]
    return int(valid.min()) if valid.size else 32

def _crumble_adj_vec(md: np.ndarray) -> np.ndarray:
    crumble  = (md == 30).reshape(16, 16)
    passable = (md >= 18).reshape(16, 16)
    adj = np.zeros((16, 16), dtype=bool)
    adj[:-1, :] |= crumble[1:,  :]
    adj[1:,  :] |= crumble[:-1, :]
    adj[:, :-1] |= crumble[:, 1:]
    adj[:, 1:]  |= crumble[:, :-1]
    adj &= passable
    return adj.ravel().astype(np.uint8)

def _crumble_gain(md: np.ndarray, passable: np.ndarray, start: int) -> int:
    """Gain in reachable targets from stepping the best crumble.
    Baseline: avoid-crumbles BFS (correct for gated levels)."""
    passable_nc = _make_passable_nc(md)
    dist_nc = np.full(256, -1, dtype=np.int16)
    _bfs(passable_nc, start, dist_nc, md)
    target_mask = (md == 19) | (md == 45)
    baseline = int(np.sum((dist_nc >= 0) & target_mask))

    dist_full = np.full(256, -1, dtype=np.int16)
    _bfs(passable, start, dist_full, md)
    crumble_idxs = np.where((md == 30) & (dist_full >= 0))[0]

    best_gain = 0
    for ci in crumble_idxs:
        p2 = passable.copy(); p2[ci] = False
        reachable = np.zeros(256, dtype=bool)
        for delta, lb, rb in _DELTAS:
            if lb and ci % 16 == 0:  continue
            if rb and ci % 16 == 15: continue
            ni = ci + delta
            if ni < 0 or ni > 255:   continue
            if not p2[ni]:            continue
            t_ni = int(md[ni])
            blocked = False
            if t_ni == 40 and (delta == 1 or delta == 16 or delta == -16): blocked = True
            elif t_ni == 41 and (delta == -1 or delta == 16 or delta == -16): blocked = True
            elif t_ni == 42 and (delta == 16 or delta == 1 or delta == -1): blocked = True
            elif t_ni == 43 and (delta == -16 or delta == 1 or delta == -1): blocked = True
            if blocked: continue
            d2 = np.full(256, -1, dtype=np.int16)
            _bfs(p2, ni, d2, md)
            reachable |= (d2 >= 0)
        gain = int(np.sum(reachable & target_mask)) - baseline
        if gain > best_gain:
            best_gain = gain
    return max(0, best_gain)

def _nearest_crumble_dist(dist_full: np.ndarray, md: np.ndarray) -> int:
    hits = dist_full[md == 30]
    valid = hits[hits >= 0]
    return int(valid.min()) if valid.size else 32

def _solve_crumble_order(md: np.ndarray, start_idx: int) -> List[Tuple[int, int]]:
    """Find optimal target collection order for crumble-maze levels.

    Fully dynamic: reads tile types from the map data, nothing is hardcoded.
    Returns ordered waypoint list of (x, y) positions the agent should visit
    through the crumble maze.  Empty list when the level has no crumble
    dependency or when the maze is too complex (fall back to nearest-target).
    Runs once per level at init time.
    """
    import time as _time

    passable = md >= 18
    # Dynamically find ALL carrots, eggs, crumbles, and finish on the map
    targets = frozenset(
        (i % 16, i // 16) for i in range(256) if md[i] in (19, 45)
    )
    crumble_s = frozenset(
        (i % 16, i // 16) for i in range(256) if md[i] == 30
    )
    finish_pos: Optional[Tuple[int, int]] = None
    for i in range(256):
        if md[i] == 44:
            finish_pos = (i % 16, i // 16)
            break

    if not targets or not crumble_s:
        return []

    # Check if all targets reachable without stepping on crumbles
    p_nc = passable & (md != 30)
    d_nc = np.full(256, -1, dtype=np.int16)
    _bfs(p_nc, start_idx, d_nc, md)
    unreachable_nc = [t for t in targets if d_nc[t[0] + t[1] * 16] < 0]
    if not unreachable_nc:
        return []  # All targets reachable without crumbles — no maze

    # Skip levels where DFS would be intractable or unnecessary:
    # - Too many unreachable targets or crumbles -> combinatorial explosion
    # - Very few crumbles (<=2) -> not a real maze, just simple bridges
    # - More unreachable targets than crumbles -> scattered shortcuts, not maze
    if (len(unreachable_nc) > 12 or len(crumble_s) > 14
            or len(crumble_s) <= 2
            or len(unreachable_nc) > len(crumble_s)):
        return []

    # Full-passable BFS to rank bridge crumbles by distance
    d_full = np.full(256, -1, dtype=np.int16)
    _bfs(passable, start_idx, d_full, md)

    # Bridges = reachable crumbles adjacent to a tile reachable without crumbles
    bridges: List[Tuple[int, int, int]] = []
    for cx, cy in crumble_s:
        ci = cx + cy * 16
        if d_full[ci] < 0:
            continue
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ax, ay = cx + ddx, cy + ddy
            if 0 <= ax < 16 and 0 <= ay < 16 and d_nc[ax + ay * 16] >= 0:
                bridges.append((cx, cy, int(d_full[ci])))
                break
    bridges.sort(key=lambda t: t[2])

    if not bridges:
        return []  # No bridge crumbles found

    best: List[Tuple[int, int]] = []
    visited: set = set()
    deadline = _time.monotonic() + 2.0  # 2 sec timeout

    def _dfs(x: int, y: int, rem: frozenset, cr: frozenset,
             path: list, on_cr: bool, depth: int) -> None:
        nonlocal best
        if best or _time.monotonic() > deadline or depth > 60:
            return
        if not rem:
            # All targets collected — verify finish is still reachable
            if finish_pos is not None:
                tp = passable.copy()
                for bx, by in crumble_s - cr:
                    tp[bx + by * 16] = False
                if on_cr:
                    tp[x + y * 16] = False
                d = np.full(256, -1, dtype=np.int16)
                _bfs(tp, x + y * 16, d, md)
                if d[finish_pos[0] + finish_pos[1] * 16] >= 0:
                    best = list(path)
            else:
                best = list(path)
            return
        sk = (x, y, rem, cr)
        if sk in visited:
            return
        visited.add(sk)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue
            ni = nx + ny * 16
            if not passable[ni]:
                continue
            if (nx, ny) in crumble_s and (nx, ny) not in cr:
                continue  # Broken crumble = hole
            nr = rem - {(nx, ny)} if (nx, ny) in rem else rem
            nc = cr - {(x, y)} if on_cr else cr
            noc = (nx, ny) in cr
            path.append((nx, ny))
            _dfs(nx, ny, nr, nc, path, noc, depth + 1)
            if best:
                return
            path.pop()

    for cx, cy, _ in bridges:
        visited.clear()
        _dfs(cx, cy, targets, crumble_s, [(cx, cy)], True, 0)
        if best:
            break

    if best:
        print(f"    [solver] Crumble-maze path: {len(best)} waypoints")
    return best

def _reachability(passable: np.ndarray, start: int, md: np.ndarray) -> Tuple[int, bool, int]:
    dist = np.full(256, -1, dtype=np.int16)
    _bfs(passable, start, dist, md)
    rc = int(np.sum((dist >= 0) & ((md == 19) | (md == 45))))
    fi = np.where((dist >= 0) & (md == 44))[0]
    fr = fi.size > 0
    return rc, fr, (int(dist[fi].min()) if fr else -1)

def _count_reachable_targets(passable: np.ndarray, start: int, md: np.ndarray) -> int:
    dist = np.full(256, -1, dtype=np.int16)
    _bfs(passable, start, dist, md)
    return int(np.sum((dist >= 0) & ((md == 19) | (md == 45))))

# ── per-level config ─────────────────────────────────────────────────────────
LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps":250,  "episodes":2000,  "distance_scale":2.0, "post_penalty":-0.5,
         "crumble_area_bonus":15.0, "stall_steps":150, "new_tile_bonus":0.3},
    2:  {"max_steps":280,  "episodes":2000,  "distance_scale":2.0, "post_penalty":-0.5,
         "crumble_area_bonus":15.0, "stall_steps":150, "new_tile_bonus":0.3},
    3:  {"max_steps":320,  "episodes":2500,  "distance_scale":2.0, "post_penalty":-0.5,
         "crumble_area_bonus":15.0, "stall_steps":180, "new_tile_bonus":0.3},
    4:  {"max_steps":400,  "episodes":3000,  "distance_scale":2.0, "post_penalty":-0.5,
         "crumble_area_bonus":15.0, "stall_steps":200, "new_tile_bonus":0.3},
    5:  {"max_steps":600,  "episodes":15000, "distance_scale":2.0, "post_penalty":-0.1,
         "crumble_area_bonus":25.0, "stall_steps":350, "new_tile_bonus":0.8},
    6:  {"max_steps":550,  "episodes":12000, "distance_scale":2.0, "post_penalty":-0.1,
         "crumble_area_bonus":25.0, "stall_steps":350, "new_tile_bonus":0.8},
    7:  {"max_steps":650,  "episodes":10000, "distance_scale":2.0, "post_penalty":-0.3,
         "crumble_area_bonus":20.0, "stall_steps":300, "new_tile_bonus":0.5},
    # L8: Crumble maze requires entering via BOTTOM bridge (9,11), collecting
    # all 8 carrots in a specific zigzag, then exiting via TOP bridge (9,8).
    # Waypoint solver pre-computes the optimal order; distance shaping guides
    # the agent toward the next waypoint instead of nearest carrot.
    8:  {"max_steps":800,  "episodes":25000, "distance_scale":3.5, "post_penalty":-0.1,
         "crumble_area_bonus":30.0, "stall_steps":600, "new_tile_bonus":1.5},
    9:  {"max_steps":700,  "episodes":10000, "distance_scale":2.5, "post_penalty":-0.1,
         "crumble_area_bonus":20.0, "stall_steps":400, "new_tile_bonus":0.6},
    10: {"max_steps":700,  "episodes":12000, "distance_scale":2.5, "post_penalty":-0.1,
         "crumble_area_bonus":28.0, "stall_steps":400, "new_tile_bonus":0.8},
}

def _get_level_cfg(level: int) -> Dict:
    base = {"max_steps":400,"episodes":4000,"distance_scale":2.0,"post_penalty":-0.5,
            "crumble_area_bonus":15.0,"stall_steps":200,"new_tile_bonus":0.3}
    cfg = dict(base); cfg.update(LEVEL_CONFIG.get(level, {}))
    return cfg

@lru_cache(maxsize=64)
def _load_map_stats(map_kind: str, level: int) -> Dict[str, int]:
    mi  = Map(map_kind, level).load_map_info()
    arr = np.asarray(mi.data, dtype=np.uint8)
    return {"carrots":  int(np.count_nonzero(arr == 19)),
            "eggs":     int(np.count_nonzero(arr == 45)),
            "crumbles": int(np.count_nonzero(arr == 30)),
            "conveyors":int(np.count_nonzero((arr >= 40) & (arr <= 43)))}

# ── DQNConfig ────────────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    gamma:               float = 0.99
    lr:                  float = 5e-4
    batch_size:          int   = 512
    replay_capacity:     int   = 120_000
    min_replay_size:     int   = 1024
    target_update_steps: int   = 400
    train_every_steps:   int   = 4
    epsilon_start:       float = 1.0
    epsilon_min:         float = 0.03
    epsilon_decay:       float = 0.9995
    death_penalty:       float = -150.0
    carrot_bonus:        float = 30.0
    egg_bonus:           float = 30.0
    completion_bonus:    float = 500.0
    all_collected_bonus: float = 100.0
    crumble_area_bonus:  float = 25.0
    safe_crumble_bonus:  float = 5.0
    reachability_loss_penalty: float = -80.0
    finish_loss_penalty: float = -160.0
    incomplete_penalty:  float = -30.0
    invalid_move_penalty:float = -0.1
    step_penalty:        float = -0.01
    revisit_penalty:     float = -0.03
    new_tile_bonus:      float = 0.3
    waypoint_milestone_bonus: float = 5.0
    terminal_oversample:      int = 15
    all_collected_oversample: int = 5
    crumble_gain_oversample:  int = 4
    warmup_episodes:     int   = 5
    stall_steps:         int   = 200
    report_every:        int   = 100
    save_every:          int   = 500
    seed:                int   = 42
    use_amp:             bool  = True

# ── MapStateCache ────────────────────────────────────────────────────────────
class MapStateCache:
    """Recomputed only when map tiles change (crumble stepped / carrot eaten)."""
    def __init__(self) -> None:
        self._state_id = -1
        self.passable:    np.ndarray = np.zeros(256, dtype=bool)
        self.passable_nc: np.ndarray = np.zeros(256, dtype=bool)
        self.crumble_adj: np.ndarray = np.zeros(256, dtype=np.uint8)
        self.crumble_gain: int = 0
        self.dist_full:   np.ndarray = np.full(256, -1, dtype=np.int16)
        self.nearest_crumble: int = 32
        self.target_mask: np.ndarray = np.zeros(256, dtype=bool)
        self.n_targets:   int = 0
        self.finish_reachable: bool = False
        self.finish_dist: int = 32

    def refresh(self, md: np.ndarray, state_id: int, start_idx: int) -> None:
        if state_id == self._state_id:
            return
        self._state_id   = state_id
        self.passable    = _make_passable(md)
        self.passable_nc = _make_passable_nc(md)
        self.crumble_adj = _crumble_adj_vec(md)
        self.target_mask = (md == 19) | (md == 45)
        self.n_targets   = int(self.target_mask.sum())
        if state_id == 0:
            self.crumble_gain = _crumble_gain(md, self.passable, start_idx)
        else:
            self.crumble_gain = 0
        self.dist_full[:] = -1
        _bfs(self.passable, start_idx, self.dist_full, md)
        self.nearest_crumble = _nearest_crumble_dist(self.dist_full, md)
        fi = self.dist_full[md == 44]
        vf = fi[fi >= 0]
        self.finish_reachable = vf.size > 0
        self.finish_dist = int(vf.min()) if vf.size else 32

# ── FastBobbyEnv ─────────────────────────────────────────────────────────────
def _apply_switch(md: List[int], tile: int) -> None:
    lut = _RED_MAP if tile == 22 else _BLUE_MAP
    for i in range(256):
        v = md[i]
        if v in lut: md[i] = lut[v]

class FastBobbyEnv:
    ACTIONS = [State.Left, State.Right, State.Up, State.Down]

    def __init__(self, map_kind: str, map_number: int, max_steps: int) -> None:
        self.map_kind = map_kind; self.map_number = map_number; self.max_steps = max_steps
        self._map_obj = Map(map_kind, map_number); self._fresh: Optional[MapInfo] = None
        self.map_info: Optional[MapInfo] = None; self.bobby: Optional[Bobby] = None
        self.step_count = 0; self.episode_done = False; self.action_space_n = 4
        self._expected_carrots = 0; self._expected_eggs = 0
        self._map_state_id = 0; self._visited_tiles: Set[Tuple[int,int]] = set()
        self._cache = MapStateCache()
        self._md_np: np.ndarray = np.zeros(256, dtype=np.uint8)
        # Waypoint system for crumble-maze levels
        self._waypoints: List[Tuple[int,int]] = []
        self._wp_idx: int = 0
        self._wp_level_key: Tuple[str,int] = ("", 0)

    def set_map(self, map_kind: str, map_number: int) -> None:
        if map_kind != self.map_kind or map_number != self.map_number:
            self.map_kind = map_kind; self.map_number = map_number
            self._map_obj = Map(map_kind, map_number); self._fresh = None

    def reset(self) -> None:
        if self._fresh is None: self._fresh = self._map_obj.load_map_info()
        mi = self._fresh
        self.map_info = MapInfo(data=mi.data.copy(), coord_start=mi.coord_start,
                                carrot_total=mi.carrot_total, egg_total=mi.egg_total)
        self._expected_carrots = mi.carrot_total; self._expected_eggs = mi.egg_total
        self.bobby = Bobby(start_frame=0, start_time=0, coord_src=mi.coord_start)
        self.bobby.state = State.Down; self.bobby.coord_dest = self.bobby.coord_src
        self.step_count = 0; self.episode_done = False; self._map_state_id = 0
        self._visited_tiles = {mi.coord_start}
        np.copyto(self._md_np, np.array(mi.data, dtype=np.uint8))
        si = mi.coord_start[0] + mi.coord_start[1] * GRID_SIZE
        self._cache.refresh(self._md_np, self._map_state_id, si)
        # Solve crumble maze once per level (reuse across episodes)
        lk = (self.map_kind, self.map_number)
        if lk != self._wp_level_key:
            self._waypoints = _solve_crumble_order(self._md_np, si)
            self._wp_level_key = lk
        self._wp_idx = 0

    def step(self, action: int) -> Tuple[float, bool, Dict]:
        assert self.bobby is not None and self.map_info is not None
        b = self.bobby; md = self.map_info.data
        before_carrot = b.carrot_count; before_egg = b.egg_count

        if b.next_state in (State.Left, State.Right, State.Up, State.Down):
            desired = b.next_state; b.next_state = None
        else:
            desired = self.ACTIONS[action]
        b.state = desired; b.coord_dest = b.coord_src; b.update_dest(md)

        invalid_move = b.coord_dest == b.coord_src
        moved        = b.coord_src  != b.coord_dest
        map_mutated  = False; ot = 0
        destroyed_crumble = False; reachable_before = 0; finish_reachable_before = False

        if moved:
            old_pos = b.coord_src[0]  + b.coord_src[1]  * GRID_SIZE
            new_pos = b.coord_dest[0] + b.coord_dest[1] * GRID_SIZE
            ot = md[old_pos]
            if ot == 30:
                destroyed_crumble = True
                reachable_before  = self._cache.n_targets
                finish_reachable_before = self._cache.finish_reachable

            if   ot == 24: md[old_pos]=25; self._md_np[old_pos]=25; map_mutated=True
            elif ot == 25: md[old_pos]=26; self._md_np[old_pos]=26; map_mutated=True
            elif ot == 26: md[old_pos]=27; self._md_np[old_pos]=27; map_mutated=True
            elif ot == 27: md[old_pos]=24; self._md_np[old_pos]=24; map_mutated=True
            elif ot == 28: md[old_pos]=29; self._md_np[old_pos]=29; map_mutated=True
            elif ot == 29: md[old_pos]=28; self._md_np[old_pos]=28; map_mutated=True
            elif ot == 30: md[old_pos]=31; self._md_np[old_pos]=31; map_mutated=True
            elif ot == 45: md[old_pos]=46; self._md_np[old_pos]=46; b.egg_count+=1; map_mutated=True

            nt = md[new_pos]
            if   nt == 19: md[new_pos]=20; self._md_np[new_pos]=20; b.carrot_count+=1; map_mutated=True
            elif nt == 22: _apply_switch(md,22); np.copyto(self._md_np, np.array(md, dtype=np.uint8)); map_mutated=True
            elif nt == 32: md[new_pos]=18; self._md_np[new_pos]=18; b.key_gray+=1; map_mutated=True
            elif nt == 33 and b.key_gray>0: md[new_pos]=18; self._md_np[new_pos]=18; b.key_gray-=1; map_mutated=True
            elif nt == 34: md[new_pos]=18; self._md_np[new_pos]=18; b.key_yellow+=1; map_mutated=True
            elif nt == 35 and b.key_yellow>0: md[new_pos]=18; self._md_np[new_pos]=18; b.key_yellow-=1; map_mutated=True
            elif nt == 36: md[new_pos]=18; self._md_np[new_pos]=18; b.key_red+=1; map_mutated=True
            elif nt == 37 and b.key_red>0: md[new_pos]=18; self._md_np[new_pos]=18; b.key_red-=1; map_mutated=True
            elif nt == 38: _apply_switch(md,38); np.copyto(self._md_np, np.array(md, dtype=np.uint8)); map_mutated=True
            elif nt == 40: b.next_state=State.Left
            elif nt == 41: b.next_state=State.Right
            elif nt == 42: b.next_state=State.Up
            elif nt == 43: b.next_state=State.Down
            elif nt == 31: b.dead=True
            b.coord_src = b.coord_dest

        if map_mutated:
            self._map_state_id += 1
            si = b.coord_src[0] + b.coord_src[1] * GRID_SIZE
            self._cache.refresh(self._md_np, self._map_state_id, si)

        cur_pos  = b.coord_src[0] + b.coord_src[1] * GRID_SIZE
        cur_tile = md[cur_pos]
        if cur_tile == 31: b.dead = True

        is_new_tile = b.coord_src not in self._visited_tiles
        if moved: self._visited_tiles.add(b.coord_src)

        # Track waypoint progression — robust: skip past already-visited or
        # already-collected/broken waypoints so agent recovers from deviations
        waypoint_hit = False
        if self._waypoints and self._wp_idx < len(self._waypoints):
            while self._wp_idx < len(self._waypoints):
                wp = self._waypoints[self._wp_idx]
                wp_i = wp[0] + wp[1] * GRID_SIZE
                wp_tile = md[wp_i]
                at_wp = (b.coord_src == wp)
                # Skip waypoints that are gone: collected carrot (20), collected
                # egg (46), broken crumble (31) — the map already changed.
                wp_gone = wp_tile in (20, 31, 46)
                if at_wp or wp_gone:
                    self._wp_idx += 1
                    if at_wp: waypoint_hit = True
                else:
                    break

        all_collected = self._is_finished()
        on_finish     = (cur_tile == 44) and all_collected and not b.dead
        carrot_delta  = b.carrot_count - before_carrot
        egg_delta     = b.egg_count    - before_egg

        newly_reachable = 0; finish_newly_lost = False; carrots_lost = 0
        if destroyed_crumble and not b.dead:
            ra = self._cache.n_targets; fra = self._cache.finish_reachable
            if ra > reachable_before: newly_reachable = ra - reachable_before
            if ra < reachable_before: carrots_lost = reachable_before - ra
            if finish_reachable_before and not fra: finish_newly_lost = True

        self.step_count += 1
        done = b.dead or on_finish or (self.step_count >= self.max_steps)
        self.episode_done = done
        return 0.0, done, {
            "collected_carrot":  carrot_delta, "collected_egg": egg_delta,
            "all_collected":     all_collected, "invalid_move": invalid_move,
            "dead":              b.dead, "level_completed": on_finish,
            "position":          b.coord_src, "moved": moved,
            "carrots_lost":      carrots_lost, "finish_lost": finish_newly_lost,
            "destroyed_crumble": destroyed_crumble, "newly_reachable": newly_reachable,
            "is_new_tile":       is_new_tile, "waypoint_hit": waypoint_hit,
        }

    def _is_finished(self) -> bool:
        b = self.bobby; assert b is not None
        if self._expected_carrots > 0: return b.carrot_count >= self._expected_carrots
        if self._expected_eggs    > 0: return b.egg_count    >= self._expected_eggs
        return True

    def close(self) -> None: pass

# ── observation ──────────────────────────────────────────────────────────────
def _semantic_channels(env: FastBobbyEnv) -> Tuple[np.ndarray, np.ndarray]:
    b = env.bobby; mi = env.map_info
    assert b is not None and mi is not None
    md_np = env._md_np; cache = env._cache
    all_collected = env._is_finished()

    np.copyto(_CH_BUF, _TILE_CH[md_np])
    _CH_BUF[md_np == 44] = 5 if all_collected else 4
    _GRID_BUF.fill(0)
    for ch in range(13): _GRID_BUF[ch][_CH_BUF == ch] = 1
    _GRID_BUF[13] = cache.crumble_adj

    grid = _GRID_BUF.reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).copy()
    px, py = b.coord_src
    grid[1, py, px] = 2
    if all_collected: grid[5, :, :] = 1

    remaining = max(0, (mi.carrot_total - b.carrot_count) + (mi.egg_total - b.egg_count))
    rem_norm  = float(remaining) / float(max(1, mi.carrot_total + mi.egg_total))

    # ── Conveyor-aware BFS distance (THE key fix for L8) ─────────────────
    agent_idx = px + py * GRID_SIZE
    _DIST_BUF[:] = -1
    _bfs(cache.passable, agent_idx, _DIST_BUF, md_np)   # ← now conveyor-aware
    if all_collected:
        fh = _DIST_BUF[md_np == 44]; vf = fh[fh >= 0]
        bfs_dist = int(vf.min()) if vf.size else 32
    elif env._waypoints and env._wp_idx < len(env._waypoints):
        # Waypoint-guided: scan forward to find next REACHABLE waypoint.
        # If the agent deviated and broke a planned crumble, the immediate
        # next waypoint may be unreachable — skip to the first one we can
        # still reach so the distance gradient keeps pointing somewhere useful.
        bfs_dist = -1
        for _wi in range(env._wp_idx, len(env._waypoints)):
            _wx, _wy = env._waypoints[_wi]
            _wd = int(_DIST_BUF[_wx + _wy * GRID_SIZE])
            if _wd >= 0:
                bfs_dist = _wd
                break
        if bfs_dist < 0:  # No waypoint reachable — fall back to nearest target
            bfs_dist = _nearest_target(_DIST_BUF, cache.target_mask)
    else:
        bfs_dist = _nearest_target(_DIST_BUF, cache.target_mask)
    dist_norm = min(1.0, bfs_dist / 32.0)

    # Crumble gain / nearest crumble from cache
    crumble_g_norm    = min(1.0, cache.crumble_gain / max(1, remaining + 1))
    crumble_dist_norm = min(1.0, cache.nearest_crumble / 32.0)

    inv = np.array([
        float(b.key_gray   > 0), float(b.key_yellow > 0), float(b.key_red > 0),
        rem_norm, dist_norm, crumble_g_norm, crumble_dist_norm,
    ], dtype=np.float32)

    # ── Channel 14: BFS distance gradient (directional navigation map) ───
    # Encodes BFS distance from EVERY tile to the current target as a spatial
    # heatmap.  The CNN can see the gradient and learn which direction to move
    # instead of relying on a single scalar distance value.
    # Values: 0 = unreachable/wall, 1..8 quantised distance steps (closer=higher)
    reachable = _DIST_BUF >= 0
    max_d = max(1, int(_DIST_BUF[reachable].max())) if reachable.any() else 1
    # Vectorised: closer tiles → higher values (8=on target, 1=far), 0=wall
    _GRAD_BUF[:] = np.where(reachable,
                             np.clip(8 - (8 * _DIST_BUF) // max_d, 1, 8), 0)
    grid[14] = _GRAD_BUF.reshape(GRID_SIZE, GRID_SIZE).astype(np.uint8)

    return grid, inv

# ── reward shaping ───────────────────────────────────────────────────────────
def _shape_reward(info: Dict, cfg: DQNConfig, lcfg: Dict,
                  prev_inv: np.ndarray, curr_inv: np.ndarray,
                  env: FastBobbyEnv, vc: np.ndarray) -> float:
    r = cfg.step_penalty
    if info["invalid_move"]: r += cfg.invalid_move_penalty
    if info["dead"]:         r += cfg.death_penalty

    cd = int(info["collected_carrot"]); ed = int(info["collected_egg"])
    b = env.bobby; mi = env.map_info; pf = 0.0
    if b and mi:
        tot = mi.carrot_total + mi.egg_total
        pf  = float(b.carrot_count + b.egg_count) / float(max(1, tot))
    if cd > 0: r += cfg.carrot_bonus * cd + 5.0 * pf * cd
    if ed > 0: r += cfg.egg_bonus    * ed + 5.0 * pf * ed
    if info["level_completed"]: r += cfg.completion_bonus

    all_col = bool(info["all_collected"]); lvl_done = bool(info["level_completed"])
    if all_col and (cd > 0 or ed > 0):
        if b and mi and b.carrot_count + b.egg_count >= mi.carrot_total + mi.egg_total:
            r += cfg.all_collected_bonus

    # Distance shaping using conveyor-aware BFS distance
    r += lcfg["distance_scale"] * (float(prev_inv[4]) - float(curr_inv[4]))

    nr = int(info.get("newly_reachable", 0))
    if nr > 0:
        r += lcfg.get("crumble_area_bonus", cfg.crumble_area_bonus) * nr
        r += cfg.safe_crumble_bonus

    cl = int(info.get("carrots_lost", 0))
    if cl > 0: r += cfg.reachability_loss_penalty * cl
    if info.get("finish_lost"): r += cfg.finish_loss_penalty

    if all_col and not lvl_done: r += lcfg["post_penalty"]

    if b:
        vx, vy = b.coord_src
        visits = int(vc[vx + vy * GRID_SIZE])
        if visits > 2:
            scale = max(0.05, 1.0 - 1.2 * pf)
            r += cfg.revisit_penalty * scale * min(visits - 2, 5)

    if bool(info.get("is_new_tile")):
        r += lcfg.get("new_tile_bonus", cfg.new_tile_bonus)
    if bool(info.get("waypoint_hit")):
        r += cfg.waypoint_milestone_bonus
    return r

# ── ReplayBuffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap: int) -> None:
        self._cap=cap; self._ptr=0; self._size=0
        
        def _alloc(shape, dtype):
            t = torch.zeros(shape, dtype=dtype)
            return t.pin_memory() if torch.cuda.is_available() else t
            
        self._sg=_alloc((cap,GRID_CHANNELS,GRID_SIZE,GRID_SIZE), torch.uint8)
        self._sv=_alloc((cap,INV_FEATURES), torch.float32)
        self._a =_alloc(cap, torch.int64)
        self._r =_alloc(cap, torch.float32)
        self._ng=_alloc((cap,GRID_CHANNELS,GRID_SIZE,GRID_SIZE), torch.uint8)
        self._nv=_alloc((cap,INV_FEATURES), torch.float32)
        self._d =_alloc(cap, torch.float32)
        
    def __len__(self) -> int: return self._size

    def push(self, sg, sv, a, r, ng, nv, d) -> None:
        i=self._ptr
        self._sg[i]=torch.from_numpy(sg)
        self._sv[i]=torch.from_numpy(sv)
        self._a[i]=int(a)
        self._r[i]=float(r)
        self._ng[i]=torch.from_numpy(ng)
        self._nv[i]=torch.from_numpy(nv)
        self._d[i]=float(d)
        self._ptr=(i+1)%self._cap; self._size=min(self._size+1,self._cap)

    def sample(self, n: int, rf: float=0.35) -> tuple:
        nr=int(n*rf); nu=n-nr
        iu=torch.randint(0,self._size,(nu,))
        win=max(n,self._size//4); st=(self._ptr-win)%self._cap
        if self._size>=win:
            if st+win<=self._cap:
                pool=torch.arange(st,st+win)
            else:
                pool=torch.cat([torch.arange(st,self._cap),torch.arange(0,(st+win)%self._cap)])
            ir=pool[torch.randint(0,len(pool),(nr,))]
        else:
            ir=torch.randint(0,self._size,(nr,))
        idx=torch.cat([iu,ir]); idx=idx[torch.randperm(len(idx))]
        return (self._sg[idx],self._sv[idx],self._a[idx],
                self._r[idx],self._ng[idx],self._nv[idx],self._d[idx])

# ── DuelingDQN ───────────────────────────────────────────────────────────────
class DuelingDQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS,32,3,padding=1), nn.GroupNorm(8,32), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(inplace=True), nn.Flatten())
        m = 64*GRID_SIZE*GRID_SIZE+INV_FEATURES
        self.shared     = nn.Sequential(nn.Linear(m,512),nn.ReLU(inplace=True))
        self.value_head = nn.Sequential(nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,1))
        self.adv_head   = nn.Sequential(nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,n_actions))

    def forward(self, g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        s=self.shared(torch.cat([self.conv(g),v],1))
        a=self.adv_head(s); val=self.value_head(s)
        return val+a-a.mean(1,keepdim=True)

# ── DQNAgent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig, device: torch.device) -> None:
        self.n_actions=n_actions; self.cfg=cfg; self.device=device
        self.epsilon=cfg.epsilon_start; self.total_steps=0; self.current_episode=0
        self.policy_net=DuelingDQNCNN(n_actions).to(device).to(memory_format=torch.channels_last)
        self.target_net=DuelingDQNCNN(n_actions).to(device).to(memory_format=torch.channels_last)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=cfg.lr,eps=1.5e-4)
        self.loss_fn=nn.SmoothL1Loss(); self.replay=ReplayBuffer(cfg.replay_capacity)
        self.use_amp=cfg.use_amp and device.type=="cuda"
        self.scaler=torch.amp.GradScaler("cuda") if self.use_amp else None
        self._grid_buf=torch.zeros(1,GRID_CHANNELS,GRID_SIZE,GRID_SIZE,dtype=torch.float32,device=device).to(memory_format=torch.channels_last)
        self._inv_buf=torch.zeros(1,INV_FEATURES,dtype=torch.float32,device=device)

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random()<self.epsilon: return random.randint(0,self.n_actions-1)
        with torch.no_grad():
            self._grid_buf.copy_(torch.from_numpy(grid).unsqueeze(0),non_blocking=True)
            self._inv_buf.copy_(torch.from_numpy(inv).unsqueeze(0),non_blocking=True)
            return int(self.policy_net(self._grid_buf,self._inv_buf).argmax(1).item())

    def optimize_step(self) -> Optional[float]:
        if len(self.replay)<self.cfg.min_replay_size: return None
        if self.total_steps%self.cfg.train_every_steps!=0: return None
        sg,sv,a,r,ng,nv,d=self.replay.sample(self.cfg.batch_size)
        sg_t=sg.to(self.device,dtype=torch.float32,non_blocking=True).to(memory_format=torch.channels_last)
        sv_t=sv.to(self.device,non_blocking=True)
        a_t =a.unsqueeze(1).to(self.device,non_blocking=True)
        r_t =r.to(self.device,non_blocking=True)
        ng_t=ng.to(self.device,dtype=torch.float32,non_blocking=True).to(memory_format=torch.channels_last)
        nv_t=nv.to(self.device,non_blocking=True)
        d_t =d.to(self.device,non_blocking=True)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                qp=self.policy_net(sg_t,sv_t).gather(1,a_t).squeeze(1)
                with torch.no_grad():
                    ba=self.policy_net(ng_t,nv_t).argmax(1,keepdim=True)
                    qn=self.target_net(ng_t,nv_t).gather(1,ba).squeeze(1)
                    tgt=r_t+(1.0-d_t)*self.cfg.gamma*qn
                loss=self.loss_fn(qp,tgt)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),10.0)
            self.scaler.step(self.optimizer); self.scaler.update()
        else:
            qp=self.policy_net(sg_t,sv_t).gather(1,a_t).squeeze(1)
            with torch.no_grad():
                ba=self.policy_net(ng_t,nv_t).argmax(1,keepdim=True)
                qn=self.target_net(ng_t,nv_t).gather(1,ba).squeeze(1)
                tgt=r_t+(1.0-d_t)*self.cfg.gamma*qn
            loss=self.loss_fn(qp,tgt)
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),10.0); self.optimizer.step()
        if self.total_steps%self.cfg.target_update_steps==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon=max(self.cfg.epsilon_min,self.epsilon*self.cfg.epsilon_decay)

    def save(self, path: Path, level: int, map_kind: str) -> None:
        path.parent.mkdir(parents=True,exist_ok=True)
        def _sd(m): return getattr(m,"_orig_mod",m).state_dict()
        torch.save({"policy":_sd(self.policy_net),"target":_sd(self.target_net),
                    "optim":self.optimizer.state_dict(),"epsilon":self.epsilon,
                    "total_steps":self.total_steps,"current_episode":self.current_episode,
                    "level":level,"map_kind":map_kind,
                    "grid_channels":GRID_CHANNELS,"inv_features":INV_FEATURES},path)

    def load(self, path: Path) -> Dict:
        ck=torch.load(path,map_location=self.device,weights_only=False)
        def _b(m): return getattr(m,"_orig_mod",m)
        _b(self.policy_net).load_state_dict(ck["policy"])
        _b(self.target_net).load_state_dict(ck.get("target",ck["policy"]))
        if "optim" in ck: self.optimizer.load_state_dict(ck["optim"])
        self.epsilon=float(ck.get("epsilon",self.cfg.epsilon_start))
        self.total_steps=int(ck.get("total_steps",0))
        self.current_episode=int(ck.get("current_episode",0))
        return {"level":ck.get("level",-1),"map_kind":ck.get("map_kind","normal")}

# ── helpers ───────────────────────────────────────────────────────────────────
def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True

def _make_env(mk,lvl,ms): return FastBobbyEnv(map_kind=mk,map_number=lvl,max_steps=ms)

def _maybe_compile(m,device):
    if not hasattr(torch,"compile") or device.type!="cuda": return m
    try: return torch.compile(m,mode="default")
    except Exception as e: print(f"[warn] compile: {e}"); return m

# ── training loop ────────────────────────────────────────────────────────────
def train_one_level(agent,level,cfg,map_kind,
                    target_collect_rate=0.90,target_success_rate=0.90,ckpt_path=None):
    lcfg=_get_level_cfg(level); max_steps=lcfg["max_steps"]; n_ep=lcfg["episodes"]
    stall_lim=lcfg.get("stall_steps",cfg.stall_steps); stats=_load_map_stats(map_kind,level)
    env=_make_env(map_kind,level,max_steps)
    rh=[]; sh=[]; ch=[]; ph=[]; loss_win=deque(maxlen=cfg.report_every)
    best_succ=-1.0; ew=0; stag=0

    print(f"  max_steps={max_steps} ep={n_ep} dist={lcfg['distance_scale']} "
          f"post={lcfg['post_penalty']} crumble_bonus={lcfg.get('crumble_area_bonus',cfg.crumble_area_bonus):.0f} "
          f"stall={stall_lim} | carrots={stats['carrots']} crumbles={stats['crumbles']} "
          f"conveyors={stats['conveyors']} | batch={cfg.batch_size} lr={cfg.lr} amp={cfg.use_amp}")

    t0=time.time(); env_steps=0
    s_ep=agent.current_episode+1; e_ep=s_ep+n_ep

    for episode in range(s_ep,e_ep):
        agent.current_episode=episode
        env.set_map(map_kind,level); env.reset()
        grid,inv=_semantic_channels(env)
        done=False; steps=0; ep_r=0.0; info={}; prev_inv=inv.copy()
        vc=np.zeros(GRID_SIZE*GRID_SIZE,dtype=np.int16)
        if env.bobby: sx,sy=env.bobby.coord_src; vc[sx+sy*GRID_SIZE]=1
        last_col=0; last_vis=len(env._visited_tiles); stall=0; last_wp=0

        while not done and steps<max_steps:
            if episode<=cfg.warmup_episodes and agent.total_steps<500:
                action=random.randint(0,agent.n_actions-1)
            else:
                action=agent.select_action(grid,inv)

            _,done,info=env.step(action)
            next_grid,next_inv=_semantic_channels(env)
            if env.bobby: px,py=env.bobby.coord_src; vc[px+py*GRID_SIZE]+=1

            shaped=_shape_reward(info,cfg,lcfg,prev_inv,next_inv,env,vc)

            now_col=0; now_vis=len(env._visited_tiles); now_wp=env._wp_idx
            if env.bobby: now_col=env.bobby.carrot_count+env.bobby.egg_count
            if now_col>last_col or now_vis>last_vis or now_wp>last_wp:
                last_col=now_col; last_vis=now_vis; last_wp=now_wp; stall=0
            else: stall+=1
            if not done and stall>=stall_lim: done=True; info["stall_terminated"]=True

            if done and not info.get("level_completed"):
                if env.bobby and env.map_info:
                    tot=env.map_info.carrot_total+env.map_info.egg_total
                    dn=env.bobby.carrot_count+env.bobby.egg_count
                    rf=1.0-float(dn)/float(max(1,tot))
                    if rf>0: shaped+=cfg.incomplete_penalty*rf

            agent.replay.push(grid,inv,action,shaped,next_grid,next_inv,done)
            nr=int(info.get("newly_reachable",0))
            nos=(cfg.terminal_oversample if info.get("level_completed") else
                 cfg.all_collected_oversample if (info.get("all_collected") and not info.get("level_completed")) else
                 cfg.crumble_gain_oversample if nr>0 else 0)
            for _ in range(nos): agent.replay.push(grid,inv,action,shaped,next_grid,next_inv,done)

            agent.total_steps+=1; env_steps+=1
            loss=agent.optimize_step()
            if loss is not None: loss_win.append(loss)
            prev_inv=next_inv; grid,inv=next_grid,next_inv
            ep_r+=shaped; steps+=1

        agent.decay_epsilon()
        succ=1.0 if info.get("level_completed") else 0.0
        coll=1.0 if info.get("all_collected")   else 0.0
        tc=(env.map_info.carrot_total+env.map_info.egg_total) if env.map_info else 1
        dc=(env.bobby.carrot_count+env.bobby.egg_count)       if env.bobby   else 0
        prog=float(dc)/float(max(1,tc))
        rh.append(ep_r); sh.append(succ); ch.append(coll); ph.append(prog)

        if episode%cfg.report_every==0 or episode==s_ep:
            n=cfg.report_every
            ar=float(np.mean(rh[-n:])); as_=float(np.mean(sh[-n:]))
            ac=float(np.mean(ch[-n:])); ap=float(np.mean(ph[-n:]))
            al=float(np.mean(loss_win)) if loss_win else 0.0
            sps=env_steps/max(time.time()-t0,1e-6)
            eta=((e_ep-episode)*max_steps)/max(sps,1)/3600
            print(f"[L{level}] ep={episode:5d} | reward={ar:8.1f} | "
                  f"progress={ap:5.1%} | collected={ac:5.1%} | success={as_:5.1%} | "
                  f"eps={agent.epsilon:.3f} | loss={al:.4f} | sps={sps:6.0f} | ETA={eta:.1f}h")

            if as_==0.0 and ap<0.05 and episode>=s_ep+4*cfg.report_every:
                stag+=1
                if stag>=3:
                    old=agent.epsilon; agent.epsilon=max(agent.epsilon,0.80)
                    print(f"[L{level}] Stagnation rescue: eps {old:.3f}→{agent.epsilon:.3f}"); stag=0
            else: stag=0

            if ac>=target_collect_rate and as_<target_success_rate:
                old=agent.epsilon; agent.epsilon=max(cfg.epsilon_min,min(agent.epsilon,0.06))
                if agent.epsilon!=old: print(f"[L{level}] Finish-phase: eps {old:.3f}→{agent.epsilon:.3f}")

            if as_>=target_success_rate and ac>=target_collect_rate:
                ew+=1
                if ew>=2: print(f"[L{level}] Early stop."); break
            else: ew=0

        if ckpt_path is not None and episode%cfg.save_every==0:
            n=min(cfg.report_every,len(sh)); rs=float(np.mean(sh[-n:]))
            if rs>best_succ:
                best_succ=rs
                bp=ckpt_path.parent/f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(bp,level,map_kind)
                print(f"  [ckpt] New best: success={rs:.1%}→{bp.name}")

    env.close()
    return {"mean_reward":float(np.mean(rh)) if rh else 0.0,
            "success_rate":float(np.mean(sh)) if sh else 0.0,
            "all_collected_rate":float(np.mean(ch)) if ch else 0.0,
            "progress_rate":float(np.mean(ph)) if ph else 0.0}

# ── play ─────────────────────────────────────────────────────────────────────
def play_trained_dqn(model_path,map_kind,map_number,episodes=5,render=False,render_fps=5.0,cfg=None):
    if cfg is None: cfg=DQNConfig()
    lcfg=_get_level_cfg(map_number); max_steps=lcfg["max_steps"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    silent_rc=RewardConfig(step=0.0,carrot=0.0,egg=0.0,finish=0.0,death=0.0,invalid_move=0.0,
        distance_delta_scale=0.0,new_best_target_distance_scale=0.0,
        new_best_finish_distance_scale=0.0,post_collection_step_penalty=0.0,
        no_progress_penalty_after=999999,no_progress_penalty=0.0,
        no_progress_penalty_hard_after=999999,no_progress_penalty_hard=0.0,all_collected_bonus=0.0)
    full_env=BobbyCarrotEnv(map_kind=map_kind,map_number=map_number,observation_mode="full",
        local_view_size=3,include_inventory=True,headless=not render,max_steps=max_steps,reward_config=silent_rc)
    agent=DQNAgent(full_env.action_space_n,cfg,device)
    meta=agent.load(model_path); agent.epsilon=0.0; agent.policy_net.eval()
    print(f"Loaded {model_path} (level={meta['level']},kind={meta['map_kind']})")
    proxy=FastBobbyEnv(map_kind,map_number,max_steps)
    for ep in range(1,episodes+1):
        full_env.set_map(map_kind=map_kind,map_number=map_number); full_env.reset()
        proxy.map_info=full_env.map_info; proxy.bobby=full_env.bobby
        proxy._expected_carrots=full_env.map_info.carrot_total if full_env.map_info else 0
        proxy._expected_eggs=full_env.map_info.egg_total if full_env.map_info else 0
        proxy._visited_tiles=set()
        if full_env.map_info: np.copyto(proxy._md_np,np.array(full_env.map_info.data,dtype=np.uint8))
        if proxy.bobby:
            si=proxy.bobby.coord_src[0]+proxy.bobby.coord_src[1]*GRID_SIZE
            proxy._cache.refresh(proxy._md_np,0,si)
        grid,inv=_semantic_channels(proxy); done=False; steps=0; info={}
        while not done and steps<max_steps:
            action=agent.select_action(grid,inv)
            _,_,done,info=full_env.step(action)
            proxy.map_info=full_env.map_info; proxy.bobby=full_env.bobby
            if full_env.map_info: np.copyto(proxy._md_np,np.array(full_env.map_info.data,dtype=np.uint8))
            if proxy.bobby:
                si=proxy.bobby.coord_src[0]+proxy.bobby.coord_src[1]*GRID_SIZE
                proxy._cache.refresh(proxy._md_np,steps,si)
            grid,inv=_semantic_channels(proxy); steps+=1
            if render:
                full_env.render()
                if render_fps>0: time.sleep(1.0/render_fps)
        print(f"Play ep {ep}/{episodes} | collected={bool(info.get('all_collected'))} | "
              f"success={bool(info.get('level_completed'))} | steps={steps}")
    full_env.close()

# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser():
    p=argparse.ArgumentParser(description="Bobby Carrot DQN")
    p.add_argument("--play",action="store_true")
    p.add_argument("--map-kind",default="normal",choices=["normal","egg"])
    p.add_argument("--map-number",type=int,default=1)
    p.add_argument("--levels",type=int,nargs="+",default=None)
    p.add_argument("--individual-levels",action="store_true")
    p.add_argument("--episodes-per-level",type=int,default=None)
    p.add_argument("--max-steps",type=int,default=None)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--gamma",type=float,default=0.99)
    p.add_argument("--epsilon-start",type=float,default=1.0)
    p.add_argument("--epsilon-min",type=float,default=0.03)
    p.add_argument("--epsilon-decay",type=float,default=0.9995)
    p.add_argument("--completion-bonus",type=float,default=500.0)
    p.add_argument("--death-penalty",type=float,default=-150.0)
    p.add_argument("--carrot-bonus",type=float,default=30.0)
    p.add_argument("--egg-bonus",type=float,default=30.0)
    p.add_argument("--all-collected-bonus",type=float,default=100.0)
    p.add_argument("--crumble-area-bonus",type=float,default=25.0)
    p.add_argument("--reachability-loss-penalty",type=float,default=-80.0)
    p.add_argument("--incomplete-penalty",type=float,default=-30.0)
    p.add_argument("--invalid-move-penalty",type=float,default=-0.1)
    p.add_argument("--step-penalty",type=float,default=-0.01)
    p.add_argument("--revisit-penalty",type=float,default=-0.03)
    p.add_argument("--new-tile-bonus",type=float,default=0.3)
    p.add_argument("--terminal-oversample",type=int,default=15)
    p.add_argument("--all-collected-oversample",type=int,default=5)
    p.add_argument("--warmup-episodes",type=int,default=5)
    p.add_argument("--stall-steps",type=int,default=None)
    p.add_argument("--train-every",type=int,default=2)
    p.add_argument("--report-every",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-path",default=str(Path(__file__).resolve().parent/"dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",default=str(Path(__file__).resolve().parent/"dqn_checkpoints"))
    p.add_argument("--play-episodes",type=int,default=5)
    p.add_argument("--no-render",action="store_true")
    p.add_argument("--render-fps",type=float,default=5.0)
    p.add_argument("--target-collect-rate",type=float,default=0.90)
    p.add_argument("--target-success-rate",type=float,default=0.90)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--no-compile",action="store_true")
    p.add_argument("--no-amp",action="store_true")
    return p

def _cfg_from_args(args):
    return DQNConfig(gamma=args.gamma,lr=args.lr,batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,completion_bonus=args.completion_bonus,
        death_penalty=args.death_penalty,carrot_bonus=args.carrot_bonus,
        egg_bonus=args.egg_bonus,all_collected_bonus=args.all_collected_bonus,
        crumble_area_bonus=args.crumble_area_bonus,
        reachability_loss_penalty=args.reachability_loss_penalty,
        incomplete_penalty=args.incomplete_penalty,
        invalid_move_penalty=args.invalid_move_penalty,
        step_penalty=args.step_penalty,revisit_penalty=args.revisit_penalty,
        new_tile_bonus=args.new_tile_bonus,terminal_oversample=args.terminal_oversample,
        all_collected_oversample=args.all_collected_oversample,
        warmup_episodes=args.warmup_episodes,
        stall_steps=args.stall_steps if args.stall_steps else 200,
        train_every_steps=args.train_every,report_every=args.report_every,
        seed=args.seed,use_amp=not args.no_amp)

def _apply_cli_overrides(level,args):
    if args.episodes_per_level: LEVEL_CONFIG.setdefault(level,{})["episodes"]=args.episodes_per_level
    if args.max_steps:          LEVEL_CONFIG.setdefault(level,{})["max_steps"]=args.max_steps
    if args.stall_steps:        LEVEL_CONFIG.setdefault(level,{})["stall_steps"]=args.stall_steps

def _main():
    args=_build_parser().parse_args(); cfg=_cfg_from_args(args)
    _seed_everything(cfg.seed)
    model_path=Path(args.model_path); ckpt_dir=Path(args.checkpoint_dir)

    if args.play:
        play_trained_dqn(model_path,args.map_kind,args.map_number,
                         args.play_episodes,not args.no_render,args.render_fps,cfg); return

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type=="cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | AMP: {cfg.use_amp} | train_every: {cfg.train_every_steps}")

    levels=[int(l) for l in (args.levels or [args.map_number])]
    for lvl in levels: _apply_cli_overrides(lvl,args)

    probe=_make_env(args.map_kind,levels[0],10); probe.reset()
    n_actions=probe.action_space_n; probe.close()

    if args.individual_levels:
        for lvl in levels:
            print(f"\n=== Independent training — level {lvl} ===")
            agent=DQNAgent(n_actions,cfg,device)
            if not args.no_compile:
                agent.policy_net=_maybe_compile(agent.policy_net,device)
                agent.target_net=_maybe_compile(agent.target_net,device)
            out=ckpt_dir/f"dqn_level{lvl}_individual.pt"
            summary=train_one_level(agent,lvl,cfg,args.map_kind,
                                    args.target_collect_rate,args.target_success_rate,out)
            agent.save(out,lvl,args.map_kind); print(f"Saved {out} | {summary}")
        return

    agent=DQNAgent(n_actions,cfg,device)
    if not args.no_compile:
        agent.policy_net=_maybe_compile(agent.policy_net,device)
        agent.target_net=_maybe_compile(agent.target_net,device)

    if args.resume and model_path.exists():
        meta=agent.load(model_path)
        print(f"Resumed from {model_path} (level={meta['level']},kind={meta['map_kind']})")

    for lvl in levels:
        print(f"\n=== Sequential training — level {lvl} ===")
        out=ckpt_dir/f"dqn_level{lvl}_sequential.pt"
        summary=train_one_level(agent,lvl,cfg,args.map_kind,
                                args.target_collect_rate,args.target_success_rate,out)
        agent.save(out,lvl,args.map_kind)
        print(f"Saved {out} | mean_reward={summary['mean_reward']:.2f} | "
              f"progress={summary['progress_rate']:.2%} | "
              f"all_collected={summary['all_collected_rate']:.2%} | "
              f"success={summary['success_rate']:.2%}")

    agent.save(model_path,levels[-1],args.map_kind)
    print(f"\nFinal model saved to: {model_path}")

if __name__=="__main__":
    _main()
