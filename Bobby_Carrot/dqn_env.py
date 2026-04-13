"""Bobby Carrot DQN Environment — optimised physics, BFS, and reward shaping.

Extracted from the monolithic train_dqn.py into a focused module.
Key improvements over the original:
  - Unified single-pass BFS (was 2 separate BFS calls per step)
  - Pre-computed adjacency validity table (vectorised _can_move_tile)
  - Stronger path-optimality rewards (BFS Δ 0.30, escalating revisit penalty)
  - 13-channel grid observation with directional BFS hint
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Locate Game_Python ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent
while not (_ROOT / "Game_Python").exists() and _ROOT.parent != _ROOT:
    _ROOT = _ROOT.parent
_GAME_DIR = _ROOT / "Game_Python"
if not _GAME_DIR.exists():
    raise RuntimeError(
        f"Cannot find Game_Python directory starting from {_HERE}.\n"
        "Place this file inside the project root (next to Game_Python/)."
    )
if str(_GAME_DIR) not in sys.path:
    sys.path.insert(0, str(_GAME_DIR))

from bobby_carrot.game import Map, MapInfo  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
N_ACTIONS = 4
GRID_CHANNELS = 13  # 10 semantic + agent_pos + visited + direction_hint
INV_FEATURES = 8

# ── Tile class lookup table ───────────────────────────────────────────────────
# 0=wall  1=floor  2=carrot  3=egg  4=finish-inactive  5=finish-active
# 6=crumble  7=hole  8=conveyor  9=key/lock/switch/arrow
_TILE_CH = np.zeros(256, dtype=np.uint8)
for _v in range(256):
    if   _v < 18:                                                    _TILE_CH[_v] = 0
    elif _v in (18, 20, 46, 21):                                     _TILE_CH[_v] = 1
    elif _v == 19:                                                   _TILE_CH[_v] = 2
    elif _v == 45:                                                   _TILE_CH[_v] = 3
    elif _v == 44:                                                   _TILE_CH[_v] = 4
    elif _v == 30:                                                   _TILE_CH[_v] = 6
    elif _v == 31:                                                   _TILE_CH[_v] = 7
    elif _v in (40, 41, 42, 43):                                     _TILE_CH[_v] = 8
    elif _v in (22,23,24,25,26,27,28,29,32,33,34,35,36,37,38,39):   _TILE_CH[_v] = 9
    else:                                                            _TILE_CH[_v] = 1

# Pre-built one-hot table: shape (10, 256)
_ONE_HOT = (
    np.arange(10, dtype=np.float32)[:, None]
    == _TILE_CH[None, :].astype(np.float32)
).astype(np.float32)

# Switch LUTs
_LUT_RED_F = np.array([22,23,24,25,26,27,28,29], dtype=np.uint8)
_LUT_RED_T = np.array([23,22,25,26,27,24,29,28], dtype=np.uint8)
_LUT_BLU_F = np.array([38,39,40,41,42,43], dtype=np.uint8)
_LUT_BLU_T = np.array([39,38,41,40,43,42], dtype=np.uint8)

# Crumble/arrow tile transitions on leaving
_LEAVE_FROM = np.array([30,24,25,26,27,28,29], dtype=np.uint8)
_LEAVE_TO   = np.array([31,25,26,27,24,29,28], dtype=np.uint8)
_LEAVE_DICT = {int(f): int(t) for f, t in zip(_LEAVE_FROM, _LEAVE_TO)}

# ── Direction deltas  ─────────────────────────────────────────────────────────
# ACTION indices: 0=Left(-1,0)  1=Right(+1,0)  2=Up(0,-1)  3=Down(0,+1)
ACTION_DELTA: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_FLAT_DELTAS = np.array([-1, 1, -16, 16], dtype=np.int32)


def _apply_lut(md: np.ndarray, frm: np.ndarray, to: np.ndarray) -> None:
    """In-place vectorised LUT swap on 256-element tile array."""
    tmp = md.copy()
    for f, t in zip(frm, to):
        md[tmp == f] = t


# ── Physics helpers ───────────────────────────────────────────────────────────
def _can_move_tile(t_from: int, t_to: int, delta: int) -> bool:
    """Exact mirror of game.py update_dest conveyor/arrow restrictions."""
    if t_to < 18 or t_to == 31 or t_to == 46:
        return False
    # Exit restrictions (current tile)
    if t_from == 40 and delta != -1:  return False
    if t_from == 41 and delta != 1:   return False
    if t_from == 42 and delta != -16: return False
    if t_from == 43 and delta != 16:  return False
    if t_from in (28, 40, 41) and delta in (-16, 16): return False
    if t_from in (29, 42, 43) and delta in (-1, 1):   return False
    # Entry restrictions (destination tile)
    if t_to == 40 and delta != -1:  return False
    if t_to == 41 and delta != 1:   return False
    if t_to == 42 and delta != -16: return False
    if t_to == 43 and delta != 16:  return False
    if t_to in (28, 40, 41) and delta in (-16, 16): return False
    if t_to in (29, 42, 43) and delta in (-1, 1):   return False
    return True


# ── Pre-computed adjacency validity (vectorised BFS) ─────────────────────────
# _ADJ_VALID[from_tile][to_tile][dir_idx] = True/False
# dir_idx: 0=-1, 1=+1, 2=-16, 3=+16
_ADJ_VALID = np.zeros((256, 256, 4), dtype=np.bool_)
for _ft in range(256):
    for _tt in range(256):
        for _di, _dd in enumerate([-1, 1, -16, 16]):
            _ADJ_VALID[_ft, _tt, _di] = _can_move_tile(_ft, _tt, _dd)


def _bfs_unified(md: np.ndarray, start: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Single BFS pass returning (dist_normal, dist_no_crumble).

    Instead of running two separate BFS (one normal, one with crumble=wall),
    we run ONE BFS and track crumble-safe distances simultaneously.
    Cells reached via a crumble tile (30) get dist_safe = -1.
    """
    dist = np.full(256, -1, dtype=np.int32)
    dist_safe = np.full(256, -1, dtype=np.int32)
    dist[start] = 0
    # Start is crumble-safe unless it IS a crumble tile
    if md[start] != 30:
        dist_safe[start] = 0

    frontier = np.array([start], dtype=np.int32)
    d = 1
    while frontier.size:
        next_cells: List[np.ndarray] = []
        for di, delta in enumerate([-1, 1, -16, 16]):
            cands = frontier + delta
            # Boundary checks
            if delta == -1:    bound = (frontier % 16) > 0
            elif delta == 1:   bound = (frontier % 16) < 15
            elif delta == -16: bound = frontier >= 16
            else:              bound = frontier <= 239
            cands     = cands[bound]
            fr_masked = frontier[bound]
            if not cands.size:
                continue
            # Only unvisited
            unvis = dist[cands] == -1
            cands     = cands[unvis]
            fr_masked = fr_masked[unvis]
            if not cands.size:
                continue
            # Vectorised validity check via pre-computed table
            keep = _ADJ_VALID[md[fr_masked], md[cands], di]
            cands     = cands[keep]
            fr_masked = fr_masked[keep]
            if cands.size:
                dist[cands] = d
                # Crumble-safe distance: propagate only if NEITHER the
                # source NOR destination is a crumble tile, AND the source
                # was crumble-safe itself.
                safe_source = dist_safe[fr_masked] >= 0
                not_crumble_dest = md[cands] != 30
                safe_mask = safe_source & not_crumble_dest
                safe_cands = cands[safe_mask]
                if safe_cands.size:
                    # Only set if not already set (first visit = shortest)
                    unset = dist_safe[safe_cands] == -1
                    dist_safe[safe_cands[unset]] = d
                next_cells.append(cands)
        if next_cells:
            frontier = np.unique(np.concatenate(next_cells))
        else:
            break
        d += 1
    return dist, dist_safe


def _count_reachable(bfs: np.ndarray, md: np.ndarray) -> int:
    """Count remaining targets (carrots + eggs) reachable from agent via BFS."""
    target_mask = (md == 19) | (md == 45)
    if not target_mask.any():
        return 0
    return int(np.sum(bfs[target_mask] >= 0))


def _auto_max_steps(md: np.ndarray) -> int:
    passable  = int(np.sum(md >= 18))
    n_targets = int(np.sum((md == 19) | (md == 45)))
    return max(400, passable * 5 + n_targets * 10)


# ── Environment ───────────────────────────────────────────────────────────────
class BobbyEnv:
    """Lightweight, headless RL environment for Bobby Carrot.

    Implements game physics directly on a numpy tile array for maximum
    training throughput while keeping exact parity with game.py rules.
    """

    def __init__(self, map_kind: str, map_number: int,
                 max_steps: Optional[int] = None) -> None:
        # Validate level bounds
        if map_kind == "normal" and not (1 <= map_number <= 30):
            raise ValueError(f"Normal levels must be 1-30, got {map_number}")
        if map_kind == "egg" and not (1 <= map_number <= 20):
            raise ValueError(f"Egg levels must be 1-20, got {map_number}")

        self.map_kind   = map_kind
        self.map_number = map_number
        self._map_obj   = Map(map_kind, map_number)
        self._fresh: Optional[MapInfo] = None
        self._max_steps_override = max_steps

        self.md          = np.zeros(256, dtype=np.uint8)
        self.pos         = (0, 0)
        self.n_targets   = 0
        self.n_collected = 0
        self.key_gray = self.key_yellow = self.key_red = 0
        self.dead = self.finished = False
        self.step_count  = 0
        self.max_steps   = 400
        self._visited    = np.zeros(256, dtype=np.float32)
        self._visit_counts = np.zeros(256, dtype=np.int32)  # escalating revisit
        # BFS cache (unified)
        self._bfs: Optional[np.ndarray] = None
        self._bfs_safe: Optional[np.ndarray] = None
        self._bfs_dirty  = True
        # Observation cache
        self._obs_dirty  = True
        self._cached_grid: Optional[np.ndarray] = None
        self._cached_inv:  Optional[np.ndarray] = None
        # Reward shaping state
        self._prev_reachable = 0
        self._prev_bfs_dist  = -1
        self._stall_steps = 0
        self._stall_limit = 150
        # Collection streak tracking (Rainbow-lite)
        self._steps_since_collect = 0
        self._collect_streak = 0

    def set_level(self, level: int) -> None:
        """Switch to a different level (for multi-level training). Call reset() after."""
        self.map_number = level
        self._map_obj = Map(self.map_kind, level)
        self._fresh = None

    def _load_fresh(self) -> None:
        self._fresh = self._map_obj.load_map_info()

    def reset(self) -> None:
        if self._fresh is None:
            self._load_fresh()
        mi = self._fresh
        self.md          = np.array(mi.data, dtype=np.uint8)
        self.pos         = mi.coord_start
        self.n_targets   = int(mi.carrot_total) + int(mi.egg_total)
        self.n_collected = 0
        self.key_gray = self.key_yellow = self.key_red = 0
        self.dead = self.finished = False
        self.step_count  = 0
        self.max_steps   = (self._max_steps_override if self._max_steps_override
                            else _auto_max_steps(self.md))
        self._visited    = np.zeros(256, dtype=np.float32)
        self._visit_counts = np.zeros(256, dtype=np.int32)
        px, py = self.pos
        self._visited[px + py * 16] = 1.0
        self._visit_counts[px + py * 16] = 1
        self._bfs_dirty  = True
        self._bfs        = None
        self._bfs_safe   = None
        self._obs_dirty  = True
        self._cached_grid = None
        self._cached_inv  = None
        # Stall
        self._stall_steps = 0
        self._stall_limit = max(200, self.max_steps // 2)
        # Collection streak (Rainbow-lite)
        self._steps_since_collect = 0
        self._collect_streak = 0
        # Compute initial BFS + reachability
        bfs, _ = self._get_bfs_pair()
        self._prev_reachable = _count_reachable(bfs, self.md)
        self._prev_bfs_dist = self._bfs_to_nearest(bfs)

    def _can_move(self, fx: int, fy: int, dx: int, dy: int) -> bool:
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < 16 and 0 <= ny < 16):
            return False
        delta = dx + dy * 16
        t_f   = int(self.md[fx + fy * 16])
        t_t   = int(self.md[nx + ny * 16])
        if t_t == 33 and self.key_gray   == 0: return False
        if t_t == 35 and self.key_yellow == 0: return False
        if t_t == 37 and self.key_red    == 0: return False
        return _can_move_tile(t_f, t_t, delta)

    def _bfs_to_nearest(self, bfs: Optional[np.ndarray] = None) -> int:
        """Min BFS distance to nearest remaining target (or finish if all done).
        PREFERS targets reachable without crossing crumble tiles.
        """
        if bfs is None:
            bfs, _ = self._get_bfs_pair()
        bfs_safe = self._bfs_safe
        all_done = self.n_collected == self.n_targets
        tmask = (self.md == 44) if all_done else ((self.md == 19) | (self.md == 45))
        if not tmask.any():
            return -1
        # Prefer targets in current section (no crumble crossing)
        if bfs_safe is not None:
            safe_dists = bfs_safe[tmask]
            safe_valid = safe_dists[safe_dists >= 0]
            if safe_valid.size:
                return int(safe_valid.min())
        # Fallback: must cross crumble to reach any target
        tdist = bfs[tmask]
        valid = tdist[tdist >= 0]
        return int(valid.min()) if valid.size else -1

    def step(self, action: int) -> Tuple[float, bool, bool]:
        """Execute action. Returns (reward, done, won)."""
        dx, dy = ACTION_DELTA[action]
        fx, fy = self.pos

        # ── Invalid move ──────────────────────────────────────────────────
        if not self._can_move(fx, fy, dx, dy):
            self.step_count += 1
            self._stall_steps += 1
            done = (self.step_count >= self.max_steps
                    or self._stall_steps >= self._stall_limit)
            return -0.2, done, False

        nx, ny   = fx + dx, fy + dy
        from_idx = fx + fy * 16
        to_idx   = nx + ny * 16
        t_from   = int(self.md[from_idx])
        t_to     = int(self.md[to_idx])
        reward   = -0.02
        collected = False

        # ── Leaving tile mutations ────────────────────────────────────────
        if t_from in _LEAVE_DICT:
            self.md[from_idx] = _LEAVE_DICT[t_from]

        # ── Landing tile effects ──────────────────────────────────────────
        if t_to == 31:  # death hole
            self.dead = True
            self.pos  = (nx, ny)
            self.step_count += 1
            self._bfs_dirty = True
            self._obs_dirty = True
            return -30.0, True, False

        if t_to == 19:  # carrot
            self.md[to_idx] = 20
            self.n_collected += 1
            reward += 5.0
            collected = True
        elif t_to == 45:  # egg
            self.md[to_idx] = 46
            self.n_collected += 1
            reward += 5.0
            collected = True
        elif t_to == 32:  # gray key
            self.md[to_idx] = 18; self.key_gray   += 1
        elif t_to == 33 and self.key_gray > 0:  # gray lock
            self.md[to_idx] = 18; self.key_gray   -= 1
        elif t_to == 34:  # yellow key
            self.md[to_idx] = 18; self.key_yellow += 1
        elif t_to == 35 and self.key_yellow > 0:  # yellow lock
            self.md[to_idx] = 18; self.key_yellow -= 1
        elif t_to == 36:  # red key
            self.md[to_idx] = 18; self.key_red    += 1
        elif t_to == 37 and self.key_red > 0:  # red lock
            self.md[to_idx] = 18; self.key_red    -= 1
        elif t_to == 22:  # red switch
            _apply_lut(self.md, _LUT_RED_F, _LUT_RED_T)
        elif t_to == 38:  # blue switch
            _apply_lut(self.md, _LUT_BLU_F, _LUT_BLU_T)

        # Update position
        self.pos = (nx, ny)
        self._bfs_dirty = True
        self._obs_dirty = True

        # ── Escalating revisit penalty (softened for maze backtracking) ──
        self._visit_counts[to_idx] += 1
        vc = self._visit_counts[to_idx]
        new_tile = self._visited[to_idx] == 0.0
        if new_tile:
            self._visited[to_idx] = 1.0
            reward += 0.1
        else:
            # Softened cap: -0.10 instead of -0.15 (L4 requires backtracking)
            revisit_pen = min(0.10, 0.05 * vc)
            reward -= revisit_pen

        self.step_count += 1
        self._steps_since_collect += 1

        # ── Win condition ─────────────────────────────────────────────────
        if t_to == 44 and self.n_collected == self.n_targets:
            self.finished = True
            return reward + 50.0, True, True

        # ── Collection streak & efficiency bonus (Rainbow-lite) ───────────
        if collected and self.n_targets > 0:
            progress = self.n_collected / self.n_targets

            # Streak bonus: rapid successive collections are rewarded
            if self._steps_since_collect <= 15:
                self._collect_streak += 1
                reward += 0.5 * self._collect_streak  # escalating streak
            else:
                self._collect_streak = 1

            # Path efficiency: bonus for finding short paths between targets
            efficiency = max(0.0, 1.0 - self._steps_since_collect / max(1.0, float(self.max_steps)))
            reward += 1.0 * efficiency

            self._steps_since_collect = 0

            # Milestone bonuses
            if self.n_collected == self.n_targets:
                reward += 10.0  # all targets collected
            elif progress >= 0.75 and (self.n_collected - 1) / self.n_targets < 0.75:
                reward += 3.0   # 75% milestone
            elif progress >= 0.50 and (self.n_collected - 1) / self.n_targets < 0.50:
                reward += 2.0   # 50% milestone
            elif progress >= 0.25 and (self.n_collected - 1) / self.n_targets < 0.25:
                reward += 1.0   # 25% milestone

        # ── Reachability check ────────────────────────────────────────────
        bfs, _ = self._get_bfs_pair()
        remaining = int(np.sum((self.md == 19) | (self.md == 45)))

        if remaining > 0:
            reachable = _count_reachable(bfs, self.md)
            expected = self._prev_reachable - (1 if collected else 0)
            lost = max(0, expected - reachable)
            if lost > 0:
                reward -= 10.0 * lost
            self._prev_reachable = reachable
            if reachable == 0:
                return reward - 30.0, True, False
        else:
            self._prev_reachable = 0
            exit_mask = self.md == 44
            if exit_mask.any() and not np.any(bfs[exit_mask] >= 0):
                return reward - 30.0, True, False

        # ── BFS distance shaping (STRONGER: 0.50) ────────────────────────
        bfs_to = self._bfs_to_nearest(bfs)
        if bfs_to >= 0 and self._prev_bfs_dist >= 0 and not collected:
            delta_d = self._prev_bfs_dist - bfs_to
            reward += 0.50 * delta_d  # was 0.30, increased for faster convergence
        self._prev_bfs_dist = bfs_to

        # ── Time pressure (quadratic) ─────────────────────────────────────
        t_frac = float(self.step_count) / max(1.0, float(self.max_steps))
        reward -= 0.01 * t_frac * t_frac

        # ── Stall detection ───────────────────────────────────────────────
        if collected or new_tile:
            self._stall_steps = 0
        else:
            self._stall_steps += 1

        done = (self.step_count >= self.max_steps
                or self._stall_steps >= self._stall_limit)
        return reward, done, False

    def _get_bfs_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (bfs_normal, bfs_safe) using unified single-pass BFS."""
        if self._bfs_dirty or self._bfs is None:
            px, py = self.pos
            self._bfs, self._bfs_safe = _bfs_unified(self.md, px + py * 16)
            self._bfs_dirty = False
        return self._bfs, self._bfs_safe

    # Keep legacy names for compatibility with teacher action code
    def get_bfs(self) -> np.ndarray:
        bfs, _ = self._get_bfs_pair()
        return bfs

    def get_bfs_safe(self) -> np.ndarray:
        _, bfs_safe = self._get_bfs_pair()
        return bfs_safe

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """13-channel grid + 8-dim inventory vector."""
        if not self._obs_dirty and self._cached_grid is not None:
            return self._cached_grid, self._cached_inv  # type: ignore[return-value]

        px, py = self.pos
        md     = self.md

        # 13-channel grid
        grid = np.zeros((GRID_CHANNELS, 16, 16), dtype=np.float32)

        # Channels 0-9: semantic one-hot
        grid[:10] = _ONE_HOT[:, md].reshape(10, 16, 16)

        # Promote finish channel 4->5 when all collected
        if self.n_collected == self.n_targets:
            grid[5] = grid[4]
            grid[4] = 0.0

        # BFS navigation heatmap in channel 9
        bfs, bfs_safe = self._get_bfs_pair()
        all_done = self.n_collected == self.n_targets
        tmask = (md == 44) if all_done else ((md == 19) | (md == 45))

        safe_reachable = tmask & (bfs_safe >= 0)
        if safe_reachable.any():
            bfs_for_grad = bfs_safe
            bfs_to_target = int(bfs_safe[safe_reachable].min())
        else:
            bfs_for_grad = bfs
            bfs_to_target = 32
            if tmask.any():
                tdist = bfs[tmask]
                valid = tdist[tdist >= 0]
                if valid.size:
                    bfs_to_target = int(valid.min())

        reachable = bfs_for_grad >= 0
        if reachable.any():
            max_d = max(1, int(bfs_for_grad[reachable].max()))
            grad  = np.where(reachable, 1.0 - bfs_for_grad.astype(np.float32) / max_d, 0.0)
        else:
            grad  = np.zeros(256, dtype=np.float32)
        grid[9] = grad.reshape(16, 16)

        # Channel 10 — agent position
        grid[10, py, px] = 1.0

        # Channel 11 — visited mask
        grid[11] = self._visited.reshape(16, 16)

        # Channel 12 — directional BFS hint (NEW)
        # Encodes normalized direction to nearest target at each reachable cell
        dir_hint = np.zeros(256, dtype=np.float32)
        for di, delta in enumerate([-1, 1, -16, 16]):
            # For the agent's current position, mark which neighbor is closest
            agent_idx = px + py * 16
            ni = agent_idx + delta
            if 0 <= ni < 256:
                nx_c, ny_c = ni % 16, ni // 16
                # Boundary check
                if delta == -1 and px == 0: continue
                if delta == 1 and px == 15: continue
                if delta == -16 and py == 0: continue
                if delta == 16 and py == 15: continue
                if bfs_for_grad[ni] >= 0 and bfs_for_grad[ni] < bfs_for_grad[agent_idx]:
                    dir_hint[ni] = 1.0
        grid[12] = dir_hint.reshape(16, 16)

        # Inventory vector (8 features)
        n_rem     = self.n_targets - self.n_collected
        rem       = float(n_rem) / max(1.0, float(self.n_targets))
        dist_n    = min(1.0, bfs_to_target / 32.0)
        phase     = 1.0 if all_done else 0.0
        progress  = float(self.n_collected) / max(1.0, float(self.n_targets))
        has_key   = 1.0 if (self.key_gray + self.key_yellow + self.key_red) > 0 else 0.0
        visit_rat = float(np.sum(self._visited > 0)) / 256.0
        reach_rat = (float(self._prev_reachable) / max(1.0, float(n_rem))
                     if n_rem > 0 else 1.0)
        steps_rem = 1.0 - float(self.step_count) / max(1.0, float(self.max_steps))

        inv = np.array([rem, dist_n, phase, progress, has_key, visit_rat,
                        reach_rat, steps_rem], dtype=np.float32)

        self._cached_grid = grid
        self._cached_inv  = inv
        self._obs_dirty   = False
        return grid, inv
