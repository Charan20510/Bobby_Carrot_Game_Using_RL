from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── locate Game_Python ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent
while not (_ROOT / "Game_Python").exists() and _ROOT.parent != _ROOT:
    _ROOT = _ROOT.parent
_GAME_DIR = _ROOT / "Game_Python"
if not _GAME_DIR.exists():
    raise RuntimeError(
        f"Cannot find Game_Python directory starting from {_HERE}.\n"
        "Place train_dqn.py inside the project root (next to Game_Python/)."
    )
if str(_GAME_DIR) not in sys.path:
    sys.path.insert(0, str(_GAME_DIR))

from bobby_carrot.game import (  # noqa: E402
    Map, MapInfo, Bobby, State, Assets,
    VIEW_WIDTH, VIEW_HEIGHT, FRAMES, WIDTH_POINTS_DELTA, HEIGHT_POINTS_DELTA,
    VIEW_WIDTH_POINTS, VIEW_HEIGHT_POINTS, FRAMES_PER_STEP, WIDTH_POINTS, HEIGHT_POINTS,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as e:
    raise RuntimeError("PyTorch is required. Install it first.") from e


# ── Constants ─────────────────────────────────────────────────────────────────
N_ACTIONS     = 4
GRID_CHANNELS = 12   # R6: 10 semantic + agent_pos + visited
INV_FEATURES  = 8    # rem, bfs_norm, phase, progress, has_key, visit_ratio,
                      #   reachable_ratio, steps_remaining_norm

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
    elif _v in (22,23,24,25,26,27,28,29,32,33,34,35,36,37,38,39):  _TILE_CH[_v] = 9
    else:                                                            _TILE_CH[_v] = 1

# S2: Prebuilt one-hot table: shape (10, 256)
_ONE_HOT = (
    np.arange(10, dtype=np.float32)[:, None]
    == _TILE_CH[None, :].astype(np.float32)
).astype(np.float32)  # (10, 256)

# S6: Switch LUTs
_LUT_RED_F = np.array([22,23,24,25,26,27,28,29], dtype=np.uint8)
_LUT_RED_T = np.array([23,22,25,26,27,24,29,28], dtype=np.uint8)
_LUT_BLU_F = np.array([38,39,40,41,42,43], dtype=np.uint8)
_LUT_BLU_T = np.array([39,38,41,40,43,42], dtype=np.uint8)

# Crumble/arrow tile transitions on leaving
_LEAVE_FROM = np.array([30,24,25,26,27,28,29], dtype=np.uint8)
_LEAVE_TO   = np.array([31,25,26,27,24,29,28], dtype=np.uint8)
_LEAVE_DICT = {int(f): int(t) for f, t in zip(_LEAVE_FROM, _LEAVE_TO)}


def _apply_lut(md: np.ndarray, frm: np.ndarray, to: np.ndarray) -> None:
    """In-place vectorised LUT swap on 256-element tile array."""
    tmp = md.copy()
    for f, t in zip(frm, to):
        md[tmp == f] = t


# ── Physics helpers ───────────────────────────────────────────────────────────
def _can_move_tile(t_from: int, t_to: int, delta: int) -> bool:
    """Exact mirror of game.py update_dest conveyor/arrow restrictions."""
    if t_to < 18 or t_to == 31: return False
    # Exit restrictions (current tile)
    if t_from == 40 and delta != -1:  return False
    if t_from == 41 and delta != 1:   return False
    if t_from == 42 and delta != -16: return False
    if t_from == 43 and delta != 16:  return False
    if t_from in (28,40,41) and delta in (-16,16): return False
    if t_from in (29,42,43) and delta in (-1,1):   return False
    # Entry restrictions (destination tile)
    if t_to == 40 and delta != -1:  return False
    if t_to == 41 and delta != 1:   return False
    if t_to == 42 and delta != -16: return False
    if t_to == 43 and delta != 16:  return False
    if t_to in (28,40,41) and delta in (-16,16): return False
    if t_to in (29,42,43) and delta in (-1,1):   return False
    return True


def _bfs_numpy(md: np.ndarray, start: int) -> np.ndarray:
    """S5: BFS via numpy frontier expansion."""
    dist = np.full(256, -1, dtype=np.int32)
    dist[start] = 0
    frontier = np.array([start], dtype=np.int32)
    d = 1
    while frontier.size:
        next_cells: List[np.ndarray] = []
        for delta in (-1, 1, -16, 16):
            cands = frontier + delta
            if delta == -1:    bound = (frontier % 16) > 0
            elif delta == 1:   bound = (frontier % 16) < 15
            elif delta == -16: bound = frontier >= 16
            else:              bound = frontier <= 239
            cands     = cands[bound]
            fr_masked = frontier[bound]
            if not cands.size: continue
            unvis     = dist[cands] == -1
            cands     = cands[unvis];  fr_masked = fr_masked[unvis]
            if not cands.size: continue
            keep = np.array([
                _can_move_tile(int(md[f]), int(md[c]), delta)
                for f, c in zip(fr_masked, cands)
            ], dtype=bool)
            cands = cands[keep]
            if cands.size:
                dist[cands] = d
                next_cells.append(cands)
        if next_cells:
            frontier = np.unique(np.concatenate(next_cells))
        else:
            break
        d += 1
    return dist


def _bfs_no_crumble(md: np.ndarray, start: int) -> np.ndarray:
    """BFS treating crumble tiles (30) as walls.
    Finds targets reachable in the CURRENT section without crossing
    any crumble tile boundary. This guides the agent to collect all
    targets in one section before moving to the next.
    """
    md_safe = md.copy()
    md_safe[md_safe == 30] = 0  # crumble → impassable
    return _bfs_numpy(md_safe, start)


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
    ACTION_DELTA: List[Tuple[int,int]] = [(-1,0),(1,0),(0,-1),(0,1)]

    def __init__(self, map_kind: str, map_number: int,
                 max_steps: Optional[int] = None) -> None:
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
        self._bfs: Optional[np.ndarray] = None
        self._bfs_dirty  = True
        self._bfs_safe: Optional[np.ndarray] = None
        self._bfs_safe_dirty = True
        # S2: observation cache
        self._obs_dirty  = True
        self._cached_grid: Optional[np.ndarray] = None
        self._cached_inv:  Optional[np.ndarray] = None
        # R2: reachability tracking
        self._prev_reachable = 0
        # R3: BFS distance shaping
        self._prev_bfs_dist  = -1
        # R7: stall detection
        self._stall_steps = 0
        self._stall_limit = 150

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
        px, py = self.pos
        self._visited[px + py * 16] = 1.0
        self._bfs_dirty  = True
        self._bfs        = None
        self._bfs_safe_dirty = True
        self._bfs_safe   = None
        self._obs_dirty  = True
        self._cached_grid = None
        self._cached_inv  = None
        # R7: stall
        self._stall_steps = 0
        self._stall_limit = max(100, self.max_steps // 3)
        # R2: compute initial reachability
        bfs = self.get_bfs()
        self._prev_reachable = _count_reachable(bfs, self.md)
        # R3: initial BFS distance
        self._prev_bfs_dist = self._bfs_to_nearest(bfs)

    def _can_move(self, fx: int, fy: int, dx: int, dy: int) -> bool:
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < 16 and 0 <= ny < 16): return False
        delta = dx + dy * 16
        t_f   = int(self.md[fx + fy * 16])
        t_t   = int(self.md[nx + ny * 16])
        if t_t == 33 and self.key_gray   == 0: return False
        if t_t == 35 and self.key_yellow == 0: return False
        if t_t == 37 and self.key_red    == 0: return False
        return _can_move_tile(t_f, t_t, delta)

    def _bfs_to_nearest(self, bfs: Optional[np.ndarray] = None) -> int:
        """Min BFS distance to nearest remaining target (or finish if all done).
        PREFERS targets reachable without crossing crumble tiles (current section).
        Falls back to normal BFS if no targets exist in the current section.
        """
        if bfs is None:
            bfs = self.get_bfs()
        all_done = self.n_collected == self.n_targets
        tmask = (self.md == 44) if all_done else ((self.md == 19) | (self.md == 45))
        if not tmask.any():
            return -1
        # Prefer targets in current section (no crumble crossing)
        bfs_safe = self.get_bfs_safe()
        safe_dists = bfs_safe[tmask]
        safe_valid = safe_dists[safe_dists >= 0]
        if safe_valid.size:
            return int(safe_valid.min())
        # Fallback: must cross crumble to reach any target
        tdist = bfs[tmask]
        valid = tdist[tdist >= 0]
        return int(valid.min()) if valid.size else -1

    def step(self, action: int) -> Tuple[float, bool, bool]:
        dx, dy = self.ACTION_DELTA[action]
        fx, fy = self.pos

        # ── Invalid move ──────────────────────────────────────────────────
        if not self._can_move(fx, fy, dx, dy):
            self.step_count += 1
            self._stall_steps += 1
            done = (self.step_count >= self.max_steps
                    or self._stall_steps >= self._stall_limit)
            # Obs unchanged on invalid move — keep cache (S2)
            return -0.2, done, False

        nx, ny   = fx + dx, fy + dy
        from_idx = fx + fy * 16
        to_idx   = nx + ny * 16
        t_from   = int(self.md[from_idx])
        t_to     = int(self.md[to_idx])
        reward   = -0.02   # R8: mild step cost
        mutated  = False
        collected = False

        # ── Leaving tile mutations ────────────────────────────────────────
        if t_from in _LEAVE_DICT:
            self.md[from_idx] = _LEAVE_DICT[t_from]
            mutated = True

        # ── Landing tile effects ──────────────────────────────────────────
        if t_to == 31:                                    # death hole
            self.dead = True
            self.pos  = (nx, ny)
            self.step_count += 1
            self._bfs_dirty = True
            self._obs_dirty = True
            return -30.0, True, False

        if t_to == 19:                                    # carrot
            self.md[to_idx] = 20
            self.n_collected += 1
            reward += 5.0
            mutated = True
            collected = True
        elif t_to == 45:                                  # egg
            self.md[to_idx] = 46
            self.n_collected += 1
            reward += 5.0
            mutated = True
            collected = True
        elif t_to == 32:                                  # gray key
            self.md[to_idx] = 18; self.key_gray   += 1; mutated = True
        elif t_to == 33 and self.key_gray > 0:            # gray lock
            self.md[to_idx] = 18; self.key_gray   -= 1; mutated = True
        elif t_to == 34:                                  # yellow key
            self.md[to_idx] = 18; self.key_yellow += 1; mutated = True
        elif t_to == 35 and self.key_yellow > 0:          # yellow lock
            self.md[to_idx] = 18; self.key_yellow -= 1; mutated = True
        elif t_to == 36:                                  # red key
            self.md[to_idx] = 18; self.key_red    += 1; mutated = True
        elif t_to == 37 and self.key_red > 0:             # red lock
            self.md[to_idx] = 18; self.key_red    -= 1; mutated = True
        elif t_to == 22:                                  # red switch
            _apply_lut(self.md, _LUT_RED_F, _LUT_RED_T); mutated = True
        elif t_to == 38:                                  # blue switch
            _apply_lut(self.md, _LUT_BLU_F, _LUT_BLU_T); mutated = True

        # Update position
        self.pos = (nx, ny)

        # R1: BFS is ALWAYS dirty after a valid move (position changed)
        self._bfs_dirty = True
        self._bfs_safe_dirty = True

        # ── New tile exploration bonus / revisit penalty ──────────────────
        new_tile = self._visited[to_idx] == 0.0
        if new_tile:
            self._visited[to_idx] = 1.0
            reward += 0.1
        else:
            reward -= 0.05   # discourage revisiting already-explored tiles

        self.step_count += 1
        self._obs_dirty = True

        # ── Win condition ─────────────────────────────────────────────────
        if t_to == 44 and self.n_collected == self.n_targets:
            self.finished = True
            return reward + 50.0, True, True

        # ── All-targets-collected milestone bonus ─────────────────────────
        # Fires when agent collects the LAST target but is NOT on the exit yet.
        # Strongly incentivizes 100% collection before navigating to exit.
        if collected and self.n_collected == self.n_targets:
            reward += 10.0

        # ── R2: Reachability check ────────────────────────────────────────
        # Compute BFS from new position (forces recompute since _bfs_dirty)
        bfs = self.get_bfs()
        remaining = int(np.sum((self.md == 19) | (self.md == 45)))

        if remaining > 0:
            reachable = _count_reachable(bfs, self.md)
            # Expected reachable = prev - collected_this_step
            expected = self._prev_reachable - (1 if collected else 0)
            lost = max(0, expected - reachable)
            if lost > 0:
                reward -= 10.0 * lost  # heavy penalty per permanently lost target
            self._prev_reachable = reachable
            # Catastrophic: targets remain but NONE are reachable (crumble blocked)
            if reachable == 0:
                return reward - 30.0, True, False
        else:
            self._prev_reachable = 0
            # All targets collected — verify exit is still reachable
            exit_mask = self.md == 44
            if exit_mask.any() and not np.any(bfs[exit_mask] >= 0):
                return reward - 30.0, True, False  # exit unreachable

        # ── R3: BFS distance shaping (stronger) ──────────────────────────
        bfs_to = self._bfs_to_nearest(bfs)
        if bfs_to >= 0 and self._prev_bfs_dist >= 0 and not collected:
            delta_d = self._prev_bfs_dist - bfs_to
            reward += 0.15 * delta_d  # strong directional guidance
        self._prev_bfs_dist = bfs_to

        # ── R7: Stall detection ───────────────────────────────────────────
        if collected or new_tile:
            self._stall_steps = 0
        else:
            self._stall_steps += 1

        done = (self.step_count >= self.max_steps
                or self._stall_steps >= self._stall_limit)
        return reward, done, False

    def get_bfs(self) -> np.ndarray:
        if self._bfs_dirty or self._bfs is None:
            px, py    = self.pos
            self._bfs = _bfs_numpy(self.md, px + py * 16)
            self._bfs_dirty = False
        return self._bfs

    def get_bfs_safe(self) -> np.ndarray:
        """BFS treating crumble tiles as walls — for current-section navigation."""
        if self._bfs_safe_dirty or self._bfs_safe is None:
            px, py = self.pos
            self._bfs_safe = _bfs_no_crumble(self.md, px + py * 16)
            self._bfs_safe_dirty = False
        return self._bfs_safe

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """S2: Cached; R6: 12-channel grid."""
        if not self._obs_dirty and self._cached_grid is not None:
            return self._cached_grid, self._cached_inv  # type: ignore[return-value]

        px, py = self.pos
        md     = self.md

        # R6: 12-channel grid
        grid = np.zeros((GRID_CHANNELS, 16, 16), dtype=np.float32)

        # Channels 0-9: semantic one-hot (S2: single fancy-index)
        grid[:10] = _ONE_HOT[:, md].reshape(10, 16, 16)

        # Promote finish channel 4->5 when all collected
        if self.n_collected == self.n_targets:
            grid[5] = grid[4]
            grid[4] = 0.0

        # BFS navigation heatmap in channel 9 (replaces key/switch semantic)
        # CRUMBLE-AWARE: prefer targets in current section (no crumble crossing).
        # This teaches the agent to clear one section before moving to the next.
        bfs = self.get_bfs()
        bfs_safe = self.get_bfs_safe()
        all_done = self.n_collected == self.n_targets
        tmask = (md == 44) if all_done else ((md == 19) | (md == 45))

        # Choose which BFS to use for gradient: safe first, fallback to normal
        safe_reachable = tmask & (bfs_safe >= 0)
        if safe_reachable.any():
            bfs_for_grad = bfs_safe   # guide within current section
            bfs_to_target = int(bfs_safe[safe_reachable].min())
        else:
            bfs_for_grad = bfs        # must cross crumble
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

        # R6: Channel 10 — agent position
        grid[10, py, px] = 1.0

        # R6: Channel 11 — visited mask
        grid[11] = self._visited.reshape(16, 16)

        # Inventory vector (R6: 8 features)
        n_rem     = self.n_targets - self.n_collected
        rem       = float(n_rem) / max(1.0, float(self.n_targets))
        dist_n    = min(1.0, bfs_to_target / 32.0)
        phase     = 1.0 if all_done else 0.0
        progress  = float(self.n_collected) / max(1.0, float(self.n_targets))
        has_key   = 1.0 if (self.key_gray + self.key_yellow + self.key_red) > 0 else 0.0
        visit_rat = float(np.sum(self._visited > 0)) / 256.0
        # R2: reachability ratio — what fraction of remaining targets is BFS-reachable
        reach_rat = (float(self._prev_reachable) / max(1.0, float(n_rem))
                     if n_rem > 0 else 1.0)
        steps_rem = 1.0 - float(self.step_count) / max(1.0, float(self.max_steps))

        inv = np.array([rem, dist_n, phase, progress, has_key, visit_rat,
                        reach_rat, steps_rem], dtype=np.float32)

        self._cached_grid = grid
        self._cached_inv  = inv
        self._obs_dirty   = False
        return grid, inv


# ── Network — R5: larger strided CNN ──────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    3 strided convolutions: 16x16 -> 8x8 -> 4x4 -> 2x2.
    Wider channels (64/128/128) => flat=512 for better planning capacity.
    """
    def __init__(self, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        # 16x16 -> 8x8 -> 4x4 -> 2x2
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 64, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(64,           128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128,          128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Flatten(),  # -> 128*2*2 = 512
        )
        flat  = 128 * 2 * 2  # 512
        merge = flat + INV_FEATURES  # 520

        self.fc = nn.Sequential(
            nn.Linear(merge, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        self.val = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, n_actions))

    def forward(self, g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x   = self.fc(torch.cat([self.conv(g), v], dim=1))
        val = self.val(x)
        adv = self.adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ── Replay buffer — pinned memory + async transfer (S3) ───────────────────────
class ReplayBuffer:
    def __init__(self, cap: int = 150_000) -> None:
        self._cap  = cap
        self._ptr  = 0
        self._size = 0
        pin = {"pin_memory": True} if torch.cuda.is_available() else {}
        self._sg = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16, **pin)
        self._sv = torch.zeros((cap, INV_FEATURES),          dtype=torch.float32, **pin)
        self._a  = torch.zeros(cap,                          dtype=torch.int64,   **pin)
        self._r  = torch.zeros(cap,                          dtype=torch.float32, **pin)
        self._ng = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16, **pin)
        self._nv = torch.zeros((cap, INV_FEATURES),          dtype=torch.float32, **pin)
        self._d  = torch.zeros(cap,                          dtype=torch.float32, **pin)
        self._t  = torch.zeros(cap,                          dtype=torch.int64,   **pin)
        self._tm = torch.zeros(cap,                          dtype=torch.float32, **pin)

    def __len__(self) -> int:
        return self._size

    def push(self, sg: np.ndarray, sv: np.ndarray, a: int, r: float,
             ng: np.ndarray, nv: np.ndarray, done: bool,
             teacher: Optional[int] = None) -> None:
        i   = self._ptr
        r   = float(np.clip(r, -60.0, 60.0))
        self._sg[i] = torch.from_numpy(sg).to(torch.float16)
        self._sv[i] = torch.from_numpy(sv)
        self._a[i]  = a
        self._r[i]  = r
        self._ng[i] = torch.from_numpy(ng).to(torch.float16)
        self._nv[i] = torch.from_numpy(nv)
        self._d[i]  = float(done)
        self._t[i]  = int(teacher if teacher is not None else 0)
        self._tm[i] = 1.0 if teacher is not None else 0.0
        self._ptr   = (i + 1) % self._cap
        self._size  = min(self._size + 1, self._cap)

    def sample(self, n: int, device: torch.device) -> tuple:
        n_uni   = int(n * 0.7)
        n_rec   = n - n_uni
        idx_uni = torch.randint(0, self._size, (n_uni,))
        win     = max(n_rec, min(self._size, 10_000))
        start   = (self._ptr - win) % self._cap
        if start + win <= self._cap:
            pool = torch.arange(start, start + win)
        else:
            pool = torch.cat([torch.arange(start, self._cap),
                              torch.arange(0, (start + win) % self._cap)])
        pool    = pool[pool < self._size]
        if len(pool) < n_rec:
            pool = torch.randint(0, self._size, (n_rec,))
        idx_rec = pool[torch.randint(0, len(pool), (n_rec,))]
        idx     = torch.cat([idx_uni, idx_rec])
        nb      = {"non_blocking": True}
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
        )


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, device: torch.device, lr: float = 3e-4,
                 gamma: float = 0.99, batch_size: int = 512,
                 target_update: int = 500, teacher_weight: float = 0.25) -> None:
        self.device      = device
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.teacher_weight = teacher_weight
        self.epsilon     = 1.0
        self.total_steps = 0
        self.policy      = DuelingDQN().to(device)
        self.target      = DuelingDQN().to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt         = optim.Adam(self.policy.parameters(), lr=lr,
                                      eps=1e-4, weight_decay=1e-5)
        self.replay      = ReplayBuffer()
        self._target_upd = target_update
        self.use_amp     = device.type == "cuda"
        self.scaler      = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.policy.train()  # S4: stay in train mode permanently

    def act(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        g = torch.from_numpy(grid).unsqueeze(0).to(self.device, non_blocking=True)
        v = torch.from_numpy(inv).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            return int(self.policy(g, v).argmax(1).item())

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        sg, sv, a, r, ng, nv, d, t, tm = self.replay.sample(self.batch_size, self.device)

        def _loss() -> torch.Tensor:
            q_all = self.policy(sg, sv)
            qp = q_all.gather(1, a).squeeze(1)
            with torch.no_grad():
                best_a = self.policy(ng, nv).argmax(1, keepdim=True)
                qn     = self.target(ng, nv).gather(1, best_a).squeeze(1)
                tgt    = (r + (1.0 - d) * self.gamma * qn).clamp(-70.0, 120.0)
            td_loss = F.smooth_l1_loss(qp, tgt)
            if float(tm.sum().item()) > 0.0:
                teacher_loss = F.cross_entropy(q_all, t, reduction="none")
                teacher_loss = (teacher_loss * tm).sum() / tm.sum().clamp_min(1.0)
                return td_loss + self.teacher_weight * teacher_loss
            return td_loss

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss = _loss()
            if not torch.isfinite(loss): return None
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss = _loss()
            if not torch.isfinite(loss): return None
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

        self.total_steps += 1
        if self.total_steps % self._target_upd == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def save(self, path: Path, level: int, map_kind: str,
             extra: Optional[Dict] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policy":      self.policy.state_dict(),
            "target":      self.target.state_dict(),
            "optim":       self.opt.state_dict(),
            "epsilon":     self.epsilon,
            "total_steps": self.total_steps,
            "level":       level,
            "map_kind":    map_kind,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f"  [saved] {path.name}")

    def load(self, path: Path) -> Dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck.get("target", ck["policy"]))
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
        return {"level": ck.get("level",-1), "map_kind": ck.get("map_kind","normal"),
                "total_eps": total_ep, "best_sr": float(ck.get("best_sr", 0.0))}


# ── Level parsing helper ──────────────────────────────────────────────────────
def _parse_levels(s: str) -> List[int]:
    """Parse level specification: '3-25', '3,5,7', or '4'."""
    levels: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            levels.extend(range(int(lo), int(hi) + 1))
        else:
            levels.append(int(part))
    return sorted(set(levels))


def _bfs_teacher_action(env: BobbyEnv) -> Optional[int]:
    """Return a BFS-guided action when a reachable target exists.

    Returns None when no target is reachable from the current state.
    """
    bfs = env.get_bfs()
    bfs_safe = env.get_bfs_safe()
    px, py = env.pos

    all_done = env.n_collected == env.n_targets
    tmask = (env.md == 44) if all_done else ((env.md == 19) | (env.md == 45))
    if not tmask.any():
        return None

    safe_reachable = tmask & (bfs_safe >= 0)
    if safe_reachable.any():
        use_bfs = bfs_safe
    else:
        normal_reachable = tmask & (bfs >= 0)
        if not normal_reachable.any():
            return None
        use_bfs = bfs

    best_action: Optional[int] = None
    best_dist: Optional[int] = None
    for action, (dx, dy) in enumerate(BobbyEnv.ACTION_DELTA):
        if not env._can_move(px, py, dx, dy):
            continue
        nx, ny = px + dx, py + dy
        dist = int(use_bfs[nx + ny * 16])
        if dist < 0:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_action = action

    return best_action


def _bfs_best_action(env: BobbyEnv) -> int:
    """Pick the action that moves the agent one step closer to the nearest target.
    Uses crumble-aware BFS (collects current section first).
    Falls back to random if no clear path exists.
    Fully generic — works on any map, no hardcoding.
    """
    teacher_action = _bfs_teacher_action(env)
    if teacher_action is not None:
        return teacher_action

    # Fallback: random valid action
    px, py = env.pos
    valid_actions = [
        action
        for action, (dx, dy) in enumerate(BobbyEnv.ACTION_DELTA)
        if env._can_move(px, py, dx, dy)
    ]
    return random.choice(valid_actions) if valid_actions else random.randint(0, N_ACTIONS - 1)


# ── Training loop ──────────────────────────────────────────────────────────────
def train(
    map_kind:      str   = "normal",
    level:         int   = 9,
    levels:        Optional[List[int]] = None,
    n_episodes:    int   = 4000,
    max_steps:     Optional[int] = None,
    lr:            float = 3e-4,
    batch_size:    int   = 512,
    gamma:         float = 0.99,
    eps_start:     float = 1.0,
    eps_min:       float = 0.08,
    eps_decay:     float = 0.998,     # R4: ~1500 eps to reach eps_min
    warmup_eps:    int   = 50,        # R4: more random exploration first
    n_envs:        int   = 8,
    grad_every:    int   = 4,
    report_every:  int   = 100,
    save_every:    int   = 500,
    target_sr:     float = 0.90,
    target_cr:     float = 0.90,
    model_path:    str   = "",
    ckpt_dir:      str   = "dqn_checkpoints",
    resume:        bool  = False,
    seed:          int   = 42,
    target_update: int   = 500,
) -> None:

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    # Resolve level pool for single-level or multi-level training
    level_pool = levels if levels and len(levels) > 0 else [level]
    multi = len(level_pool) > 1

    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    if not model_path:
        if multi:
            model_path = str(ckpt / f"dqn_{map_kind}{min(level_pool)}_{max(level_pool)}.pt")
        else:
            model_path = str(ckpt / f"dqn_{map_kind}{level_pool[0]}.pt")
    out = Path(model_path)

    # Probe map for metadata
    probe = BobbyEnv(map_kind, level_pool[0], max_steps)
    probe.reset()
    eff_max = probe.max_steps
    if multi:
        print(f"Multi-level training: {len(level_pool)} levels "
              f"({min(level_pool)}-{max(level_pool)}) | n_envs={n_envs}")
    else:
        print(f"Level {map_kind}-{level_pool[0]} | targets={probe.n_targets} "
              f"| max_steps={eff_max} | n_envs={n_envs}")

    # S7: N_ENVS parallel environments (each on a random level from pool)
    envs: List[BobbyEnv] = []
    for _ in range(n_envs):
        lvl = random.choice(level_pool)
        envs.append(BobbyEnv(map_kind, lvl, max_steps))
    for e in envs:
        e.reset()

    agent = DQNAgent(device, lr=lr, gamma=gamma,
                     batch_size=batch_size, target_update=target_update)
    agent.epsilon = eps_start

    start_eps = 0
    if resume and out.exists():
        ck_info = agent.load(out)
        start_eps = ck_info.get("total_eps", 0)

    # Metrics
    completed_cr:  List[float] = []
    completed_win: List[float] = []
    losses_buf:    List[float] = []

    total_eps  = start_eps
    env_steps  = 0
    best_sr    = 0.0
    early_win  = 0
    stag_count = 0          # stagnation detector: 0% success + high collection
    t0         = time.time()
    target_eps = total_eps + n_episodes

    tag = f"L{min(level_pool)}-{max(level_pool)}" if multi else f"L{level_pool[0]}"
    print(f"\n=== Training {tag} for {n_episodes} episodes "
          f"(ep {total_eps}->{target_eps}) ===")
    print(f"eps: {agent.epsilon:.2f} -> {eps_min:.2f} (decay={eps_decay}/ep, "
          f"~{int(-1/np.log(eps_decay))} eps to min) | "
          f"warmup={warmup_eps} | grad_every={grad_every} | batch={batch_size}")

    last_report_ep = -1
    last_save_ep   = -1

    while total_eps < target_eps:
        # ── Collect one step across all envs (S7) ────────────────────────
        for i, env in enumerate(envs):
            g, v   = env.get_obs()

            # BFS-guided exploration: during warmup and epsilon-random,
            # 50% actions follow BFS pathfinding toward nearest target.
            # This gives the replay buffer good trajectories to learn from
            # while maintaining exploration randomness.
            if total_eps < warmup_eps:
                # Warmup: mostly BFS-guided (70%) to seed replay with useful data
                if random.random() < 0.7:
                    action = _bfs_best_action(env)
                else:
                    action = random.randint(0, N_ACTIONS - 1)
            elif random.random() < agent.epsilon:
                # Epsilon exploration: 50% BFS-guided, 50% random
                if random.random() < 0.5:
                    action = _bfs_best_action(env)
                else:
                    action = random.randint(0, N_ACTIONS - 1)
            else:
                # Exploitation: use policy network
                action = agent.act(g, v)

            teacher_action = _bfs_teacher_action(env)
            r, done, won = env.step(action)
            ng, nv = env.get_obs()
            agent.replay.push(g, v, action, r, ng, nv, done, teacher=teacher_action)

            if done:
                cr = env.n_collected / max(1, env.n_targets)
                completed_cr.append(cr)
                completed_win.append(1.0 if env.finished else 0.0)
                total_eps += 1

                # R4: Epsilon decay PER COMPLETED EPISODE (not per env-step!)
                if total_eps > warmup_eps:
                    agent.epsilon = max(eps_min, agent.epsilon * eps_decay)

                # Multi-level: assign random level from pool on reset
                if multi:
                    env.set_level(random.choice(level_pool))
                env.reset()

        env_steps += n_envs

        # ── Warmup: no gradient updates until warmup_eps completed episodes
        if total_eps < warmup_eps:
            continue

        # ── Gradient update (S8) ─────────────────────────────────────────
        if env_steps % grad_every == 0:
            loss = agent.update()
            if loss is not None:
                losses_buf.append(loss)

        # ── Reporting ────────────────────────────────────────────────────
        if total_eps > 0 and total_eps % report_every == 0 and total_eps != last_report_ep:
            last_report_ep = total_eps
            window = min(report_every, len(completed_win))
            sr  = float(np.mean(completed_win[-window:]))
            cr  = float(np.mean(completed_cr[-window:]))
            al  = float(np.mean(losses_buf[-200:])) if losses_buf else 0.0
            sps = env_steps / max(time.time() - t0, 1e-6)
            eta = max(0.0, (target_eps - total_eps) * eff_max / max(sps, 1) / 60)
            print(f"[{tag}] ep={total_eps:5d}/{target_eps} | "
                  f"collected={cr:5.1%} | success={sr:5.1%} | "
                  f"eps={agent.epsilon:.3f} | loss={al:.4f} | "
                  f"sps={sps:.0f} | ETA={eta:.1f}min")

            if sr > best_sr:
                best_sr = sr
                best_name = (f"dqn_{map_kind}{min(level_pool)}_{max(level_pool)}_best.pt"
                             if multi else f"dqn_{map_kind}{level_pool[0]}_best.pt")
                agent.save(ckpt / best_name, level_pool[0], map_kind,
                           extra={"total_eps": total_eps, "best_sr": best_sr})

            if sr >= target_sr and cr >= target_cr and total_eps >= 200:
                early_win += 1
                if early_win >= 3:
                    print(f"[{tag}] Early stop: sr={sr:.1%} cr={cr:.1%}")
                    agent.save(out, level_pool[0], map_kind,
                               extra={"total_eps": total_eps, "best_sr": best_sr})
                    return
            else:
                early_win = 0

            # ── Stagnation detection: high collection but zero success ────
            # If the agent collects well but never finishes, it's stuck on a
            # bad route. Temporarily boost epsilon to force path exploration.
            if sr < 0.01 and cr > 0.60 and total_eps > warmup_eps + 200:
                stag_count += 1
                if stag_count >= 3:  # 3 consecutive reports with 0% success
                    old_eps = agent.epsilon
                    agent.epsilon = max(agent.epsilon, 0.30)
                    print(f"  [!] Stagnation: eps {old_eps:.3f} -> {agent.epsilon:.3f}")
                    stag_count = 0
            else:
                stag_count = 0

        if total_eps > 0 and total_eps % save_every == 0 and total_eps != last_save_ep:
            last_save_ep = total_eps
            agent.save(out, level_pool[0], map_kind,
                       extra={"total_eps": total_eps, "best_sr": best_sr})

    agent.save(out, level_pool[0], map_kind,
               extra={"total_eps": total_eps, "best_sr": best_sr})
    if completed_win:
        window = min(100, len(completed_win))
        print(f"\nDone. Best success={best_sr:.1%} | "
              f"Final collected={np.mean(completed_cr[-window:]):.1%} | "
              f"Model -> {out}")


# ── Play / evaluation ──────────────────────────────────────────────────────────
def play(
    map_kind:   str = "normal",
    level:      int = 9,
    levels:     Optional[List[int]] = None,
    model_path: str = "",
    n_episodes: int = 20,
    ckpt_dir:   str = "dqn_checkpoints",
    max_steps:  Optional[int] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent(device)

    if not model_path:
        best = Path(ckpt_dir) / f"dqn_{map_kind}{level}_best.pt"
        last = Path(ckpt_dir) / f"dqn_{map_kind}{level}.pt"
        model_path = str(best if best.exists() else last)

    agent.load(Path(model_path))
    agent.epsilon = 0.0
    agent.policy.eval()

    eval_levels = levels if levels and len(levels) > 0 else [level]

    all_wins  = 0
    all_total = 0

    for lvl in eval_levels:
        env  = BobbyEnv(map_kind, lvl, max_steps)
        wins = 0
        if len(eval_levels) > 1:
            print(f"\n--- Level {lvl} ---")
        for ep in range(1, n_episodes + 1):
            env.reset()
            g, v  = env.get_obs()
            done  = False
            steps = 0
            while not done:
                a = agent.act(g, v)
                _, done, _ = env.step(a)
                g, v = env.get_obs()
                steps += 1
            wins += int(env.finished)
            print(f"Ep {ep:3d}: {'WIN ' if env.finished else 'FAIL'} | "
                  f"steps={steps:4d} | collected={env.n_collected}/{env.n_targets}")
        print(f"Level {lvl} success: {wins}/{n_episodes} = {wins/n_episodes:.1%}")
        all_wins  += wins
        all_total += n_episodes

    if len(eval_levels) > 1:
        print(f"\nOverall success: {all_wins}/{all_total} = {all_wins/all_total:.1%}")


# ── GUI Play with Pygame rendering ─────────────────────────────────────────────
def play_gui(
    map_kind:    str = "normal",
    level:       int = 9,
    model_path:  str = "",
    n_episodes:  int = 3,
    ckpt_dir:    str = "dqn_checkpoints",
    max_steps:   Optional[int] = None,
    fps:         int = 60,
) -> None:
    """Play the trained agent with full Pygame visual rendering.
    Bobby walks around, collects carrots, and finishes the level.
    """
    import pygame
    from pygame import Rect

    # ── Load agent ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent(device)

    if not model_path:
        best = Path(ckpt_dir) / f"dqn_{map_kind}{level}_best.pt"
        last = Path(ckpt_dir) / f"dqn_{map_kind}{level}.pt"
        local_best = Path(f"dqn_{map_kind}{level}_best.pt")
        local_last = Path(f"dqn_{map_kind}{level}.pt")
        if local_best.exists():      model_path = str(local_best)
        elif local_last.exists():    model_path = str(local_last)
        elif best.exists():          model_path = str(best)
        else:                        model_path = str(last)

    agent.load(Path(model_path))
    agent.epsilon = 0.0
    agent.policy.eval()
    print(f"Playing level {level} with model {model_path}")

    # ── Init Pygame ───────────────────────────────────────────────────────────
    pygame.init()
    window = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption(f"Bobby Carrot RL Agent ({map_kind}-{level})")
    clock  = pygame.time.Clock()
    assets = Assets()

    AI_MOVE_COOLDOWN_MS = 150  # ms between AI decisions

    for ep in range(1, n_episodes + 1):
        # ── Set up environment + visual Bobby ────────────────────────────────
        map_obj        = Map(map_kind, level)
        map_info_fresh = map_obj.load_map_info()
        map_info       = MapInfo(map_info_fresh.data.copy(),
                                 map_info_fresh.coord_start,
                                 map_info_fresh.carrot_total,
                                 map_info_fresh.egg_total)

        frame       = 0
        timer_start = pygame.time.get_ticks()
        bobby       = Bobby(frame, timer_start, map_info.coord_start)

        env = BobbyEnv(map_kind, level, max_steps)
        env.reset()
        g, v = env.get_obs()

        action_map = {0: State.Left, 1: State.Right, 2: State.Up, 3: State.Down}

        running    = True
        done       = False
        steps_taken = 0
        result_msg  = ""

        print(f"\n--- Episode {ep}/{n_episodes} ---")

        while running:
            now_ms = pygame.time.get_ticks()

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

            # ── AI decision ───────────────────────────────────────────────────
            if (not bobby.is_walking()
                and not bobby.dead
                and not done
                and bobby.state not in (State.FadeIn, State.FadeOut, State.Death)):
                if now_ms - getattr(bobby, '_last_ai_ms', 0) >= AI_MOVE_COOLDOWN_MS:
                    action = agent.act(g, v)
                    _, env_done, env_won = env.step(action)
                    g, v = env.get_obs()
                    steps_taken += 1

                    # Apply to visual Bobby (updates coord_dest, triggers walking)
                    state_opt = action_map[action]
                    bobby.last_action_time = now_ms
                    bobby._last_ai_ms = now_ms
                    bobby.update_state(state_opt, frame, map_info.data)

            # ── Sound triggers ────────────────────────────────────────────────
            if bobby.carrot_count != getattr(bobby, '_prev_carrots', 0):
                if assets.snd_carrot:
                    assets.snd_carrot.play()
                else:
                    assets._beep()
            if bobby.dead and not getattr(bobby, '_prev_dead', False):
                assets._beep()
            bobby._prev_carrots = bobby.carrot_count
            bobby._prev_dead    = bobby.dead

            # ── Win / death ───────────────────────────────────────────────────
            if bobby.dead:
                result_msg = f"DIED after {steps_taken} steps | collected={env.n_collected}/{env.n_targets}"
                done = True
            elif (bobby.is_finished(map_info)
                  and map_info.data[bobby.coord_src[0] + bobby.coord_src[1]*16] == 44):
                if bobby.faded_out:
                    result_msg = f"WON in {steps_taken} steps | collected={env.n_collected}/{env.n_targets}"
                    done = True
                elif bobby.state != State.FadeOut:
                    bobby.start_frame = frame
                    bobby.state       = State.FadeOut

            # ── Stall failsafe (env ran out of steps but visual didn't) ───────
            if env.step_count >= env.max_steps and not done:
                result_msg = f"TIMEOUT after {steps_taken} steps | collected={env.n_collected}/{env.n_targets}"
                done = True

            if done and bobby.state not in (State.FadeOut,):
                # Let the result print and wait a moment before next episode
                print(f"  Ep {ep}: {result_msg}")
                pygame.time.wait(800)
                break
            if done and bobby.faded_out:
                print(f"  Ep {ep}: {result_msg}")
                pygame.time.wait(800)
                break

            # ── Drawing ───────────────────────────────────────────────────────
            screen = pygame.display.get_surface()
            screen.fill((0, 0, 0))

            # Camera
            step = (frame - bobby.start_frame)
            x0 = bobby.coord_src[0] * 32
            y0 = bobby.coord_src[1] * 32
            x1 = bobby.coord_dest[0] * 32
            y1 = bobby.coord_dest[1] * 32
            if bobby.state == State.Death:
                cam_x = (x1 - x0) * 6 // 8 + x0 - (VIEW_WIDTH_POINTS // 2) * 32
                cam_y = (y1 - y0) * 6 // 8 + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
            else:
                cam_x = (x1 - x0) * step // (8 * FRAMES_PER_STEP) + x0 - (VIEW_WIDTH_POINTS // 2) * 32
                cam_y = (y1 - y0) * step // (8 * FRAMES_PER_STEP) + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
            cam_x = max(0, min(cam_x + 16, WIDTH_POINTS_DELTA * 32))
            cam_y = max(0, min(cam_y + 16, HEIGHT_POINTS_DELTA * 32))
            x_right_offset = WIDTH_POINTS_DELTA * 32 - cam_x
            y_offset       = cam_y

            # Tiles
            for x in range(WIDTH_POINTS):
                for y in range(HEIGHT_POINTS):
                    item = map_info.data[x + y * 16]
                    texture = assets.tileset
                    if item == 44 and bobby.is_finished(map_info):
                        texture = assets.tile_finish
                    elif item == 40: texture = assets.tile_conveyor_left
                    elif item == 41: texture = assets.tile_conveyor_right
                    elif item == 42: texture = assets.tile_conveyor_up
                    elif item == 43: texture = assets.tile_conveyor_down
                    if (item == 44 and bobby.is_finished(map_info)) or 40 <= item <= 43:
                        src = Rect(32 * ((frame // (FRAMES // 10)) % 4), 0, 32, 32)
                    else:
                        src = Rect(32 * (item % 8), 32 * (item // 8), 32, 32)
                    dest = Rect(x * 32 - cam_x, y * 32 - cam_y, 32, 32)
                    screen.blit(texture, dest, src)

            # Bobby sprite
            bobby_src, bobby_dest = bobby.update_texture_position(frame, map_info.data)
            bobby_tex = {
                State.Idle: assets.bobby_idle,
                State.Death: assets.bobby_death,
                State.FadeIn: assets.bobby_fade,
                State.FadeOut: assets.bobby_fade,
                State.Left: assets.bobby_left,
                State.Right: assets.bobby_right,
                State.Up: assets.bobby_up,
                State.Down: assets.bobby_down,
            }[bobby.state]
            bobby_dest = bobby_dest.move(-cam_x, -cam_y)
            screen.blit(bobby_tex, bobby_dest, bobby_src)

            # HUD counter
            if map_info.carrot_total > 0:
                icon_rect = Rect(0, 0, 46, 44)
                num_left  = map_info.carrot_total - bobby.carrot_count
                icon_width = 46
            else:
                icon_rect = Rect(46, 0, 34, 44)
                num_left  = map_info.egg_total - bobby.egg_count
                icon_width = 34
            screen.blit(assets.hud,
                        (32*16 - (icon_width+4) - x_right_offset, 4 + y_offset),
                        icon_rect)
            num_10 = num_left // 10
            num_01 = num_left % 10
            screen.blit(assets.numbers,
                        (32*16 - (icon_width+4) - 2 - 12 - x_right_offset, 4+14+y_offset),
                        Rect(num_01*12, 0, 12, 18))
            screen.blit(assets.numbers,
                        (32*16 - (icon_width+4) - 2 - 12*2 - 1 - x_right_offset, 4+14+y_offset),
                        Rect(num_10*12, 0, 12, 18))

            pygame.display.flip()
            frame += 1
            clock.tick(fps)

    pygame.quit()
    print("\nDone.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bobby Carrot Reachability-Aware DQN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--play",          action="store_true")
    p.add_argument("--play-gui",      action="store_true",
                   help="Play with full Pygame GUI rendering (3 episodes)")
    p.add_argument("--map-kind",      default="normal", choices=["normal","egg"])
    p.add_argument("--level",         type=int,   default=9)
    p.add_argument("--levels",        default="",
                   help="Level range for multi-level training/testing "
                        "(e.g., '3-25' or '3,5,7'). Overrides --level.")
    p.add_argument("--model-path",    default="")
    p.add_argument("--ckpt-dir",      default="dqn_checkpoints")
    p.add_argument("--episodes",      type=int,   default=4000)
    p.add_argument("--max-steps",     type=int,   default=None)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--eps-start",     type=float, default=1.0)
    p.add_argument("--eps-min",       type=float, default=0.08)
    p.add_argument("--eps-decay",     type=float, default=0.998)
    p.add_argument("--warmup-eps",    type=int,   default=50)
    p.add_argument("--n-envs",        type=int,   default=8)
    p.add_argument("--grad-every",    type=int,   default=4)
    p.add_argument("--target-update", type=int,   default=500)
    p.add_argument("--target-sr",     type=float, default=0.90)
    p.add_argument("--target-cr",     type=float, default=0.90)
    p.add_argument("--report-every",  type=int,   default=100)
    p.add_argument("--save-every",    type=int,   default=500)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--play-episodes", type=int,   default=20)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    parsed_levels = _parse_levels(args.levels) if args.levels else None

    if args.play_gui:
        play_gui(map_kind=args.map_kind, level=args.level,
                 model_path=args.model_path, n_episodes=3,
                 ckpt_dir=args.ckpt_dir, max_steps=args.max_steps)
    elif args.play:
        play(map_kind=args.map_kind, level=args.level,
             levels=parsed_levels,
             model_path=args.model_path, n_episodes=args.play_episodes,
             ckpt_dir=args.ckpt_dir, max_steps=args.max_steps)
    else:
        train(map_kind=args.map_kind, level=args.level,
              levels=parsed_levels,
              n_episodes=args.episodes, max_steps=args.max_steps,
              lr=args.lr, batch_size=args.batch_size, gamma=args.gamma,
              eps_start=args.eps_start, eps_min=args.eps_min,
              eps_decay=args.eps_decay, warmup_eps=args.warmup_eps,
              n_envs=args.n_envs, grad_every=args.grad_every,
              report_every=args.report_every, save_every=args.save_every,
              target_sr=args.target_sr, target_cr=args.target_cr,
              model_path=args.model_path, ckpt_dir=args.ckpt_dir,
              resume=args.resume, seed=args.seed,
              target_update=args.target_update)


if __name__ == "__main__":
    main()
