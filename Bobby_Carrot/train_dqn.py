from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
# Grid / network constants
# ============================================================

GRID_SIZE = 16

# Channel map (13 total):
#  0  = wall/impassable (tile < 18)
#  1  = carrot (19)
#  2  = egg (45)
#  3  = finish ACTIVE (44, all collected)
#  4  = finish INACTIVE (44, targets remain)
#  5  = hazard/death (31, 46)
#  6  = key (32, 34, 36)
#  7  = locked door (33, 35, 37)
#  8  = crumble tile (30)  ← crumble now has its own channel
#  9  = conveyor (40-43)   ← conveyor has its own channel
#  10 = floor / other traversable (18, 20, 22, 23, 38, 39 …)
#  11 = agent position
#  12 = all_collected phase flag (whole plane = 1 when done)
GRID_CHANNELS = 13
INV_FEATURES  = 5   # [key_gray, key_yellow, key_red, remaining_norm, manhattan_norm]


# ============================================================
# Per-level config — derived from .blm map analysis
# ============================================================
#
# Why each value:
#
# L4:  35 carrots, 5 crumbles adjacent to carrots, finish at (7,10).
#      35*20+200 = 900 minimum → padded to 1200.
#      distance_scale=0.8: the L4 training log showed avg_reward climbing to 600+
#      while success stayed 0% — the agent was earning reward purely from walking
#      toward carrots without collecting. scale=2.0 was too high. 0.8 fixes this.
#
# L5:  19 carrots, 7 crumbles, tight symmetrical maze, finish far from start.
#      700 steps fine. Moderate scale=1.0.
#
# L6:  24 carrots, 11 crumbles (highest crumble density so far).
#      Many crumbles adjacent to carrots — crumble_step_penalty critical here.
#      900 steps, scale=1.0.
#
# L7:  34 carrots, 24 crumbles — most path-destructive map.
#      24 crumbles means very easy to permanently block needed paths.
#      1400 steps, heavy crumble penalty, scale=0.8.
#
# L8:  8 carrots only, 10 crumbles, 4 conveyors (LEFT/RIGHT symmetric pair).
#      Fewest carrots — 600 steps ample. Conveyors force specific entry directions.
#      Completion bonus boosted because signal is very sparse.
#
# L9:  27 carrots, 2 crumbles, 5 conveyors (RIGHT chain + DOWN chain).
#      Conveyors auto-move agent — may carry it off target, needs more steps. 900.
#
# L10: 17 carrots, 6 crumbles, 3 conveyors (1×DOWN + 2×UP).
#      Conveyors push agent vertically. finish far above start. 800 steps fine.

LEVEL_CONFIG: Dict[int, Dict] = {
    1:  {"max_steps": 600,  "episodes": 8000,  "distance_scale": 1.5, "post_penalty": -0.5},
    2:  {"max_steps": 700,  "episodes": 8000,  "distance_scale": 1.5, "post_penalty": -0.5},
    3:  {"max_steps": 800,  "episodes": 8000,  "distance_scale": 1.2, "post_penalty": -0.5},
    4:  {"max_steps": 1200, "episodes": 10000, "distance_scale": 0.8, "post_penalty": -0.8},
    5:  {"max_steps": 700,  "episodes": 8000,  "distance_scale": 1.0, "post_penalty": -0.6},
    6:  {"max_steps": 900,  "episodes": 8000,  "distance_scale": 1.0, "post_penalty": -0.6},
    7:  {"max_steps": 1400, "episodes": 12000, "distance_scale": 0.8, "post_penalty": -0.8},
    8:  {"max_steps": 600,  "episodes": 8000,  "distance_scale": 1.2, "post_penalty": -0.5},
    9:  {"max_steps": 900,  "episodes": 8000,  "distance_scale": 1.0, "post_penalty": -0.6},
    10: {"max_steps": 800,  "episodes": 8000,  "distance_scale": 1.0, "post_penalty": -0.6},
}


# ============================================================
# DQNConfig — global defaults
# ============================================================

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 150_000
    min_replay_size: int = 3000
    target_update_steps: int = 2000
    train_every_steps: int = 2
    # Exploration
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9994
    # Reward shaping (level-specific values in LEVEL_CONFIG)
    completion_bonus: float = 300.0
    death_penalty: float = -80.0
    carrot_bonus: float = 15.0
    crumble_step_penalty: float = -5.0   # per crumble step adjacent to uncollected carrot
    invalid_move_penalty: float = -0.5
    terminal_oversample: int = 4
    # Misc
    report_every: int = 100
    save_every: int = 500
    seed: int = 42
    observation_mode: str = "full"       # single source of truth used everywhere


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        grid: np.ndarray, inv: np.ndarray,
        action: int, reward: float,
        next_grid: np.ndarray, next_inv: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((
            grid.astype(np.float32, copy=False),
            inv.astype(np.float32, copy=False),
            int(action), float(reward),
            next_grid.astype(np.float32, copy=False),
            next_inv.astype(np.float32, copy=False),
            bool(done),
        ))

    def sample(self, n: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self._buf, n)
        g, v, a, r, ng, nv, d = zip(*batch)
        return (
            np.stack(g), np.stack(v),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ng), np.stack(nv),
            np.array(d, dtype=np.float32),
        )


# ============================================================
# Dueling DDQN network
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
        self.shared = nn.Sequential(nn.Linear(merged, 512), nn.ReLU())
        self.value_stream     = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        feat   = self.conv(grid)
        shared = self.shared(torch.cat([feat, inv], dim=1))
        v = self.value_stream(shared)
        a = self.advantage_stream(shared)
        return v + a - a.mean(dim=1, keepdim=True)


# ============================================================
# Observation encoding
# ============================================================

def _nearest_manhattan(positions: List[Tuple[int, int]], px: int, py: int) -> float:
    if not positions:
        return 0.0
    return min(1.0, min(abs(x - px) + abs(y - py) for x, y in positions) / (GRID_SIZE * 2))


def _nearest_finish_manhattan(env: BobbyCarrotEnv) -> float:
    assert env.map_info is not None and env.bobby is not None
    px, py = env.bobby.coord_src
    for i, tile in enumerate(env.map_info.data):
        if tile == 44:
            fx, fy = i % GRID_SIZE, i // GRID_SIZE
            return min(1.0, (abs(fx - px) + abs(fy - py)) / (GRID_SIZE * 2))
    return 0.0


def _semantic_channels(env: BobbyCarrotEnv) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode map state into (grid, inv) tensors.
    Consistent between training and play — both use this function exclusively.
    """
    assert env.map_info is not None and env.bobby is not None

    grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    all_collected = env.bobby.is_finished(env.map_info)
    uncollected: List[Tuple[int, int]] = []

    for idx, tile in enumerate(env.map_info.data):
        x, y = idx % GRID_SIZE, idx // GRID_SIZE
        if tile < 18:
            c = 0
        elif tile == 19:
            c = 1
            uncollected.append((x, y))
        elif tile == 45:
            c = 2
            uncollected.append((x, y))
        elif tile == 44:
            c = 3 if all_collected else 4
        elif tile in {31, 46}:
            c = 5
        elif tile in {32, 34, 36}:
            c = 6
        elif tile in {33, 35, 37}:
            c = 7
        elif tile == 30:
            c = 8   # crumble — dedicated channel
        elif tile in {40, 41, 42, 43}:
            c = 9   # conveyor — dedicated channel
        else:
            c = 10  # floor / other
        grid[c, y, x] = 1.0

    px, py = env.bobby.coord_src
    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
        grid[11, py, px] = 1.0
    if all_collected:
        grid[12, :, :] = 1.0

    remaining = (
        (env.map_info.carrot_total - env.bobby.carrot_count) +
        (env.map_info.egg_total    - env.bobby.egg_count)
    )
    denom = max(1, env.map_info.carrot_total + env.map_info.egg_total)
    remaining_norm = min(1.0, remaining / denom)

    # Phase 1: distance to nearest uncollected target
    # Phase 2: distance to finish tile (inv[4] meaning changes with phase)
    if not all_collected:
        manhattan_norm = _nearest_manhattan(uncollected, px, py)
    else:
        manhattan_norm = _nearest_finish_manhattan(env)

    inv = np.array([
        float(env.bobby.key_gray   > 0),
        float(env.bobby.key_yellow > 0),
        float(env.bobby.key_red    > 0),
        remaining_norm,
        manhattan_norm,
    ], dtype=np.float32)

    return grid, inv


# ============================================================
# Reward shaping
# ============================================================

def _crumble_adjacent_to_carrot(env: BobbyCarrotEnv) -> bool:
    """
    Return True if agent just stepped onto a tile that is now tile==31 (death),
    meaning it was a crumble tile (30) that got consumed, AND at least one
    neighbour is still an uncollected carrot (19).
    This detects the exact scenario where the agent destroys access to a carrot.
    """
    assert env.map_info is not None and env.bobby is not None
    px, py = env.bobby.coord_src
    if env.map_info.data[px + py * GRID_SIZE] != 31:
        return False
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = px + dx, py + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if env.map_info.data[nx + ny * GRID_SIZE] == 19:
                return True
    return False


def _shape_reward(
    raw_reward: float,
    info: Dict[str, object],
    cfg: DQNConfig,
    level_cfg: Dict,
    prev_inv: np.ndarray,
    curr_inv: np.ndarray,
    all_collected_before: bool,
    env: BobbyCarrotEnv,
) -> float:
    reward = float(raw_reward)

    distance_scale = level_cfg["distance_scale"]
    post_penalty   = level_cfg["post_penalty"]

    if bool(info.get("invalid_move", False)):
        reward += cfg.invalid_move_penalty

    if bool(info.get("dead", False)):
        reward += cfg.death_penalty

    carrot_delta = int(info.get("collected_carrot", 0))
    if carrot_delta > 0:
        reward += cfg.carrot_bonus * carrot_delta

    # Completion bonus fires immediately on level_completed — not gated by fade animation
    if bool(info.get("level_completed", False)):
        reward += cfg.completion_bonus

    all_collected_now = bool(info.get("all_collected", False))
    level_done        = bool(info.get("level_completed", False))

    # Distance shaping:
    # prev_inv[4] / curr_inv[4] = manhattan_norm (to nearest carrot in p1, finish in p2)
    # positive dist_delta = moved closer = positive reward
    dist_delta = float(prev_inv[4]) - float(curr_inv[4])

    if not all_collected_now:
        # Phase 1: shape toward collecting all targets
        reward += distance_scale * dist_delta
    elif not level_done:
        # Phase 2: shape toward finish tile (inv[4] now = finish distance)
        reward += distance_scale * dist_delta
        reward += post_penalty   # discourage wandering after collection

    # Crumble-adjacent-to-carrot penalty: punish destroying a path to an uncollected carrot
    if _crumble_adjacent_to_carrot(env):
        reward += cfg.crumble_step_penalty

    return reward


# ============================================================
# DQN Agent — Dueling + Double DQN
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

    def select_action(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            g = torch.from_numpy(grid).unsqueeze(0).to(self.device)
            v = torch.from_numpy(inv).unsqueeze(0).to(self.device)
            return int(self.policy_net(g, v).argmax(1).item())

    def optimize_step(self) -> Optional[float]:
        if len(self.replay) < self.cfg.min_replay_size:
            return None
        if self.total_steps % self.cfg.train_every_steps != 0:
            return None

        s_g, s_v, acts, rwds, ns_g, ns_v, dones = self.replay.sample(self.cfg.batch_size)

        sg_t  = torch.from_numpy(s_g).to(self.device)
        sv_t  = torch.from_numpy(s_v).to(self.device)
        a_t   = torch.from_numpy(acts).unsqueeze(1).to(self.device)
        r_t   = torch.from_numpy(rwds).to(self.device)
        nsg_t = torch.from_numpy(ns_g).to(self.device)
        nsv_t = torch.from_numpy(ns_v).to(self.device)
        d_t   = torch.from_numpy(dones).to(self.device)

        q_pred = self.policy_net(sg_t, sv_t).gather(1, a_t).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy selects action, target evaluates it
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
        map_kind=map_kind,
        map_number=level,
        observation_mode=cfg.observation_mode,
        local_view_size=3,
        include_inventory=True,
        headless=True,
        max_steps=max_steps,
    )


def _get_level_cfg(level: int) -> Dict:
    return LEVEL_CONFIG.get(level, {
        "max_steps": 800, "episodes": 8000,
        "distance_scale": 1.0, "post_penalty": -0.6,
    })


# ============================================================
# Training loop
# ============================================================

def train_one_level(
    agent: DQNAgent,
    level: int,
    cfg: DQNConfig,
    map_kind: str,
    ckpt_path: Optional[Path] = None,
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

    print(f"  Level config — max_steps={max_steps} | episodes={n_episodes} | "
          f"distance_scale={level_cfg['distance_scale']} | "
          f"post_penalty={level_cfg['post_penalty']}")

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
                prev_inv, next_inv,
                all_collected_before, env,
            )

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            if bool(info.get("level_completed", False)) and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps += 1
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
            n     = cfg.report_every
            avg_r = float(np.mean(reward_hist[-n:]))
            avg_s = float(np.mean(success_hist[-n:]))
            avg_c = float(np.mean(collected_hist[-n:]))
            avg_l = float(np.mean(loss_win)) if loss_win else 0.0
            print(
                f"[L{level}] ep={episode:5d} | "
                f"avg_reward={avg_r:8.2f} | "
                f"all_collected={avg_c:6.2%} | "
                f"success={avg_s:6.2%} | "
                f"eps={agent.epsilon:.3f} | "
                f"replay={len(agent.replay):6d} | "
                f"loss={avg_l:.4f}"
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
    map_kind: str,
    map_number: int,
    episodes: int = 5,
    render: bool = False,
    render_fps: float = 5.0,
    cfg: Optional[DQNConfig] = None,
) -> None:
    if cfg is None:
        cfg = DQNConfig()

    level_cfg = _get_level_cfg(map_number)
    max_steps = level_cfg["max_steps"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=cfg.observation_mode,
        local_view_size=3,
        include_inventory=True,
        headless=not render,
        max_steps=max_steps,
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
            f"all_collected={bool(info.get('all_collected', False))} | "
            f"success={bool(info.get('level_completed', False))} | steps={steps}"
        )

    env.close()


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bobby Carrot DQN — map-aware trainer")
    p.add_argument("--play",             action="store_true")
    p.add_argument("--map-kind",         default="normal", choices=["normal", "egg"])
    p.add_argument("--map-number",       type=int, default=1)
    p.add_argument("--levels",           type=int, nargs="+", default=None)
    p.add_argument("--individual-levels",action="store_true")
    # Per-level overrides
    p.add_argument("--episodes-per-level", type=int, default=None,
                   help="Override episodes (default: from LEVEL_CONFIG)")
    p.add_argument("--max-steps",          type=int, default=None,
                   help="Override max_steps (default: from LEVEL_CONFIG)")
    # Optimiser
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--gamma",              type=float, default=0.99)
    # Exploration
    p.add_argument("--epsilon-start",      type=float, default=1.0)
    p.add_argument("--epsilon-min",        type=float, default=0.02)
    p.add_argument("--epsilon-decay",      type=float, default=0.9994)
    # Reward
    p.add_argument("--completion-bonus",     type=float, default=300.0)
    p.add_argument("--death-penalty",        type=float, default=-80.0)
    p.add_argument("--crumble-step-penalty", type=float, default=-5.0)
    p.add_argument("--invalid-move-penalty", type=float, default=-0.5)
    p.add_argument("--terminal-oversample",  type=int,   default=4)
    # Misc
    p.add_argument("--observation-mode",   default="full", choices=["full","local","compact"])
    p.add_argument("--report-every",       type=int,   default=100)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--model-path",         default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir",     default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    # Play
    p.add_argument("--play-episodes",      type=int,   default=5)
    p.add_argument("--no-render",          action="store_true")
    p.add_argument("--render-fps",         type=float, default=5.0)
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
            out   = ckpt_dir / f"dqn_level{lvl}_individual.pt"
            summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
            agent.save(out, level=lvl, map_kind=args.map_kind)
            print(f"Saved {out} | {summary}")
        return

    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
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
            f"all_collected={summary['all_collected_rate']:.2%} | "
            f"success={summary['success_rate']:.2%}"
        )

    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"\nFinal model saved to: {model_path}")


if __name__ == "__main__":
    _main()
