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

GRID_SIZE = 16
GRID_CHANNELS = 11  # wall, carrot, egg, goal_active, goal_inactive, hazard, key, door, empty, agent, all_collected_phase
INV_FEATURES = 5    # gray_key, yellow_key, red_key, remaining_bucket, manhattan_to_nearest_target


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    # Training schedule
    episodes_per_level: int = 8000
    max_steps: int = 800
    # Optimisation
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 150_000
    min_replay_size: int = 3000
    target_update_steps: int = 2000
    train_every_steps: int = 2
    # Exploration — FIX #1: epsilon_min lowered to 0.02
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9994
    # Reward shaping — FIX #2 & #3: renamed/fixed
    completion_bonus: float = 300.0          # given at step() level, not inside shape fn
    death_penalty: float = -80.0             # extra penalty on death beyond env default
    carrot_bonus: float = 15.0               # additional bonus per carrot beyond env default
    post_collection_step_penalty: float = -0.5  # per step after all collected
    distance_shaping_scale: float = 2.0     # FIX #3: computed directly here, no info["distance_delta"]
    invalid_move_penalty: float = -0.5
    terminal_oversample: int = 4            # FIX #5: reduced from 8
    # Misc
    report_every: int = 100
    save_every: int = 500
    seed: int = 42
    observation_mode: str = "full"          # FIX #7: single source of truth, used everywhere


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        grid: np.ndarray,
        inv: np.ndarray,
        action: int,
        reward: float,
        next_grid: np.ndarray,
        next_inv: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((
            grid.astype(np.float32, copy=False),
            inv.astype(np.float32, copy=False),
            int(action),
            float(reward),
            next_grid.astype(np.float32, copy=False),
            next_inv.astype(np.float32, copy=False),
            bool(done),
        ))

    def sample(self, n: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self._buf, n)
        g, v, a, r, ng, nv, d = zip(*batch)
        return (
            np.stack(g),
            np.stack(v),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ng),
            np.stack(nv),
            np.array(d, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Network — FIX #4: Dueling DDQN architecture
# ---------------------------------------------------------------------------

class DuelingDQNCNN(nn.Module):
    """Dueling DQN with separate value and advantage streams."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        # Conv backbone
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * GRID_SIZE * GRID_SIZE
        merged = conv_out + INV_FEATURES

        # Shared FC layer
        self.shared = nn.Sequential(
            nn.Linear(merged, 512),
            nn.ReLU(),
        )
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        feat = self.conv(grid)
        merged = torch.cat([feat, inv], dim=1)
        shared = self.shared(merged)
        v = self.value_stream(shared)
        a = self.advantage_stream(shared)
        # Dueling aggregation: Q = V + (A - mean(A))
        return v + a - a.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def _nearest_target_manhattan(env: BobbyCarrotEnv) -> float:
    """Compute normalised Manhattan distance to nearest uncollected target."""
    assert env.map_info is not None and env.bobby is not None
    px, py = env.bobby.coord_src
    best = float(GRID_SIZE * 2)
    use_carrots = env.map_info.carrot_total > 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = env.map_info.data[x + y * GRID_SIZE]
            if (use_carrots and tile == 19) or (not use_carrots and tile == 45):
                d = abs(x - px) + abs(y - py)
                if d < best:
                    best = d
    return min(1.0, best / (GRID_SIZE * 2))


def _semantic_channels(env: BobbyCarrotEnv) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (grid, inv) tensors.
    FIX #3: inv now includes manhattan_to_nearest_target computed locally.
    FIX #7: consistent encoding used by both training and play.
    """
    assert env.map_info is not None and env.bobby is not None

    grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    all_collected = env.bobby.is_finished(env.map_info)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = env.map_info.data[x + y * GRID_SIZE]
            if tile < 18:
                c = 0   # wall / impassable
            elif tile == 19:
                c = 1   # carrot (uncollected)
            elif tile == 45:
                c = 2   # egg (uncollected)
            elif tile == 44:
                # goal: active only after all collectibles
                c = 3 if all_collected else 4
            elif tile in {31, 46}:
                c = 5   # hazard / death tile
            elif tile in {32, 34, 36}:
                c = 6   # key
            elif tile in {33, 35, 37}:
                c = 7   # locked door
            else:
                c = 8   # empty / other traversable
            grid[c, y, x] = 1.0

    px, py = env.bobby.coord_src
    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
        grid[9, py, px] = 1.0
    if all_collected:
        grid[10, :, :] = 1.0  # phase flag: all targets collected

    remaining_carrots = env.map_info.carrot_total - env.bobby.carrot_count
    remaining_eggs = env.map_info.egg_total - env.bobby.egg_count
    remaining_total = remaining_carrots + remaining_eggs
    remaining_norm = min(1.0, remaining_total / max(1, env.map_info.carrot_total + env.map_info.egg_total))

    # FIX #3: compute manhattan distance to nearest target directly here
    manhattan_norm = _nearest_target_manhattan(env) if not all_collected else 0.0

    inv = np.array([
        float(env.bobby.key_gray > 0),
        float(env.bobby.key_yellow > 0),
        float(env.bobby.key_red > 0),
        remaining_norm,
        manhattan_norm,
    ], dtype=np.float32)

    return grid, inv


# ---------------------------------------------------------------------------
# Reward shaping — FIX #2 & #3
# ---------------------------------------------------------------------------

def _shape_reward(
    raw_reward: float,
    info: Dict[str, object],
    cfg: DQNConfig,
    prev_manhattan: float,
    curr_manhattan: float,
    all_collected_before: bool,
) -> float:
    reward = float(raw_reward)

    if bool(info.get("invalid_move", False)):
        reward += cfg.invalid_move_penalty

    if bool(info.get("dead", False)):
        reward += cfg.death_penalty

    carrot_delta = int(info.get("collected_carrot", 0))
    if carrot_delta > 0:
        reward += cfg.carrot_bonus * carrot_delta

    # FIX #2: completion bonus applied here, not gated by faded_out
    if bool(info.get("level_completed", False)):
        reward += cfg.completion_bonus

    # FIX #3: distance shaping computed from prev/curr manhattan, not info["distance_delta"]
    all_collected_now = bool(info.get("all_collected", False))
    level_done = bool(info.get("level_completed", False))

    if not all_collected_now and not all_collected_before:
        # Phase 1: shape toward collecting targets
        dist_delta = prev_manhattan - curr_manhattan  # positive = moved closer
        reward += cfg.distance_shaping_scale * dist_delta

    if all_collected_now and not level_done:
        # Phase 2: penalise wandering after collection, agent should head to finish tile
        reward += cfg.post_collection_step_penalty

    return reward


# ---------------------------------------------------------------------------
# Agent — FIX #4: DDQN update rule
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig, device: torch.device) -> None:
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = device
        self.epsilon = cfg.epsilon_start
        self.total_steps = 0

        self.policy_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net = DuelingDQNCNN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr, eps=1.5e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(cfg.replay_capacity)

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
            # FIX #4: Double DQN — action selected by policy, evaluated by target
            best_actions = self.policy_net(nsg_t, nsv_t).argmax(1, keepdim=True)
            q_next = self.target_net(nsg_t, nsv_t).gather(1, best_actions).squeeze(1)
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
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim":  self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "level": level,
            "map_kind": map_kind,
            "grid_channels": GRID_CHANNELS,
            "inv_features": INV_FEATURES,
        }, path)

    def load(self, path: Path) -> Dict[str, object]:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt.get("target", ckpt["policy"]))
        if "optim" in ckpt:
            self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon    = float(ckpt.get("epsilon", self.cfg.epsilon_start))
        self.total_steps = int(ckpt.get("total_steps", 0))
        return {"level": ckpt.get("level", -1), "map_kind": ckpt.get("map_kind", "normal")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(cfg: DQNConfig, map_kind: str, level: int) -> BobbyCarrotEnv:
    """Single place that creates an env — FIX #7: observation_mode from cfg everywhere."""
    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=level,
        observation_mode=cfg.observation_mode,
        local_view_size=3,
        include_inventory=True,
        headless=True,
        max_steps=cfg.max_steps,
    )
    return env


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_level(
    agent: DQNAgent,
    level: int,
    cfg: DQNConfig,
    map_kind: str,
    ckpt_path: Optional[Path] = None,
) -> Dict[str, float]:
    env = _make_env(cfg, map_kind, level)

    reward_hist: List[float] = []
    success_hist: List[float] = []
    collected_hist: List[float] = []
    loss_win: Deque[float] = deque(maxlen=cfg.report_every)

    best_success = 0.0

    for episode in range(1, cfg.episodes_per_level + 1):
        env.set_map(map_kind=map_kind, map_number=level)
        env.reset()
        grid, inv = _semantic_channels(env)

        done = False
        steps = 0
        ep_reward = 0.0
        info: Dict[str, object] = {}
        all_collected_before = False

        # snapshot manhattan distance before each step for shaping
        prev_manhattan = _nearest_target_manhattan(env)

        while not done and steps < cfg.max_steps:
            action = agent.select_action(grid, inv)
            _, raw_reward, done, info = env.step(action)

            next_grid, next_inv = _semantic_channels(env)
            curr_manhattan = float(next_inv[4])  # index 4 = manhattan_norm in inv

            all_collected_now = bool(info.get("all_collected", False))
            shaped = _shape_reward(
                raw_reward, info, cfg,
                prev_manhattan, curr_manhattan,
                all_collected_before,
            )

            agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            # FIX #5: terminal oversample only on genuine completion, reduced count
            if bool(info.get("level_completed", False)) and cfg.terminal_oversample > 0:
                for _ in range(cfg.terminal_oversample):
                    agent.replay.push(grid, inv, action, shaped, next_grid, next_inv, done)

            agent.total_steps += 1
            loss = agent.optimize_step()
            if loss is not None:
                loss_win.append(loss)

            grid, inv = next_grid, next_inv
            prev_manhattan = curr_manhattan
            all_collected_before = all_collected_now
            ep_reward += shaped
            steps += 1

        agent.decay_epsilon()

        success = 1.0 if bool(info.get("level_completed", False)) else 0.0
        collected = 1.0 if bool(info.get("all_collected", False)) else 0.0
        reward_hist.append(ep_reward)
        success_hist.append(success)
        collected_hist.append(collected)

        if episode % cfg.report_every == 0 or episode == 1:
            n = cfg.report_every
            avg_r   = float(np.mean(reward_hist[-n:]))
            avg_s   = float(np.mean(success_hist[-n:]))
            avg_c   = float(np.mean(collected_hist[-n:]))
            avg_l   = float(np.mean(loss_win)) if loss_win else 0.0
            print(
                f"[L{level}] ep={episode:5d} | "
                f"avg_reward={avg_r:8.2f} | "
                f"all_collected={avg_c:6.2%} | "
                f"success={avg_s:6.2%} | "
                f"eps={agent.epsilon:.3f} | "
                f"replay={len(agent.replay):6d} | "
                f"loss={avg_l:.4f}"
            )

        # Auto-save best checkpoint
        if ckpt_path is not None and episode % cfg.save_every == 0:
            rolling_s = float(np.mean(success_hist[-cfg.report_every:]))
            if rolling_s >= best_success:
                best_success = rolling_s
                best_path = ckpt_path.parent / f"{ckpt_path.stem}_best{ckpt_path.suffix}"
                agent.save(best_path, level=level, map_kind=map_kind)

    env.close()
    return {
        "mean_reward":      float(np.mean(reward_hist)) if reward_hist else 0.0,
        "success_rate":     float(np.mean(success_hist)) if success_hist else 0.0,
        "all_collected_rate": float(np.mean(collected_hist)) if collected_hist else 0.0,
    }


# ---------------------------------------------------------------------------
# Play / evaluation
# ---------------------------------------------------------------------------

def play_trained_dqn(
    model_path: Path,
    map_kind: str,
    map_number: int,
    episodes: int = 5,
    max_steps: int = 800,
    render: bool = False,
    render_fps: float = 5.0,
    cfg: Optional[DQNConfig] = None,
) -> None:
    if cfg is None:
        cfg = DQNConfig(max_steps=max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX #7: use cfg.observation_mode (same as training)
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
    meta = agent.load(model_path)
    agent.epsilon = 0.0  # pure greedy at play time
    print(f"Loaded checkpoint from {model_path} (trained on level={meta['level']}, kind={meta['map_kind']})")

    for ep in range(1, episodes + 1):
        env.set_map(map_kind=map_kind, map_number=map_number)
        env.reset()
        grid, inv = _semantic_channels(env)

        done = False
        steps = 0
        total_reward = 0.0
        info: Dict[str, object] = {}

        while not done and steps < max_steps:
            action = agent.select_action(grid, inv)
            _, reward, done, info = env.step(action)
            grid, inv = _semantic_channels(env)
            total_reward += reward
            steps += 1
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bobby Carrot DQN trainer (fixed)")
    p.add_argument("--play", action="store_true")
    p.add_argument("--map-kind", default="normal", choices=["normal", "egg"])
    p.add_argument("--map-number", type=int, default=1)
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--individual-levels", action="store_true")
    p.add_argument("--episodes-per-level", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.9994)
    p.add_argument("--completion-bonus", type=float, default=300.0)
    p.add_argument("--death-penalty", type=float, default=-80.0)
    p.add_argument("--distance-shaping-scale", type=float, default=2.0)
    p.add_argument("--invalid-move-penalty", type=float, default=-0.5)
    p.add_argument("--terminal-oversample", type=int, default=4)
    p.add_argument("--observation-mode", default="full", choices=["full", "local", "compact"])
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"))
    p.add_argument("--checkpoint-dir", default=str(Path(__file__).resolve().parent / "dqn_checkpoints"))
    p.add_argument("--play-episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--render-fps", type=float, default=5.0)
    return p


def _cfg_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        episodes_per_level=args.episodes_per_level,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        completion_bonus=args.completion_bonus,
        death_penalty=args.death_penalty,
        distance_shaping_scale=args.distance_shaping_scale,
        invalid_move_penalty=args.invalid_move_penalty,
        terminal_oversample=args.terminal_oversample,
        observation_mode=args.observation_mode,
        report_every=args.report_every,
        seed=args.seed,
    )


def _main() -> None:
    args = _build_parser().parse_args()
    cfg = _cfg_from_args(args)
    _seed_everything(cfg.seed)

    model_path = Path(args.model_path)
    ckpt_dir = Path(args.checkpoint_dir)

    if args.play:
        play_trained_dqn(
            model_path=model_path,
            map_kind=args.map_kind,
            map_number=args.map_number,
            episodes=args.play_episodes,
            max_steps=args.max_steps,
            render=not args.no_render,
            render_fps=args.render_fps,
            cfg=cfg,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    levels = [int(l) for l in (args.levels or [args.map_number])]

    probe_env = _make_env(cfg, args.map_kind, levels[0])
    n_actions = probe_env.action_space_n
    probe_env.close()

    if args.individual_levels and len(levels) > 1:
        for lvl in levels:
            print(f"\n=== Independent training — level {lvl} ===")
            agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
            out = ckpt_dir / f"dqn_level{lvl}_individual.pt"
            summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
            agent.save(out, level=lvl, map_kind=args.map_kind)
            print(f"Saved {out} | {summary}")
        return

    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)

    # Optionally warm-start from an existing checkpoint
    if model_path.exists():
        meta = agent.load(model_path)
        print(f"Resumed from {model_path} (level={meta['level']}, kind={meta['map_kind']})")

    for lvl in levels:
        print(f"\n=== Sequential training — level {lvl} ===")
        out = ckpt_dir / f"dqn_level{lvl}_sequential.pt"
        summary = train_one_level(agent, lvl, cfg, args.map_kind, ckpt_path=out)
        agent.save(out, level=lvl, map_kind=args.map_kind)
        print(
            f"Saved {out} | mean_reward={summary['mean_reward']:.2f} | "
            f"all_collected={summary['all_collected_rate']:.2%} | "
            f"success={summary['success_rate']:.2%}"
        )

    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    _main()
