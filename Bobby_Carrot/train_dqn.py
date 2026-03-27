from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np

# Allow imports without requiring editable install.
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
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for DQN. Install with: pip install torch"
    ) from exc

GRID_SIZE = 16
GRID_CHANNELS = 9  # wall, carrot, egg, goal, hazard, key, door, empty, agent
INV_FEATURES = 4


@dataclass
class DQNConfig:
    episodes_per_level: int = 6000
    max_steps: int = 700
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    replay_capacity: int = 120_000
    min_replay_size: int = 2000
    target_update_steps: int = 1000
    train_every_steps: int = 4
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    report_every: int = 50
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state_grid: np.ndarray,
        state_inv: np.ndarray,
        action: int,
        reward: float,
        next_grid: np.ndarray,
        next_inv: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                state_grid.astype(np.float32, copy=False),
                state_inv.astype(np.float32, copy=False),
                int(action),
                float(reward),
                next_grid.astype(np.float32, copy=False),
                next_inv.astype(np.float32, copy=False),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self._buffer, batch_size)
        s_grid, s_inv, actions, rewards, ns_grid, ns_inv, dones = zip(*batch)
        return (
            np.stack(s_grid, axis=0),
            np.stack(s_inv, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(ns_grid, axis=0),
            np.stack(ns_inv, axis=0),
            np.array(dones, dtype=np.float32),
        )


class DQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE + INV_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        features = self.conv(grid)
        merged = torch.cat([features, inv], dim=1)
        return self.head(merged)


def _semantic_channels(env: BobbyCarrotEnv) -> Tuple[np.ndarray, np.ndarray]:
    assert env.map_info is not None
    assert env.bobby is not None

    grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = env.map_info.data[x + y * GRID_SIZE]
            if tile < 18:
                c = 0  # wall
            elif tile == 19:
                c = 1  # carrot
            elif tile == 45:
                c = 2  # egg
            elif tile == 44:
                c = 3  # goal
            elif tile in {31, 46}:
                c = 4  # hazard/death
            elif tile in {32, 34, 36}:
                c = 5  # key
            elif tile in {33, 35, 37}:
                c = 6  # door
            else:
                c = 7  # empty/other traversable dynamics
            grid[c, y, x] = 1.0

    px, py = env.bobby.coord_src
    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
        grid[8, py, px] = 1.0

    remaining_carrots = env.map_info.carrot_total - env.bobby.carrot_count
    remaining_eggs = env.map_info.egg_total - env.bobby.egg_count
    remaining_bucket = min(5, max(0, (remaining_carrots + remaining_eggs) // 2))
    inv = np.array(
        [
            float(env.bobby.key_gray > 0),
            float(env.bobby.key_yellow > 0),
            float(env.bobby.key_red > 0),
            float(remaining_bucket) / 5.0,
        ],
        dtype=np.float32,
    )
    return grid, inv


class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig, device: torch.device) -> None:
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = device

        self.policy_net = DQNCNN(n_actions=n_actions).to(device)
        self.target_net = DQNCNN(n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(capacity=cfg.replay_capacity)
        self.total_steps = 0
        self.epsilon = cfg.epsilon_start

    def select_action(self, state_grid: np.ndarray, state_inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return int(random.randint(0, self.n_actions - 1))

        with torch.no_grad():
            s_grid = torch.from_numpy(state_grid).unsqueeze(0).to(self.device)
            s_inv = torch.from_numpy(state_inv).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(s_grid, s_inv)
            return int(torch.argmax(q_vals, dim=1).item())

    def optimize_step(self) -> float | None:
        if len(self.replay) < self.cfg.min_replay_size:
            return None
        if self.total_steps % self.cfg.train_every_steps != 0:
            return None

        s_grid, s_inv, actions, rewards, ns_grid, ns_inv, dones = self.replay.sample(self.cfg.batch_size)

        s_grid_t = torch.from_numpy(s_grid).to(self.device)
        s_inv_t = torch.from_numpy(s_inv).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        ns_grid_t = torch.from_numpy(ns_grid).to(self.device)
        ns_inv_t = torch.from_numpy(ns_inv).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_pred = self.policy_net(s_grid_t, s_inv_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(ns_grid_t, ns_inv_t).max(dim=1).values
            target = rewards_t + (1.0 - dones_t) * self.cfg.gamma * q_next

        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.total_steps % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path: Path, level: int, map_kind: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "level": level,
            "map_kind": map_kind,
            "grid_channels": GRID_CHANNELS,
            "inventory_features": INV_FEATURES,
        }
        torch.save(payload, path)

    def load(self, path: Path) -> Dict[str, int | str]:
        payload = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(payload["policy_state_dict"])
        if "target_state_dict" in payload:
            self.target_net.load_state_dict(payload["target_state_dict"])
        else:
            self.target_net.load_state_dict(payload["policy_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        self.total_steps = int(payload.get("total_steps", 0))
        return {
            "level": int(payload.get("level", -1)),
            "map_kind": str(payload.get("map_kind", "normal")),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_level(
    agent: DQNAgent,
    level: int,
    cfg: DQNConfig,
    map_kind: str,
    observation_mode: str,
) -> Dict[str, float]:
    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=level,
        observation_mode=observation_mode,
        local_view_size=3,
        include_inventory=True,
        headless=True,
        max_steps=cfg.max_steps,
    )

    reward_history: List[float] = []
    success_history: List[float] = []
    all_collected_history: List[float] = []
    loss_window: Deque[float] = deque(maxlen=max(20, cfg.report_every))

    for episode in range(1, cfg.episodes_per_level + 1):
        env.set_map(map_kind=map_kind, map_number=level)
        env.reset()
        state_grid, state_inv = _semantic_channels(env)

        done = False
        steps = 0
        total_reward = 0.0
        info: Dict[str, object] = {}

        while not done and steps < cfg.max_steps:
            action = agent.select_action(state_grid, state_inv)
            _, reward, done, info = env.step(action)
            next_grid, next_inv = _semantic_channels(env)

            agent.replay.push(state_grid, state_inv, action, reward, next_grid, next_inv, done)
            agent.total_steps += 1
            loss = agent.optimize_step()
            if loss is not None:
                loss_window.append(loss)

            state_grid, state_inv = next_grid, next_inv
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        success = 1.0 if bool(info.get("level_completed", False)) else 0.0
        all_collected = 1.0 if bool(info.get("all_collected", False)) else 0.0
        reward_history.append(total_reward)
        success_history.append(success)
        all_collected_history.append(all_collected)

        if episode % cfg.report_every == 0 or episode == 1:
            avg_reward = float(np.mean(reward_history[-cfg.report_every:]))
            avg_success = float(np.mean(success_history[-cfg.report_every:]))
            avg_all = float(np.mean(all_collected_history[-cfg.report_every:]))
            avg_loss = float(np.mean(loss_window)) if loss_window else 0.0
            print(
                f"[L{level}] ep={episode:5d} | avg_reward={avg_reward:8.2f} | "
                f"all_collected={avg_all:6.2%} | success={avg_success:6.2%} | "
                f"eps={agent.epsilon:.3f} | replay={len(agent.replay):6d} | loss={avg_loss:.4f}"
            )

    env.close()

    return {
        "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
        "success_rate": float(np.mean(success_history)) if success_history else 0.0,
        "all_collected_rate": float(np.mean(all_collected_history)) if all_collected_history else 0.0,
    }


def play_trained_dqn(
    model_path: Path,
    map_kind: str,
    map_number: int,
    episodes: int,
    max_steps: int,
    render: bool,
    render_fps: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DQNConfig(max_steps=max_steps)
    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode="compact",
        local_view_size=3,
        include_inventory=True,
        headless=not render,
        max_steps=max_steps,
    )

    agent = DQNAgent(n_actions=env.action_space_n, cfg=cfg, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0

    for ep in range(1, episodes + 1):
        env.set_map(map_kind=map_kind, map_number=map_number)
        env.reset()
        state_grid, state_inv = _semantic_channels(env)

        done = False
        steps = 0
        total_reward = 0.0
        info: Dict[str, object] = {}

        while not done and steps < max_steps:
            action = agent.select_action(state_grid, state_inv)
            _, reward, done, info = env.step(action)
            state_grid, state_inv = _semantic_channels(env)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and play Bobby Carrot with DQN (PyTorch)")
    parser.add_argument("--play", action="store_true", help="Run autonomous gameplay with trained model")
    parser.add_argument("--map-kind", type=str, default="normal", choices=["normal", "egg"], help="Map kind")
    parser.add_argument("--map-number", type=int, default=3, help="Single map number")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="List of map levels to train")
    parser.add_argument("--individual-levels", action="store_true", help="Train each level independently from scratch")
    parser.add_argument("--episodes-per-level", type=int, default=6000, help="Episodes to train per level")
    parser.add_argument("--max-steps", type=int, default=700, help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.999, help="Epsilon decay per episode")
    parser.add_argument("--report-every", type=int, default=50, help="Logging interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "dqn_bobby.pt"),
        help="Model path for single training run or play mode",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "dqn_checkpoints"),
        help="Checkpoint directory when training multiple levels",
    )
    parser.add_argument("--play-episodes", type=int, default=3, help="Episodes in play mode")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering in play mode")
    parser.add_argument("--render-fps", type=float, default=5.0, help="Playback fps in play mode")
    return parser


def _make_cfg(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        episodes_per_level=args.episodes_per_level,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        report_every=args.report_every,
        seed=args.seed,
    )


def _train_levels(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    _seed_everything(cfg.seed)

    levels = args.levels if args.levels else [args.map_number]
    levels = [int(l) for l in levels]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_env = BobbyCarrotEnv(
        map_kind=args.map_kind,
        map_number=levels[0],
        observation_mode="compact",
        local_view_size=3,
        include_inventory=True,
        headless=True,
        max_steps=cfg.max_steps,
    )
    n_actions = base_env.action_space_n
    base_env.close()

    ckpt_dir = Path(args.checkpoint_dir)

    if args.individual_levels and len(levels) > 1:
        for level in levels:
            print(f"\n=== Training independently for level {level} ===")
            agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
            summary = train_one_level(
                agent=agent,
                level=level,
                cfg=cfg,
                map_kind=args.map_kind,
                observation_mode="compact",
            )
            out = ckpt_dir / f"dqn_level{level}_individual.pt"
            agent.save(out, level=level, map_kind=args.map_kind)
            print(
                f"Saved {out} | mean_reward={summary['mean_reward']:.2f} | "
                f"all_collected={summary['all_collected_rate']:.2%} | success={summary['success_rate']:.2%}"
            )
        return

    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)
    for level in levels:
        print(f"\n=== Sequential training on level {level} ===")
        summary = train_one_level(
            agent=agent,
            level=level,
            cfg=cfg,
            map_kind=args.map_kind,
            observation_mode="compact",
        )
        out = ckpt_dir / f"dqn_level{level}_sequential.pt"
        agent.save(out, level=level, map_kind=args.map_kind)
        print(
            f"Saved {out} | mean_reward={summary['mean_reward']:.2f} | "
            f"all_collected={summary['all_collected_rate']:.2%} | success={summary['success_rate']:.2%}"
        )

    model_path = Path(args.model_path)
    agent.save(model_path, level=levels[-1], map_kind=args.map_kind)
    print(f"Final sequential model saved to: {model_path}")


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.play:
        play_trained_dqn(
            model_path=Path(args.model_path),
            map_kind=args.map_kind,
            map_number=args.map_number,
            episodes=args.play_episodes,
            max_steps=args.max_steps,
            render=not args.no_render,
            render_fps=args.render_fps,
        )
        return

    _train_levels(args)


if __name__ == "__main__":
    _main()
