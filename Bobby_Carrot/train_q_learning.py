from __future__ import annotations

# pyright: reportMissingImports=false

import pickle
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


@dataclass
class QLearningConfig:
    episodes: int = 3000
    alpha: float = 0.15
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    max_steps: int = 500
    report_every: int = 50
    model_path: Path = Path(__file__).resolve().parent / "q_table_bobby.pkl"


def _obs_key(obs: np.ndarray) -> bytes:
    return obs.tobytes()


def _epsilon_greedy_action(
    q_table: Dict[bytes, np.ndarray],
    state_key: bytes,
    action_space_n: int,
    epsilon: float,
) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(0, action_space_n))

    if state_key not in q_table:
        q_table[state_key] = np.zeros(action_space_n, dtype=np.float32)
    return int(np.argmax(q_table[state_key]))


def train_q_learning(
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "local",
    local_view_size: int = 5,
    config: QLearningConfig | None = None,
) -> Dict[bytes, np.ndarray]:
    cfg = config or QLearningConfig()

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        local_view_size=local_view_size,
        headless=True,
        max_steps=cfg.max_steps,
    )

    q_table: Dict[bytes, np.ndarray] = {}
    epsilon = cfg.epsilon_start

    reward_history: List[float] = []
    success_history: List[float] = []
    step_history: List[int] = []

    for episode in range(1, cfg.episodes + 1):
        obs = env.reset()
        state_key = _obs_key(obs)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = _epsilon_greedy_action(
                q_table=q_table,
                state_key=state_key,
                action_space_n=env.action_space_n,
                epsilon=epsilon,
            )

            next_obs, reward, done, info = env.step(action)
            next_key = _obs_key(next_obs)

            if state_key not in q_table:
                q_table[state_key] = np.zeros(env.action_space_n, dtype=np.float32)
            if next_key not in q_table:
                q_table[next_key] = np.zeros(env.action_space_n, dtype=np.float32)

            target = reward
            if not done:
                target += cfg.gamma * float(np.max(q_table[next_key]))

            q_table[state_key][action] += cfg.alpha * (target - q_table[state_key][action])

            state_key = next_key
            total_reward += reward
            steps += 1

            if steps >= cfg.max_steps:
                break

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        reward_history.append(total_reward)
        success_history.append(1.0 if info.get("level_completed", False) else 0.0)
        step_history.append(steps)

        if episode % cfg.report_every == 0 or episode == 1:
            avg_reward = float(np.mean(reward_history[-cfg.report_every :]))
            avg_success = float(np.mean(success_history[-cfg.report_every :]))
            avg_steps = float(np.mean(step_history[-cfg.report_every :]))
            print(
                f"Episode {episode:4d} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"success_rate={avg_success:5.2%} | "
                f"avg_steps={avg_steps:6.1f} | "
                f"epsilon={epsilon:.3f}"
            )

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.model_path.open("wb") as f:
        pickle.dump(q_table, f)

    env.close()

    print(f"Training complete. Q-table saved to: {cfg.model_path}")
    return q_table


def load_q_table(model_path: Path | None = None) -> Dict[bytes, np.ndarray]:
    path = model_path or (Path(__file__).resolve().parent / "q_table_bobby.pkl")
    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def play_trained_agent(
    model_path: Path | None = None,
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "local",
    local_view_size: int = 5,
    render: bool = True,
    max_steps: int = 500,
) -> Tuple[float, bool, int]:
    q_table = load_q_table(model_path)

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        local_view_size=local_view_size,
        headless=not render,
        max_steps=max_steps,
    )

    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    success = False

    while not done and steps < max_steps:
        state_key = _obs_key(obs)
        if state_key in q_table:
            action = int(np.argmax(q_table[state_key]))
        else:
            action = int(np.random.randint(0, env.action_space_n))

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            env.render()

        if info.get("level_completed", False):
            success = True

    env.close()

    print(
        f"Play finished | total_reward={total_reward:.2f} | "
        f"success={success} | steps={steps}"
    )

    return total_reward, success, steps


def evaluate_q_table(
    episodes: int = 100,
    model_path: Path | None = None,
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "local",
    local_view_size: int = 5,
    max_steps: int = 500,
) -> Dict[str, float]:
    rewards: List[float] = []
    successes: List[float] = []
    steps_list: List[int] = []

    for _ in range(episodes):
        total_reward, success, steps = play_trained_agent(
            model_path=model_path,
            map_kind=map_kind,
            map_number=map_number,
            observation_mode=observation_mode,
            local_view_size=local_view_size,
            render=False,
            max_steps=max_steps,
        )
        rewards.append(total_reward)
        successes.append(1.0 if success else 0.0)
        steps_list.append(steps)

    metrics = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps_list)),
    }

    print("Evaluation summary")
    print(f"episodes={episodes}")
    print(f"mean_reward={metrics['mean_reward']:.2f}")
    print(f"std_reward={metrics['std_reward']:.2f}")
    print(f"success_rate={metrics['success_rate']:.2%}")
    print(f"mean_steps={metrics['mean_steps']:.2f}")

    return metrics


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Bobby Carrot Q-learning agent")
    parser.add_argument("--eval", action="store_true", dest="eval_mode", help="Run evaluation instead of training")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training/evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--map-kind", type=str, default="normal", choices=["normal", "egg"], help="Map kind")
    parser.add_argument("--map-number", type=int, default=1, help="Map number")
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="local",
        choices=["local", "full"],
        help="Observation type",
    )
    parser.add_argument("--local-view-size", type=int, default=5, help="Odd local window size")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "q_table_bobby.pkl"),
        help="Path to q-table file",
    )
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay")
    parser.add_argument("--report-every", type=int, default=50, help="Training log interval")
    return parser


def _main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.eval_mode:
        evaluate_q_table(
            episodes=args.episodes,
            model_path=model_path,
            map_kind=args.map_kind,
            map_number=args.map_number,
            observation_mode=args.observation_mode,
            local_view_size=args.local_view_size,
            max_steps=args.max_steps,
        )
        return

    cfg = QLearningConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        max_steps=args.max_steps,
        report_every=args.report_every,
        model_path=model_path,
    )

    train_q_learning(
        map_kind=args.map_kind,
        map_number=args.map_number,
        observation_mode=args.observation_mode,
        local_view_size=args.local_view_size,
        config=cfg,
    )


if __name__ == "__main__":
    _main()
