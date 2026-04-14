"""Bobby Carrot DQN — Training, evaluation, and GUI play.

Rainbow-lite orchestrator with:
  - NoisyNet exploration (replaces ε-greedy after warmup)
  - N-step returns for faster credit assignment
  - PER with β annealing for importance sampling
  - BFS teacher mixing during warmup only

Modules:
  - dqn_env.py    → BobbyEnv, BFS helpers, physics
  - dqn_model.py  → DuelingDQN with NoisyLinear
  - dqn_buffer.py → PrioritizedReplayBuffer + NStepBuffer
  - dqn_agent.py  → DQNAgent with PER-weighted updates

Usage:
  python Bobby_Carrot/train_dqn.py --level 3 --episodes 2000
  python Bobby_Carrot/train_dqn.py --play --level 3
  python Bobby_Carrot/train_dqn.py --play-gui --level 3
"""
from __future__ import annotations

import argparse
import random
import sys
import time
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
        "Place train_dqn.py inside the project root (next to Game_Python/)."
    )
if str(_GAME_DIR) not in sys.path:
    sys.path.insert(0, str(_GAME_DIR))

from bobby_carrot.game import (  # type: ignore[missing-import] # noqa: E402
    Map, MapInfo, Bobby, State, Assets,
    VIEW_WIDTH, VIEW_HEIGHT, FRAMES, WIDTH_POINTS_DELTA, HEIGHT_POINTS_DELTA,
    VIEW_WIDTH_POINTS, VIEW_HEIGHT_POINTS, FRAMES_PER_STEP, WIDTH_POINTS,
    HEIGHT_POINTS,
)

# Import from modular components (env has no torch dependency)
from dqn_env import BobbyEnv, N_ACTIONS, ACTION_DELTA  # noqa: E402

# torch and DQNAgent imported lazily (only needed for train/play)
torch = None  # type: ignore[assignment]
DQNAgent = None  # type: ignore[assignment]
NStepBuffer = None  # type: ignore[assignment]


def _ensure_torch():
    """Lazy-import torch and DQNAgent. Called at the start of train/play."""
    global torch, DQNAgent, NStepBuffer
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError as e:
            raise RuntimeError("PyTorch is required. Install it first.") from e
    if DQNAgent is None:
        from dqn_agent import DQNAgent as _Agent
        DQNAgent = _Agent
    if NStepBuffer is None:
        from dqn_buffer import NStepBuffer as _NSB
        NStepBuffer = _NSB


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


# ── BFS teacher helpers ───────────────────────────────────────────────────────
def _bfs_teacher_action(env: BobbyEnv) -> Optional[int]:
    """Return a BFS-guided action when a reachable target exists."""
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
    for action, (dx, dy) in enumerate(ACTION_DELTA):
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
    """Pick the action that moves the agent one step closer to nearest target."""
    teacher = _bfs_teacher_action(env)
    if teacher is not None:
        return teacher
    px, py = env.pos
    valid = [a for a, (dx, dy) in enumerate(ACTION_DELTA) if env._can_move(px, py, dx, dy)]
    return random.choice(valid) if valid else random.randint(0, N_ACTIONS - 1)


# ── Training loop ─────────────────────────────────────────────────────────────
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
    eps_min:       float = 0.05,
    eps_decay:     float = 0.998,
    warmup_eps:    int   = 50,
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
    n_step:        int   = 4,
    noisy:         bool  = True,
) -> None:

    _ensure_torch()

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dev_info = ""
    if device.type == "cuda":
        dev_info = f" | {torch.cuda.get_device_name(0)}"
    elif device.type == "mps":
        dev_info = " | Apple Silicon GPU"
    print(f"Device: {device}{dev_info}")

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

    probe = BobbyEnv(map_kind, level_pool[0], max_steps)
    probe.reset()
    eff_max = probe.max_steps
    if multi:
        print(f"Multi-level training: {len(level_pool)} levels "
              f"({min(level_pool)}-{max(level_pool)}) | n_envs={n_envs}")
    else:
        print(f"Level {map_kind}-{level_pool[0]} | targets={probe.n_targets} "
              f"| max_steps={eff_max} | n_envs={n_envs}")

    # Mode info
    mode_str = "Rainbow-lite" if noisy else "DQN+PER"
    print(f"Mode: {mode_str} | n_step={n_step} | noisy={noisy}")

    # Parallel environments
    envs: List[BobbyEnv] = []
    for _ in range(n_envs):
        lvl = random.choice(level_pool)
        envs.append(BobbyEnv(map_kind, lvl, max_steps))
    for e in envs:
        e.reset()

    # Per-env n-step buffers
    nstep_buffers = [NStepBuffer(n=n_step, gamma=gamma) for _ in range(n_envs)]

    # Estimate total gradient steps for LR scheduler
    est_grad_steps = max(10_000, n_episodes * eff_max // (n_envs * grad_every))

    agent = DQNAgent(device, lr=lr, gamma=gamma,
                     batch_size=batch_size, target_update=target_update,
                     n_step=n_step, noisy=noisy,
                     total_train_steps=est_grad_steps)
    agent.epsilon = eps_start if not noisy else 0.0  # NoisyNet doesn't use epsilon

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
    stag_count = 0
    t0         = time.time()
    target_eps = total_eps + n_episodes
    warmup_done = False

    tag = f"L{min(level_pool)}-{max(level_pool)}" if multi else f"L{level_pool[0]}"
    print(f"\n=== Training {tag} for {n_episodes} episodes "
          f"(ep {total_eps}->{target_eps}) ===")
    if noisy:
        print(f"NoisyNet exploration | warmup={warmup_eps} (BFS teacher) | "
              f"grad_every={grad_every} | batch={batch_size}")
    else:
        print(f"eps: {agent.epsilon:.2f} -> {eps_min:.2f} (decay={eps_decay}/ep) | "
              f"warmup={warmup_eps} | grad_every={grad_every} | batch={batch_size}")

    last_report_ep = -1
    last_save_ep   = -1

    while total_eps < target_eps:
        # ── BATCHED observation collection ────────────────────────────────
        obs_pairs = [env.get_obs() for env in envs]
        grids = [p[0] for p in obs_pairs]
        invs  = [p[1] for p in obs_pairs]

        # ── Action selection ──────────────────────────────────────────────
        in_warmup = total_eps < warmup_eps

        if in_warmup:
            # Warmup: mostly BFS-guided (70%) to fill replay buffer
            actions = []
            for env in envs:
                if random.random() < 0.7:
                    actions.append(_bfs_best_action(env))
                else:
                    actions.append(random.randint(0, N_ACTIONS - 1))
        else:
            if not warmup_done:
                warmup_done = True
                print(f"  [warmup done] Switching to {'NoisyNet' if noisy else 'ε-greedy'} exploration")

            # NoisyNet: just use policy (noise provides exploration)
            # ε-greedy fallback: use epsilon for exploration
            policy_actions = agent.act_batch(grids, invs)

            if noisy:
                actions = policy_actions
            else:
                actions = []
                for i, env in enumerate(envs):
                    if random.random() < agent.epsilon:
                        actions.append(random.randint(0, N_ACTIONS - 1))
                    else:
                        actions.append(policy_actions[i])

        # ── Step all envs and collect transitions ─────────────────────────
        for i, env in enumerate(envs):
            g, v = grids[i], invs[i]
            teacher_action = _bfs_teacher_action(env) if in_warmup else None
            r, done, won = env.step(actions[i])
            ng, nv = env.get_obs()

            # Feed transition through n-step buffer
            raw_transition = (g, v, actions[i], r, ng, nv, done, teacher_action)
            nstep_result = nstep_buffers[i].append(raw_transition)
            if nstep_result is not None:
                agent.replay.push(*nstep_result)

            if done:
                # Flush remaining n-step transitions
                for flushed in nstep_buffers[i].flush():
                    agent.replay.push(*flushed)
                nstep_buffers[i] = NStepBuffer(n=n_step, gamma=gamma)

                cr = env.n_collected / max(1, env.n_targets)
                completed_cr.append(cr)
                completed_win.append(1.0 if env.finished else 0.0)
                total_eps += 1

                if not noisy and total_eps > warmup_eps:
                    agent.epsilon = max(eps_min, agent.epsilon * eps_decay)

                if multi:
                    env.set_level(random.choice(level_pool))
                env.reset()

        env_steps += n_envs

        # ── Warmup: no gradient updates ───────────────────────────────────
        if in_warmup:
            continue

        # ── PER beta annealing ────────────────────────────────────────────
        progress = min(1.0, (total_eps - warmup_eps) / max(1, n_episodes - warmup_eps))
        agent.replay.anneal_beta(progress)

        # ── Gradient update ───────────────────────────────────────────────
        if env_steps % grad_every == 0:
            loss = agent.update()
            if loss is not None:
                losses_buf.append(loss)

        # ── Reporting ─────────────────────────────────────────────────────
        next_report = ((last_report_ep // report_every) + 1) * report_every if last_report_ep >= 0 else report_every
        if total_eps >= next_report and total_eps != last_report_ep:
            last_report_ep = total_eps
            window = min(report_every, len(completed_win))
            sr  = float(np.mean(completed_win[-window:]))
            cr  = float(np.mean(completed_cr[-window:]))
            al  = float(np.mean(losses_buf[-200:])) if losses_buf else 0.0
            sps = env_steps / max(time.time() - t0, 1e-6)
            eta = max(0.0, (target_eps - total_eps) * eff_max / max(sps, 1) / 60)
            cur_lr = agent.opt.param_groups[0]['lr']
            per_beta = agent.replay.beta

            if noisy:
                print(f"[{tag}] ep={total_eps:5d}/{target_eps} | "
                      f"collected={cr:5.1%} | success={sr:5.1%} | "
                      f"β={per_beta:.2f} | lr={cur_lr:.1e} | "
                      f"loss={al:.4f} | sps={sps:.0f} | ETA={eta:.1f}min")
            else:
                print(f"[{tag}] ep={total_eps:5d}/{target_eps} | "
                      f"collected={cr:5.1%} | success={sr:5.1%} | "
                      f"eps={agent.epsilon:.3f} | β={per_beta:.2f} | "
                      f"loss={al:.4f} | sps={sps:.0f} | ETA={eta:.1f}min")

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

            # Stagnation detection (only for non-noisy mode)
            if not noisy:
                if sr < 0.01 and cr > 0.60 and total_eps > warmup_eps + 200:
                    stag_count += 1
                    if stag_count >= 3:
                        old_eps = agent.epsilon
                        agent.epsilon = max(agent.epsilon, 0.30)
                        print(f"  [!] Stagnation: eps {old_eps:.3f} -> {agent.epsilon:.3f}")
                        stag_count = 0
                else:
                    stag_count = 0

        next_save = ((last_save_ep // save_every) + 1) * save_every if last_save_ep >= 0 else save_every
        if total_eps >= next_save and total_eps != last_save_ep:
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
    noisy:      bool = True,
) -> None:
    _ensure_torch()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    agent  = DQNAgent(device, compile_model=False, noisy=noisy)

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
    noisy:       bool = True,
) -> None:
    """Play the trained agent with full Pygame visual rendering."""
    _ensure_torch()
    import pygame
    from pygame import Rect

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    agent  = DQNAgent(device, compile_model=False, noisy=noisy)

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

    pygame.init()
    window = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption(f"Bobby Carrot RL Agent ({map_kind}-{level})")
    clock  = pygame.time.Clock()
    assets = Assets()

    AI_MOVE_COOLDOWN_MS = 150

    for ep in range(1, n_episodes + 1):
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

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

            # AI decision
            if (not bobby.is_walking()
                and not bobby.dead
                and not done
                and bobby.state not in (State.FadeIn, State.FadeOut, State.Death)):
                if now_ms - getattr(bobby, '_last_ai_ms', 0) >= AI_MOVE_COOLDOWN_MS:
                    action = agent.act(g, v)
                    _, env_done, env_won = env.step(action)
                    g, v = env.get_obs()
                    steps_taken += 1

                    state_opt = action_map[action]
                    bobby.last_action_time = now_ms
                    bobby._last_ai_ms = now_ms
                    bobby.update_state(state_opt, frame, map_info.data)

            # Sound triggers
            if bobby.carrot_count != getattr(bobby, '_prev_carrots', 0):
                if assets.snd_carrot:
                    assets.snd_carrot.play()
                else:
                    assets._beep()
            if bobby.dead and not getattr(bobby, '_prev_dead', False):
                assets._beep()
            bobby._prev_carrots = bobby.carrot_count
            bobby._prev_dead    = bobby.dead

            # Win / death
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

            if env.step_count >= env.max_steps and not done:
                result_msg = f"TIMEOUT after {steps_taken} steps | collected={env.n_collected}/{env.n_targets}"
                done = True

            if done and bobby.state not in (State.FadeOut,):
                print(f"  Ep {ep}: {result_msg}")
                pygame.time.wait(800)
                break
            if done and bobby.faded_out:
                print(f"  Ep {ep}: {result_msg}")
                pygame.time.wait(800)
                break

            # Drawing
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
                    if texture is not None:
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
            if bobby_tex is not None:
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
            if assets.hud is not None:
                screen.blit(assets.hud,
                            (32*16 - (icon_width+4) - x_right_offset, 4 + y_offset),
                            icon_rect)
            if assets.numbers is not None:
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
        description="Bobby Carrot Rainbow-lite DQN (PER + NoisyNet + N-step)",
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
    p.add_argument("--eps-min",       type=float, default=0.05)
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
    # Rainbow-lite options
    p.add_argument("--n-step",        type=int,   default=4,
                   help="N-step returns for faster credit assignment")
    p.add_argument("--no-noisy",      action="store_true",
                   help="Disable NoisyNet (use ε-greedy instead)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    parsed_levels = _parse_levels(args.levels) if args.levels else None
    noisy = not args.no_noisy

    if args.play_gui:
        play_gui(map_kind=args.map_kind, level=args.level,
                 model_path=args.model_path, n_episodes=3,
                 ckpt_dir=args.ckpt_dir, max_steps=args.max_steps,
                 noisy=noisy)
    elif args.play:
        play(map_kind=args.map_kind, level=args.level,
             levels=parsed_levels,
             model_path=args.model_path, n_episodes=args.play_episodes,
             ckpt_dir=args.ckpt_dir, max_steps=args.max_steps,
             noisy=noisy)
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
              target_update=args.target_update,
              n_step=args.n_step, noisy=noisy)


if __name__ == "__main__":
    main()
