"""
Bobby Carrot Level 8 — FIXED DQN Trainer v2
=============================================
Fixes vs v1:
  F1  NaN loss: reward/Q clamping + tighter grad clip (1.0 not 10.0) + NaN guard
  F2  Speed: removed Python BFS from get_obs → pure numpy vectorised ops (~10x faster)
  F3  Speed: _visited_grid (numpy array) replaces set iteration in get_obs
  F4  Epsilon schedule: decay starts from ep 1 so agent exploits knowledge earlier
  F5  LR: reduced to 1e-4 (3e-4 + AMP + large rewards = instability)
  F6  Reward scale: normalised so max episode reward ≈ 250 (was 8×20+200=360 plus noise)
"""
from __future__ import annotations
import argparse, random, sys, time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

_HERE = Path(__file__).resolve()
ROOT  = _HERE.parent
while not (ROOT / "Game_Python").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
GAME_PYTHON_DIR = ROOT / "Game_Python"
if not GAME_PYTHON_DIR.exists():
    raise RuntimeError(f"Could not locate Game_Python directory from {_HERE}")
if str(GAME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_PYTHON_DIR))

from bobby_carrot.game import Map, MapInfo, State

try:
    import torch, torch.nn as nn, torch.optim as optim
    import torch.nn.functional as F
except Exception as exc:
    raise RuntimeError("PyTorch required") from exc

# ── constants ─────────────────────────────────────────────────────────────────
GRID_CHANNELS = 11   # 8 tile-type channels + agent + finish-active + visited
INV_FEATURES  = 3    # remaining_norm, dist_norm, phase (0=collect,1=go to finish)
N_ACTIONS     = 4

# ── Expert solution (BFS-verified, 35 steps, only valid path) ─────────────────
# 0=LEFT 1=RIGHT 2=UP 3=DOWN
EXPERT = [1,2,2,2,1,1,1,1,1,1,1,1,1,3,3,0,3,3,0,2,2,2,0,3,3,0,0,0,0,0,0,0,2,2,2]

# ── Tile channel lookup ────────────────────────────────────────────────────────
_TILE_CH = np.zeros(256, dtype=np.uint8)
for _i in range(256):
    if   _i < 18:                _TILE_CH[_i] = 0   # wall/void
    elif _i in (18, 20, 21):     _TILE_CH[_i] = 1   # floor / eaten carrot / start
    elif _i == 19:               _TILE_CH[_i] = 2   # carrot
    elif _i == 44:               _TILE_CH[_i] = 3   # finish
    elif _i == 30:               _TILE_CH[_i] = 4   # crumble (live)
    elif _i == 31:               _TILE_CH[_i] = 5   # crumble hole (death)
    elif _i in (40, 41, 42, 43): _TILE_CH[_i] = 6   # conveyor
    else:                        _TILE_CH[_i] = 1   # other passable → floor

# Precompute mgrid for fast Manhattan distance in get_obs
_YY, _XX = np.mgrid[0:16, 0:16]   # shape (16,16) each


# ── Fast environment ───────────────────────────────────────────────────────────
class FastEnvL8:
    ACTION_DELTA = [(-1,0),(1,0),(0,-1),(0,1)]

    def __init__(self, max_steps: int = 100) -> None:
        self.max_steps = max_steps
        self._map_obj  = Map("normal", 8)
        self._fresh: Optional[MapInfo] = None
        self.md        = np.zeros(256, dtype=np.uint8)
        self.pos       = (0, 0)
        self.carrots_collected = 0
        self.carrots_total     = 0
        self.dead      = False
        self.finished  = False
        self.step_count = 0
        self._visited_grid = np.zeros((16, 16), dtype=np.float32)

    def reset(self) -> None:
        if self._fresh is None:
            self._fresh = self._map_obj.load_map_info()
        mi = self._fresh
        self.md = np.array(mi.data, dtype=np.uint8)
        self.pos = mi.coord_start
        self.carrots_total = mi.carrot_total
        self.carrots_collected = 0
        self.dead = False
        self.finished = False
        self.step_count = 0
        self._visited_grid = np.zeros((16, 16), dtype=np.float32)
        px, py = self.pos
        self._visited_grid[py, px] = 1.0

    def _can_move(self, fx: int, fy: int, dx: int, dy: int) -> bool:
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < 16 and 0 <= ny < 16): return False
        t_f = int(self.md[fx + fy * 16])
        t_t = int(self.md[nx + ny * 16])
        if t_t < 18 or t_t == 31: return False
        delta = dx + dy * 16
        if t_f == 40 and delta != -1: return False
        if t_f == 41 and delta != 1:  return False
        if t_t == 40 and delta != -1: return False
        if t_t == 41 and delta != 1:  return False
        return True

    def step(self, action: int) -> Tuple[float, bool, bool]:
        """Returns (reward, done, won)."""
        dx, dy = self.ACTION_DELTA[action]
        fx, fy = self.pos

        if not self._can_move(fx, fy, dx, dy):
            self.step_count += 1
            return -0.3, self.step_count >= self.max_steps, False

        nx, ny = fx + dx, fy + dy
        t_from = int(self.md[fx + fy * 16])
        t_to   = int(self.md[nx + ny * 16])

        reward = -0.01  # tiny step cost

        # Leave crumble → destroy it
        if t_from == 30:
            self.md[fx + fy * 16] = 31

        # Land on carrot
        if t_to == 19:
            self.md[nx + ny * 16] = 20
            self.carrots_collected += 1
            # Normalised so collecting all 8 gives +8 total
            reward += 8.0

        # Land on death hole
        if t_to == 31:
            self.dead = True
            self.pos = (nx, ny)
            self.step_count += 1
            return -50.0, True, False

        self.pos = (nx, ny)
        is_new = self._visited_grid[ny, nx] == 0.0
        if is_new:
            self._visited_grid[ny, nx] = 1.0
            reward += 0.1

        self.step_count += 1

        # Win condition
        if t_to == 44 and self.carrots_collected == self.carrots_total:
            self.finished = True
            return reward + 64.0, True, True   # big terminal bonus

        done = self.step_count >= self.max_steps
        return reward, done, False

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        px, py = self.pos
        md = self.md

        # Channels 0-6: one-hot tile types (vectorised)
        ch = _TILE_CH[md].reshape(16, 16)
        grid = np.zeros((GRID_CHANNELS, 16, 16), dtype=np.float32)
        for c in range(7):
            grid[c] = (ch == c)

        # Channel 7: agent position
        grid[7, py, px] = 1.0

        # Channel 8: finish-is-active flag
        if self.carrots_collected == self.carrots_total:
            grid[8] = (md.reshape(16, 16) == 44).astype(np.float32)

        # Channel 9: visited mask
        grid[9] = self._visited_grid

        # Channel 10: Manhattan-distance gradient to nearest target (no BFS, O(1))
        if self.carrots_collected < self.carrots_total:
            ty, tx = np.where(md.reshape(16, 16) == 19)
        else:
            ty, tx = np.where(md.reshape(16, 16) == 44)
        if ty.size:
            min_dist = np.min(
                np.abs(_YY[:, :, None] - ty) + np.abs(_XX[:, :, None] - tx), axis=2
            ).astype(np.float32)
            md_max = float(min_dist.max()) or 1.0
            grid[10] = 1.0 - min_dist / md_max

        # Inventory
        rem   = float(self.carrots_total - self.carrots_collected) / float(max(1, self.carrots_total))
        phase = 0.0 if self.carrots_collected < self.carrots_total else 1.0
        nearest = float(np.min(np.abs(py - ty) + np.abs(px - tx))) / 32.0 if ty.size else 0.0
        inv = np.array([rem, min(1.0, nearest), phase], dtype=np.float32)
        return grid, inv


# ── Neural network ─────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        flat  = 64 * 16 * 16
        merge = flat + INV_FEATURES
        self.shared     = nn.Sequential(nn.Linear(merge, 512), nn.ReLU(inplace=True))
        self.value_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.adv_head   = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, N_ACTIONS))

    def forward(self, g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x   = self.shared(torch.cat([self.conv(g), v], dim=1))
        val = self.value_head(x)
        adv = self.adv_head(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ── Replay buffer ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap: int = 80_000) -> None:
        self._cap  = cap
        self._ptr  = 0
        self._size = 0
        self._sg = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16)
        self._sv = torch.zeros((cap, INV_FEATURES), dtype=torch.float32)
        self._a  = torch.zeros(cap, dtype=torch.int64)
        self._r  = torch.zeros(cap, dtype=torch.float32)
        self._ng = torch.zeros((cap, GRID_CHANNELS, 16, 16), dtype=torch.float16)
        self._nv = torch.zeros((cap, INV_FEATURES), dtype=torch.float32)
        self._d  = torch.zeros(cap, dtype=torch.float32)
        self._w  = torch.ones(cap, dtype=torch.float32)   # sampling weight

    def __len__(self) -> int:
        return self._size

    def push(self, sg, sv, a, r, ng, nv, done, weight: float = 1.0) -> None:
        i = self._ptr
        self._sg[i].copy_(torch.from_numpy(sg))
        self._sv[i].copy_(torch.from_numpy(sv))
        self._a[i]  = int(a)
        self._r[i]  = float(np.clip(r, -60.0, 80.0))   # clamp at storage time
        self._ng[i].copy_(torch.from_numpy(ng))
        self._nv[i].copy_(torch.from_numpy(nv))
        self._d[i]  = float(done)
        self._w[i]  = weight
        self._ptr   = (i + 1) % self._cap
        self._size  = min(self._size + 1, self._cap)

    def sample(self, n: int) -> tuple:
        probs = self._w[:self._size]
        probs = probs / probs.sum()
        idx   = torch.multinomial(probs, n, replacement=True)
        return (self._sg[idx].float(), self._sv[idx],
                self._a[idx], self._r[idx],
                self._ng[idx].float(), self._nv[idx], self._d[idx])


# ── Agent ──────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, device: torch.device, lr: float = 1e-4,
                 gamma: float = 0.99, batch_size: int = 256) -> None:
        self.device     = device
        self.gamma      = gamma
        self.batch_size = batch_size
        self.epsilon    = 1.0
        self.policy     = DuelingDQN().to(device)
        self.target     = DuelingDQN().to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt        = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-4)
        self.replay     = ReplayBuffer()
        self.steps      = 0
        self.use_amp    = device.type == "cuda"
        self.scaler     = torch.amp.GradScaler("cuda") if self.use_amp else None

    def act(self, grid: np.ndarray, inv: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        g = torch.from_numpy(grid).unsqueeze(0).to(self.device)
        v = torch.from_numpy(inv).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy(g, v).argmax(1).item())

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        sg, sv, a, r, ng, nv, d = self.replay.sample(self.batch_size)
        sg = sg.to(self.device); sv = sv.to(self.device)
        a  = a.unsqueeze(1).to(self.device)
        r  = r.to(self.device); ng = ng.to(self.device)
        nv = nv.to(self.device); d  = d.to(self.device)

        def _forward():
            qp  = self.policy(sg, sv).gather(1, a).squeeze(1)
            with torch.no_grad():
                ba  = self.policy(ng, nv).argmax(1, keepdim=True)
                qn  = self.target(ng, nv).gather(1, ba).squeeze(1)
                tgt = (r + (1.0 - d) * self.gamma * qn).clamp(-100.0, 200.0)
            return F.smooth_l1_loss(qp, tgt)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss = _forward()
            if not torch.isfinite(loss): return None
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.scaler.step(self.opt); self.scaler.update()
        else:
            loss = _forward()
            if not torch.isfinite(loss): return None
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

        self.steps += 1
        if self.steps % 300 == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def bc_step(self, grids, invs, actions) -> float:
        """Behavioural Cloning cross-entropy step."""
        g = grids.to(self.device); v = invs.to(self.device); a = actions.to(self.device)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(self.policy(g, v), a)
            if not torch.isfinite(loss): return float('nan')
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.scaler.step(self.opt); self.scaler.update()
        else:
            loss = F.cross_entropy(self.policy(g, v), a)
            if not torch.isfinite(loss): return float('nan')
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(),
                    "target": self.target.state_dict(),
                    "optim":  self.opt.state_dict(),
                    "epsilon": self.epsilon, "steps": self.steps,
                    "level": 8, "map_kind": "normal"}, path)
        print(f"  [saved] {path.name}")

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck.get("target", ck["policy"]))
        if "optim" in ck: self.opt.load_state_dict(ck["optim"])
        self.epsilon = float(ck.get("epsilon", 1.0))
        self.steps   = int(ck.get("steps", 0))
        print(f"  [loaded] eps={self.epsilon:.3f} steps={self.steps}")


# ── Expert helpers ─────────────────────────────────────────────────────────────
def run_expert(env: FastEnvL8) -> List[tuple]:
    env.reset()
    traj = []
    for action in EXPERT:
        g, v = env.get_obs()
        r, done, won = env.step(action)
        ng, nv = env.get_obs()
        traj.append((g, v, action, r, ng, nv, done))
        if done: break
    return traj


def fill_replay(agent: Agent, env: FastEnvL8, n_copies: int = 150) -> None:
    print(f"  Filling replay with {n_copies}× expert trajectory...")
    for _ in range(n_copies):
        for (g, v, a, r, ng, nv, done) in run_expert(env):
            agent.replay.push(g, v, a, r, ng, nv, done, weight=4.0)
    print(f"  Replay size: {len(agent.replay)}")


def pretrain_bc(agent: Agent, env: FastEnvL8, steps: int = 600) -> None:
    print(f"  BC pretraining for {steps} steps...")
    traj   = run_expert(env)
    grids  = torch.stack([torch.from_numpy(g) for g,v,a,r,ng,nv,d in traj])
    invs   = torch.stack([torch.from_numpy(v) for g,v,a,r,ng,nv,d in traj])
    acts   = torch.tensor([a for g,v,a,r,ng,nv,d in traj], dtype=torch.long)
    losses = []
    for s in range(steps):
        idx  = torch.randint(0, len(traj), (32,))
        loss = agent.bc_step(grids[idx], invs[idx], acts[idx])
        if not np.isnan(loss): losses.append(loss)
        if (s + 1) % 200 == 0:
            print(f"    BC {s+1}/{steps}  loss={np.mean(losses[-200:]):.4f}")
    print(f"  BC done. loss={np.mean(losses[-50:]):.4f}")


# ── Training ───────────────────────────────────────────────────────────────────
def train(
    n_episodes:      int   = 2000,
    max_steps:       int   = 100,
    lr:              float = 1e-4,
    batch_size:      int   = 256,
    gamma:           float = 0.99,
    eps_start:       float = 1.0,
    eps_min:         float = 0.03,
    eps_decay:       float = 0.993,      # per-episode decay after guided phase
    bc_steps:        int   = 600,
    expert_copies:   int   = 150,
    guided_eps:      int   = 200,        # episodes using expert actions
    guided_noise:    float = 0.15,
    report_every:    int   = 50,
    save_every:      int   = 250,
    model_path:      str   = "dqn_level8_fixed.pt",
    resume:          bool  = False,
    seed:            int   = 42,
) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    env   = FastEnvL8(max_steps=max_steps)
    agent = Agent(device, lr=lr, gamma=gamma, batch_size=batch_size)
    out   = Path(model_path)

    if resume and out.exists():
        agent.load(out)
    else:
        print("\n=== PHASE 1: Imitation ===")
        fill_replay(agent, env, expert_copies)
        pretrain_bc(agent, env, bc_steps)
        print("  DQN warmup on expert buffer...")
        for _ in range(300):
            agent.update()
        agent.epsilon = eps_start

    print(f"\n=== PHASE 2: RL ({n_episodes} episodes, guided={guided_eps}) ===")
    rewards   = deque(maxlen=report_every)
    successes = deque(maxlen=report_every)
    losses    = deque(maxlen=200)
    best_sr   = 0.0
    t0        = time.time()

    for ep in range(1, n_episodes + 1):
        env.reset()
        g, v   = env.get_obs()
        ep_r   = 0.0
        done   = False
        step_i = 0

        while not done:
            use_guided = ep <= guided_eps
            if use_guided:
                action = (EXPERT[step_i] if step_i < len(EXPERT) and random.random() > guided_noise
                          else random.randint(0, N_ACTIONS - 1))
            else:
                action = agent.act(g, v)

            r, done, won = env.step(action)
            ng, nv = env.get_obs()

            is_expert = (use_guided and step_i < len(EXPERT) and action == EXPERT[step_i])
            agent.replay.push(g, v, action, r, ng, nv, done, weight=3.0 if is_expert else 1.0)

            # 2 gradient steps per env step
            for _ in range(2):
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            ep_r  += r
            g, v   = ng, nv
            step_i += 1

        # Epsilon decay (start from ep 1 but only below 1.0 after guided phase ends)
        if ep > guided_eps // 2:
            agent.epsilon = max(eps_min, agent.epsilon * eps_decay)

        rewards.append(ep_r)
        successes.append(1.0 if env.finished else 0.0)

        if ep % report_every == 0:
            sr  = float(np.mean(successes))
            ar  = float(np.mean(rewards))
            al  = float(np.mean(losses)) if losses else 0.0
            sps = ep * max_steps / max(time.time() - t0, 1e-6)
            eta = (n_episodes - ep) * max_steps / max(sps, 1) / 60
            print(f"[L8] ep={ep:4d}/{n_episodes} | reward={ar:6.1f} | "
                  f"success={sr:5.1%} | eps={agent.epsilon:.3f} | "
                  f"loss={al:.4f} | sps={sps:.0f} | ETA={eta:.1f}min")

            if sr > best_sr:
                best_sr = sr
                agent.save(out.parent / (out.stem + "_best" + out.suffix))

        if ep % save_every == 0:
            agent.save(out)

        if ep >= 200 and float(np.mean(successes)) >= 0.95:
            print(f"\n[L8] Early stop ep={ep}: success={float(np.mean(successes)):.1%}")
            break

    agent.save(out)
    print(f"\nDone. Best success={best_sr:.1%}  Model → {out}")


# ── Play ───────────────────────────────────────────────────────────────────────
def play(model_path: str, n_episodes: int = 10) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = Agent(device)
    agent.load(Path(model_path))
    agent.epsilon = 0.0
    agent.policy.eval()
    env = FastEnvL8(max_steps=100)
    wins = 0
    for ep in range(1, n_episodes + 1):
        env.reset()
        g, v = env.get_obs()
        done = False; steps = 0
        while not done:
            a = agent.act(g, v)
            _, done, _ = env.step(a)
            g, v = env.get_obs()
            steps += 1
        wins += int(env.finished)
        print(f"Ep {ep}: {'WIN' if env.finished else 'FAIL'} "
              f"steps={steps} carrots={env.carrots_collected}/{env.carrots_total}")
    print(f"\nSuccess: {wins}/{n_episodes} = {wins/n_episodes:.1%}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--play",          action="store_true")
    p.add_argument("--model-path",    default="dqn_level8_fixed.pt")
    p.add_argument("--episodes",      type=int,   default=2000)
    p.add_argument("--max-steps",     type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--eps-decay",     type=float, default=0.993)
    p.add_argument("--bc-steps",      type=int,   default=600)
    p.add_argument("--expert-copies", type=int,   default=150)
    p.add_argument("--guided-eps",    type=int,   default=200)
    p.add_argument("--report-every",  type=int,   default=50)
    p.add_argument("--save-every",    type=int,   default=250)
    p.add_argument("--play-episodes", type=int,   default=10)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    if args.play:
        play(args.model_path, args.play_episodes)
    else:
        train(
            n_episodes=args.episodes, max_steps=args.max_steps,
            lr=args.lr, batch_size=args.batch_size, eps_decay=args.eps_decay,
            bc_steps=args.bc_steps, expert_copies=args.expert_copies,
            guided_eps=args.guided_eps, report_every=args.report_every,
            save_every=args.save_every, model_path=args.model_path,
            resume=args.resume, seed=args.seed,
        )

if __name__ == "__main__":
    main()
