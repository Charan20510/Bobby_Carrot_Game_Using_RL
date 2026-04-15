"""Bobby Carrot DQN — Model Merge Utility.

Merges model checkpoints trained on individual levels into a single
generalised model using weight averaging + optional fine-tuning.

WHY SIMPLE AVERAGING IS RISKY:
  Neural networks trained from different random seeds or on different tasks
  converge to different loss basins.  Averaging weights across basins usually
  lands in a HIGH-loss region (imagine averaging two mountain-peak coordinates
  — you end up in the valley, which is a high-loss saddle, not a good minimum).

WHAT WORKS INSTEAD:
  1. **Sequential fine-tuning** (RECOMMENDED):
     Train on L3 → fine-tune on L4 → fine-tune on L5.
     Each checkpoint inherits the skills from previous levels and builds on them.

  2. **Weight averaging from SAME trajectory** (Model Soup):
     If you save multiple checkpoints during a SINGLE multi-level training run
     (e.g., checkpoint at ep 2000, 4000, 6000), averaging THOSE works well
     because they share the same loss basin.

  3. **Naive merge + short fine-tune** (COMPROMISE):
     Average individually-trained weights, then fine-tune on all levels jointly
     for 1000-2000 episodes to "heal" the averaged model back into a good region.

Usage:
  # Sequential fine-tuning (best quality):
  python Bobby_Carrot/merge_models.py --mode sequential \\
    --models dqn_level3.pt dqn_level4.pt dqn_level5.pt \\
    --output dqn_level3_5_merged.pt

  # Naive weight average (requires fine-tuning after):
  python Bobby_Carrot/merge_models.py --mode average \\
    --models dqn_level3.pt dqn_level4.pt dqn_level5.pt \\
    --output dqn_level3_5_averaged.pt

  # After naive merge, MUST fine-tune:
  python Bobby_Carrot/train_dqn.py --levels "3-5" --episodes 2000 \\
    --resume --model-path dqn_level3_5_averaged.pt
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict

import torch


def _load_state_dict(path: Path, device: torch.device) -> Dict:
    """Load checkpoint, handling both safe and legacy formats."""
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ck = torch.load(path, map_location=device, weights_only=False)
    return ck


def average_weights(state_dicts: List[OrderedDict]) -> OrderedDict:
    """Average model parameters across multiple state dicts.

    All state dicts must have the same architecture (same keys & shapes).
    """
    n = len(state_dicts)
    if n == 0:
        raise ValueError("No state dicts to average")
    if n == 1:
        return copy.deepcopy(state_dicts[0])

    avg = OrderedDict()
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        avg[key] = sum(tensors) / n
        # Preserve original dtype
        avg[key] = avg[key].to(state_dicts[0][key].dtype)
    return avg


def merge_sequential(paths: List[Path], device: torch.device) -> Dict:
    """Take the LAST model in the sequence as the merged result.

    In sequential fine-tuning, each model builds on the previous one,
    so the last model already contains skills from all levels.
    """
    if not paths:
        raise ValueError("No model paths provided")
    print(f"Sequential merge: using final model {paths[-1].name}")
    print(f"  (This model was fine-tuned through {len(paths)} levels)")
    return _load_state_dict(paths[-1], device)


def merge_average(paths: List[Path], device: torch.device) -> Dict:
    """Average the weights of independently trained models.

    WARNING: This typically produces a degraded model that MUST be
    fine-tuned on the combined level set for 1000-2000 episodes.
    """
    print(f"Averaging weights from {len(paths)} models...")
    checkpoints = []
    for p in paths:
        ck = _load_state_dict(p, device)
        checkpoints.append(ck)
        lvl = ck.get("level", "?")
        eps = ck.get("total_eps", "?")
        sr = ck.get("best_sr", "?")
        print(f"  {p.name}: level={lvl}, episodes={eps}, best_sr={sr}")

    # Average policy weights
    policy_dicts = [ck["policy"] for ck in checkpoints]
    avg_policy = average_weights(policy_dicts)

    # Average target weights (if available)
    target_dicts = [ck.get("target", ck["policy"]) for ck in checkpoints]
    avg_target = average_weights(target_dicts)

    # Build merged checkpoint
    merged = {
        "policy": avg_policy,
        "target": avg_target,
        "epsilon": 0.1,        # Reset to moderate exploration
        "total_steps": 0,      # Reset step count
        "total_eps": 0,        # Reset episode count
        "best_sr": 0.0,        # Reset best success rate
        "level": -1,           # Multi-level
        "map_kind": checkpoints[0].get("map_kind", "normal"),
        "noisy": checkpoints[0].get("noisy", True),
        "n_step": checkpoints[0].get("n_step", 4),
        "merge_info": {
            "method": "weight_average",
            "sources": [str(p) for p in paths],
            "n_models": len(paths),
        },
    }

    print(f"\n  ⚠️  Averaged model MUST be fine-tuned before use!")
    print(f"  Run: python Bobby_Carrot/train_dqn.py --levels \"3-5\" "
          f"--episodes 2000 --resume --model-path <output>")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge individually-trained DQN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["sequential", "average"],
                        default="average",
                        help="Merge strategy: 'sequential' (use last model) "
                             "or 'average' (weight averaging)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Paths to model checkpoints (in training order)")
    parser.add_argument("--output", required=True,
                        help="Output path for merged model")
    args = parser.parse_args()

    device = torch.device("cpu")
    paths = [Path(p) for p in args.models]

    # Validate
    for p in paths:
        if not p.exists():
            print(f"ERROR: Model not found: {p}")
            sys.exit(1)

    if args.mode == "sequential":
        merged = merge_sequential(paths, device)
    else:
        merged = merge_average(paths, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    print(f"\n  ✅ Merged model saved to: {out}")
    print(f"  Size: {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
