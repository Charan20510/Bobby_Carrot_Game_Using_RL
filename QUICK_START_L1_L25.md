# ⚡ QUICK START: Train L3-L25 (Multi-Level) + Test L26-L30

## ✅ TRAINING APPROACH

**One model trained on levels 3-25 → test generalization on unseen levels 26-30.**

- L1-L2: Trained separately with `train_q_learning.py` (tabular Q-learning)
- L3-L25: Trained together with `train_dqn.py` using `--levels 3-25` (one DQN model)
- L26-L30: **Never trained** — evaluation only with the L3-25 model

### Key Design (No Hardcoding / No Memorization)
- Observations: current tile types, BFS gradients, agent position, visited mask — all computed dynamically
- Each episode picks a **random level** from the pool → model learns GENERIC navigation
- No level number in observations → model can't memorize specific maps
- BFS gradient guides toward nearest target regardless of map layout

---

## 🚀 COLAB – COPY-PASTE COMMANDS

### STEP 1: Setup (Run Once)

```python
# Cell 1: Clone + Install
!git clone https://github.com/YOUR_REPO/Bobby_Carrot_Game_using_RL.git
%cd Bobby_Carrot_Game_using_RL
!pip install torch pygame numpy -q
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

```python
# Cell 2: Mount Drive
from google.colab import drive; drive.mount('/content/drive')
import os
os.makedirs("/content/drive/MyDrive/bobby_models", exist_ok=True)
print("✓ Drive mounted")
```

---

### STEP 2: Train L3-L25 (Multi-Level, Single Model)

```bash
%cd /content/Bobby_Carrot_Game_using_RL
!python Bobby_Carrot/train_dqn.py \
  --levels 3-25 \
  --episodes 10000 \
  --warmup-eps 80 \
  --n-envs 8 \
  --eps-decay 0.9995 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt" \
  --report-every 200 --save-every 1000
```

**What this does:** Trains one model across ALL 23 levels. Each episode picks a random level. The model learns general strategies for crumble tiles, conveyors, arrows, keys, and switches.

**Resume if Colab disconnects:**
```bash
!python Bobby_Carrot/train_dqn.py \
  --levels 3-25 \
  --episodes 5000 \
  --warmup-eps 0 \
  --n-envs 8 \
  --eps-decay 0.9995 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt" \
  --resume \
  --report-every 200 --save-every 1000
```

---

### STEP 3: Test L26-L30 (Unseen Levels)

```bash
# Test on ALL unseen levels with one command
!python Bobby_Carrot/train_dqn.py --play \
  --levels 26-30 \
  --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt"
```

This evaluates the model on each of L26, L27, L28, L29, L30 (20 episodes each) and prints per-level + overall success rate.

---

## 📊 EXPECTED RESULTS

| Phase | Target | Expected |
|-------|--------|----------|
| Training (L3-25, 10K eps) | collected >90% | ✓ After ~5000 eps |
| Training (L3-25, 10K eps) | success >40% | ✓ After ~8000 eps |
| Testing (L26-30, unseen) | success >40% | ✓ With good multi-level training |

---

## ⚠️ CRITICAL RULES

1. ✅ **DO** use `--levels 3-25` for multi-level training
2. ✅ **DO** save to Google Drive (Colab is temporary)
3. ❌ **DO NOT** train on L26-L30 (test set only)
4. ❌ **DO NOT** train per-level models for generalization testing

---

## 🚨 IF TRAINING IS SLOW

```bash
# Reduce environments if OOM
--n-envs 4

# Or increase for faster training
--n-envs 16

# Reduce batch size if GPU memory issues
--batch-size 256
```

---

## 🎯 WHAT CHANGED FROM PREVIOUS VERSION

| Before | After |
|--------|-------|
| One model per level (25 models) | One model for ALL levels |
| Model memorizes specific map | Model learns GENERIC strategies |
| ~0% success on unseen levels | >40% expected on L26-30 |
| Crumble tiles: weak penalty (-3) | Strong penalty (-10) + early termination |
| No exit reachability check | Exits immediately if exit blocked |

See [AUDIT_L1_L25_L26_L30.md](AUDIT_L1_L25_L26_L30.md) for technical details.
