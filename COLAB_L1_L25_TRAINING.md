# Train L3-L25 (Multi-Level) + Test L26-L30 — Colab Guide

## 📊 TRAINING APPROACH

```
Training Strategy:  Multi-level (one model for all levels)
Training Set:       Levels 3-25 (23 levels)
Q-Learning Set:     Levels 1-2 (separate, tabular Q-learning)
Test Set:           Levels 26-30 (5 unseen levels)
Model:              Single DQN model (dqn_multi_3_25.pt)
Goal:               >40% success on unseen L26-30
```

### Why Multi-Level Instead of Per-Level?

| Per-Level (OLD) | Multi-Level (NEW) |
|-----------------|-------------------|
| 25 separate models | 1 unified model |
| Each model memorizes one map | Model learns generic strategies |
| 0% success on unseen levels | >40% success on L26-30 expected |
| ~56 hours total | ~8-10 hours total |

---

## 🚀 COLAB SETUP (Run Once)

```python
# Cell 1: Clone + Install
!git clone https://github.com/YOUR_REPO/Bobby_Carrot_Game_using_RL.git
%cd Bobby_Carrot_Game_using_RL
!pip install torch pygame numpy -q
import torch
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

```python
# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs("/content/drive/MyDrive/bobby_models", exist_ok=True)
print("✓ Drive mounted and folder created")
```

---

## 🎯 PHASE 1: Train L1-L2 (Q-Learning)

These simple levels use tabular Q-learning (no DQN needed):

```bash
# Level 1
!python Bobby_Carrot/train_q_learning.py \
  --map-number 1 --episodes 10000 \
  --model-path "/content/drive/MyDrive/bobby_models/q_table_L1.pkl"

# Level 2
!python Bobby_Carrot/train_q_learning.py \
  --map-number 2 --episodes 10000 \
  --model-path "/content/drive/MyDrive/bobby_models/q_table_L2.pkl"
```

---

## 🎯 PHASE 2: Train L3-L25 (Multi-Level DQN)

### Initial Training (10,000 episodes)

```bash
%cd /content/Bobby_Carrot_Game_using_RL
!python Bobby_Carrot/train_dqn.py \
  --levels 3-25 \
  --episodes 10000 \
  --warmup-eps 80 \
  --n-envs 8 \
  --eps-decay 0.9995 \
  --lr 3e-4 \
  --batch-size 512 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt" \
  --report-every 200 \
  --save-every 1000
```

**Expected output:**
```
Device: cuda | Tesla T4
Multi-level training: 23 levels (3-25) | n_envs=8

=== Training L3-25 for 10000 episodes (ep 0->10000) ===
eps: 1.00 -> 0.08 (decay=0.9995/ep) | warmup=80 | grad_every=4 | batch=512
[L3-25] ep=  200/10000 | collected=25.3% | success= 0.0% | eps=0.941 | ...
[L3-25] ep=  400/10000 | collected=38.1% | success= 0.0% | eps=0.886 | ...
...
[L3-25] ep= 5000/10000 | collected=75.4% | success=18.5% | eps=0.082 | ...
...
[L3-25] ep=10000/10000 | collected=88.3% | success=42.1% | eps=0.080 | ...
```

### Resume Training (After Colab Disconnect)

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

This loads the checkpoint and trains for 5000 MORE episodes (doesn't restart from 0).

### Extended Training (If Success Rate Needs Improvement)

```bash
# Train 5000 more episodes on top of existing
!python Bobby_Carrot/train_dqn.py \
  --levels 3-25 \
  --episodes 5000 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt" \
  --resume \
  --report-every 200 --save-every 1000
```

---

## 🧪 PHASE 3: Test L26-L30 (Unseen Levels)

### Batch Test (All 5 Levels at Once)

```bash
!python Bobby_Carrot/train_dqn.py --play \
  --levels 26-30 \
  --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt"
```

**Expected output:**
```
--- Level 26 ---
Ep   1: WIN  | steps= 342 | collected=65/65
...
Level 26 success: 9/20 = 45.0%

--- Level 27 ---
...
Level 27 success: 10/20 = 50.0%

...

Overall success: 43/100 = 43.0%
```

### Test Individual Levels (Optional)

```bash
# Test just L26
!python Bobby_Carrot/train_dqn.py --play \
  --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt"
```

### Visual Test (GUI Mode)

```bash
# Watch the agent play L26 visually
!python Bobby_Carrot/train_dqn.py --play-gui \
  --level 26 \
  --model-path "/content/drive/MyDrive/bobby_models/dqn_multi_3_25.pt"
```

---

## ⏱️ TIMELINE (T4 GPU)

| Phase | What | Duration |
|-------|------|----------|
| Setup | Clone + Install + Mount | ~5 min |
| Phase 1 | L1-L2 Q-learning | ~30 min |
| Phase 2 | L3-25 Multi-level DQN (10K eps) | ~6-8 hours |
| Phase 3 | L26-30 Testing | ~10 min |
| **TOTAL** | | **~7-9 hours** |

**Compare:** Previous per-level approach = ~56 hours. **7x faster!**

---

## ✅ VALIDATION CHECKLIST

### Before Training
- [ ] Git repo cloned
- [ ] PyTorch with CUDA working
- [ ] Google Drive mounted
- [ ] Model folder created

### During Training
- [ ] Loss is decreasing
- [ ] Collected rate climbing (should reach 80%+ by ep 5000)
- [ ] Success rate appearing (should appear by ep 2000)
- [ ] Model saving to Drive every 1000 episodes
- [ ] No OOM errors

### After Testing
- [ ] L26-L30 evaluations complete
- [ ] Per-level success rates recorded
- [ ] Overall success rate >40%

---

## 🚨 TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| OOM error | Reduce `--n-envs 4` or `--batch-size 256` |
| Training too slow | Increase `--n-envs 16` |
| Colab disconnect | Use `--resume` to continue (model saved to Drive) |
| Success rate stuck at 0% | Check loss is decreasing; try `--lr 1e-4` |
| Model won't load | Ensure correct `--model-path` to Drive |

---

## ⚠️ CRITICAL RULES

1. **DO** use `--levels 3-25` for multi-level DQN training
2. **DO** save model to Google Drive
3. **DO** use `--resume` after disconnects
4. **DO NOT** train on L26-L30 (test set only)
5. **DO NOT** use per-level models for generalization testing
6. **DO NOT** change `GRID_CHANNELS` or `INV_FEATURES` (breaks saved models)

---

**Status:** PRODUCTION-READY for multi-level L3-25 training + L26-30 testing ✓
