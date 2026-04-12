# ⚡ QUICK START: L1-L25 Training + L26-L30 Testing

## ✅ CODE STATUS
**train_dqn.py: VERIFIED for L1-L30. NO ISSUES FOUND.**

All 30 normal maps exist and work correctly.

---

## 🚀 COLAB – COPY-PASTE COMMANDS

### STEP 1: Setup (Run Once)

```python
# Cell 1: Clone + Install
!git clone https://github.com/YOUR_REPO/Bobby_Carrot_Game_using_RL.git
%cd Bobby_Carrot_Game_using_RL
!pip install torch pygame numpy -q
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

```python
# Cell 2: Mount Drive + Create folders
from google.colab import drive; drive.mount('/content/drive')
import os
for L in range(1, 31): 
    os.makedirs(f"/content/drive/MyDrive/bobby_models/L{L}", exist_ok=True)
print("✓ Folders created L1-L30")
```

---

## 📊 TRAINING – L1-L5 (Pattern repeats for L6-L10, L11-L15, etc.)

### L1 – Train

```bash
%cd /content/Bobby_Carrot_Game_using_RL
!python Bobby_Carrot/train_dqn.py \
  --level 1 --episodes 2000 --warmup-eps 30 --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L1/dqn_level1.pt" \
  --report-every 100 --save-every 500
```

### L1 – Test (After training)

```bash
!python Bobby_Carrot/train_dqn.py --play --level 1 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L1/dqn_level1.pt"
```

### L2 through L5 – Same pattern (change `--level 2`, `--level 3`, etc.)

```bash
# L2
!python Bobby_Carrot/train_dqn.py --level 2 --episodes 2000 --warmup-eps 30 --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L2/dqn_level2.pt" \
  --report-every 100 --save-every 500
!python Bobby_Carrot/train_dqn.py --play --level 2 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L2/dqn_level2.pt"

# L3
!python Bobby_Carrot/train_dqn.py --level 3 --episodes 2000 --warmup-eps 30 --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L3/dqn_level3.pt" \
  --report-every 100 --save-every 500
!python Bobby_Carrot/train_dqn.py --play --level 3 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L3/dqn_level3.pt"

# L4, L5... (repeat)
```

---

## 🔄 BATCH TEMPLATE (Copy this, change level number)

For **L6-L10, L11-L15, L16-L20, L21-L25**, use this template:

```bash
# Replace X with level number (6, 7, 8, ..., 25)
!python Bobby_Carrot/train_dqn.py \
  --level X --episodes 2000 --warmup-eps 30 --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/LX/dqn_levelX.pt" \
  --report-every 100 --save-every 500

# After training, play
!python Bobby_Carrot/train_dqn.py --play --level X --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/LX/dqn_levelX.pt"
```

---

## 🧪 TESTING – L26-L30 (Unseen Levels)

**⚠️ IMPORTANT: Do NOT train L26-L30. Only test with L25 model.**

```bash
# Test L26 with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L27 with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 27 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L28 with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 28 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L29 with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 29 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L30 with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 30 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"
```

---

## 📋 SCHEDULE

| Phase | Levels | Duration |
|-------|--------|----------|
| Batch 1 | L1-L5 | ~11 hrs |
| Batch 2 | L6-L10 | ~11 hrs |
| Batch 3 | L11-L15 | ~11 hrs |
| Batch 4 | L16-L20 | ~11 hrs |
| Batch 5 | L21-L25 | ~11 hrs |
| Testing | L26-L30 | ~1 hr |
| **TOTAL** | | **~56 hrs** |

**Tip:** Run multiple notebooks in parallel to speed up.

---

## ✅ VALIDATION AFTER EACH LEVEL

After training + testing each level, check:
- [ ] Success rate > 0% by episode 500
- [ ] Loss is decreasing
- [ ] Model saved to Google Drive
- [ ] Play test completes without errors

---

## ⚠️ CRITICAL RULES

1. ❌ **DO NOT** train L26-L30 (test set only)
2. ❌ **DO NOT** share models between level trains
3. ✅ **DO** save to Google Drive
4. ✅ **DO** use separate model file per level
5. ✅ **DO** test L26-L30 with L25 model after training complete

---

## 🚨 IF ISSUES OCCUR

| Problem | Fix |
|---------|-----|
| Model not saving | Check Google Drive path exists |
| OOM error | Reduce `--n-envs 8` to `--n-envs 4` |
| Training too slow | Increase `--n-envs 8` to `--n-envs 16` |
| Model won't load | Verify CUDA available: `torch.cuda.is_available()` |

---

## 📊 EXPECTED RESULTS

| Level Group | Expected Success Rate |
|--|--|
| L1-L5 (Easy) | 60-90% |
| L6-L10 (Easy-Med) | 50-80% |
| L11-L15 (Medium) | 30-70% |
| L16-L20 (Medium-Hard) | 20-60% |
| L21-L25 (Hard) | 10-50% |
| L26-L30 (Unseen, L25 model) | 5-40% |

---

## 🎯 STATUS

✅ All levels 1-30 verified OK  
✅ train_dqn.py ready for L1-L25 training  
✅ L26-L30 test set verified  
✅ Zero issues found  
✅ **READY TO START**

---

**Go to Colab, run Setup, then train L1-L25 sequentially.**

See COLAB_L1_L25_TRAINING.md and AUDIT_L1_L25_L26_L30.md for full details.
