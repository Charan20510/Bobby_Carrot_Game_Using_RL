# Train L1-L25, Test L26-L30 in Colab

## ✅ VERIFICATION CHECKLIST

All 30 normal level maps exist:
```
✓ normal01.blm - normal30.blm (30 files total)
✓ Map loading: works for all levels
✓ Train_dqn.py: supports levels 1-30 (no hardcoded limits)
✓ Auto max_steps: dynamically computed per level
✓ No issues found
```

---

## 📊 TRAINING PLAN

```
========================================
Training Set:   normal01 to normal25
Testing Set:    normal26 to normal30
========================================

Training: Levels 1-25 (5 groups of 5 levels each)
Testing:  Levels 26-30 (unseen, 5 levels)
```

---

## 🚀 COLAB - SETUP (RUN ONCE)

```python
# Cell 1: Clone & Install
!git clone https://github.com/YOUR_REPO/Bobby_Carrot_Game_using_RL.git
%cd Bobby_Carrot_Game_using_RL
!pip install torch pygame numpy -q

import torch
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

```python
# Cell 2: Mount Drive + Create folders
from google.colab import drive
drive.mount('/content/drive')

import os
for level in range(1, 31):
    os.makedirs(f"/content/drive/MyDrive/bobby_models/L{level}", exist_ok=True)
    print(f"✓ L{level}")
```

---

## 🎯 TRAINING LEVELS 1-5 (BATCH 1)

Each level trains independently. Repeat pattern for L6-L10, L11-L15, L16-L20, L21-L25.

### Level 1 – Train

```bash
%cd /content/Bobby_Carrot_Game_using_RL
!python Bobby_Carrot/train_dqn.py \
  --level 1 \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L1/dqn_level1.pt" \
  --report-every 100 \
  --save-every 500
```

### Level 1 – Test

```bash
!python Bobby_Carrot/train_dqn.py \
  --play \
  --level 1 \
  --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L1/dqn_level1.pt"
```

---

### Level 2 – Train + Test

```bash
!python Bobby_Carrot/train_dqn.py \
  --level 2 \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L2/dqn_level2.pt" \
  --report-every 100 --save-every 500

!python Bobby_Carrot/train_dqn.py --play --level 2 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L2/dqn_level2.pt"
```

### Level 3 – Train + Test

```bash
!python Bobby_Carrot/train_dqn.py \
  --level 3 \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L3/dqn_level3.pt" \
  --report-every 100 --save-every 500

!python Bobby_Carrot/train_dqn.py --play --level 3 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L3/dqn_level3.pt"
```

### Level 4 – Train + Test

```bash
!python Bobby_Carrot/train_dqn.py \
  --level 4 \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L4/dqn_level4.pt" \
  --report-every 100 --save-every 500

!python Bobby_Carrot/train_dqn.py --play --level 4 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L4/dqn_level4.pt"
```

### Level 5 – Train + Test

```bash
!python Bobby_Carrot/train_dqn.py \
  --level 5 \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/L5/dqn_level5.pt" \
  --report-every 100 --save-every 500

!python Bobby_Carrot/train_dqn.py --play --level 5 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L5/dqn_level5.pt"
```

---

## 📋 REPEAT FOR LEVELS 6-10, 11-15, 16-20, 21-25

Use the same command pattern. **Template:**

```bash
# Replace X with level number (6, 7, 8, ... 25)
!python Bobby_Carrot/train_dqn.py \
  --level X \
  --episodes 2000 \
  --warmup-eps 30 \
  --n-envs 8 \
  --model-path "/content/drive/MyDrive/bobby_models/LX/dqn_levelX.pt" \
  --report-every 100 --save-every 500

!python Bobby_Carrot/train_dqn.py --play --level X --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/LX/dqn_levelX.pt"
```

---

## 🧪 TESTING UNSEEN LEVELS 26-30 (After L1-L25 Trained)

**Important:** Do NOT train on L26-L30. Use trained L25 model.

```bash
# Test L26 (unseen) with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L27 (unseen) with L25 model
!python Bobby_Carrot/train_dqn.py --play --level 27 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L28 (unseen)
!python Bobby_Carrot/train_dqn.py --play --level 28 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L29 (unseen)
!python Bobby_Carrot/train_dqn.py --play --level 29 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"

# Test L30 (unseen)
!python Bobby_Carrot/train_dqn.py --play --level 30 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"
```

---

## 💡 ALTERNATIVE: Test with each level's own model (Optional)

If you want to see how each trained model performs on unseen levels:

```bash
# L26 tested with L1 model
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L1/dqn_level1.pt"

# L26 tested with L5 model
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L5/dqn_level5.pt"

# L26 tested with L15 model
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L15/dqn_level15.pt"

# L26 tested with L25 model (best trained)
!python Bobby_Carrot/train_dqn.py --play --level 26 --play-episodes 20 \
  --model-path "/content/drive/MyDrive/bobby_models/L25/dqn_level25.pt"
```

---

## 📊 SUMMARY RESULTS (After All Training/Testing)

```python
# Cell: Collect all results
import os

print("=== TRAINING RESULTS ===")
for level in range(1, 26):
    model_path = f"/content/drive/MyDrive/bobby_models/L{level}/dqn_level{level}.pt"
    exists = os.path.exists(model_path)
    status = "✓ Trained" if exists else "✗ Missing"
    print(f"L{level:2d}: {status}")

print("\n=== TESTING RESULTS (L26-L30 with L25 model) ===")
print("(Run test commands above to generate scores)")
print("Expected: Success rates vary by level difficulty")
```

---

## ⏱️ TIMELINE (T4 GPU)

| Phase | Levels | Time per Level | Total |
|-------|--------|---|---|
| Batch 1 | L1-L5 | 2 hrs train + 10 min test | ~11 hours |
| Batch 2 | L6-L10 | 2 hrs train + 10 min test | ~11 hours |
| Batch 3 | L11-L15 | 2 hrs train + 10 min test | ~11 hours |
| Batch 4 | L16-L20 | 2 hrs train + 10 min test | ~11 hours |
| Batch 5 | L21-L25 | 2 hrs train + 10 min test | ~11 hours |
| **Testing** | **L26-L30** | **10 min each** | **~1 hour** |
| **TOTAL** | | | **~56 hours** |

Run multiple Colab notebooks in parallel to speed up (e.g., one for L1-L5, one for L6-L10, etc.).

---

## ✅ VALIDATION CHECKS

### Before Training:
- [ ] Git repo cloned
- [ ] PyTorch with CUDA working
- [ ] Google Drive mounted
- [ ] Folders created for L1-L30
- [ ] Normal01-30.blm files verified

### During Training:
- [ ] Success rate > 0% by episode 500 on each level
- [ ] Loss decreasing gradually
- [ ] Models saving to Drive every 500 episodes
- [ ] No OOM errors

### After Testing:
- [ ] L26-L30 models load without errors
- [ ] Play episodes complete without crashes
- [ ] Success rates recorded

---

## 🚨 IMPORTANT RULES

1. **Each level gets ONE trained model** (L1.pt, L2.pt, ..., L25.pt)
2. **DO NOT share models between training levels** (train L1 fresh, not with L25)
3. **Test L26-30 ONLY with trained models** (e.g., L25 model)
4. **DO NOT train on L26-L30** (they are test set only)
5. **Save everything to Google Drive** (Colab = temporary storage)

---

## 🎯 EXPECTED OUTCOMES

**Training (L1-L25):**
- Easy levels (L1-L5): Success rate 60-90% after 2000 episodes
- Medium levels (L6-L15): Success rate 30-70% after 2000 episodes
- Hard levels (L16-L25): Success rate 10-50% after 2000 episodes

**Testing (L26-L30 with L25 model):**
- Generalization: 5-40% success rate (unseen levels are harder)
- Model evaluation: How well does training 1-25 transfer to 26-30?

---

## Next Step

1. Run Colab setup (Cell 1 & 2)
2. Start Batch 1 (L1-L5)
3. Monitor progress
4. Repeat Batch 2-5
5. Run Testing (L26-L30)
6. Collect results

**Status:** PROD-READY for L1-25 training + L26-30 testing ✓
