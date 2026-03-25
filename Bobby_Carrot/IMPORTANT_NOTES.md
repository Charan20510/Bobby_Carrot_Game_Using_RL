# Important RL Notes

## Stable files to keep
- `train_q_learning.py` (main trainer)
- `Game_Python/bobby_carrot/rl_env.py` (environment)
- `q_table_level1_inventory.pkl` (best level-1 model: 100% clear + 100% completion in eval)
- `q_table_bobby.pkl` (optional fallback / legacy default model path)

## Key run command
```powershell
.venv\Scripts\python.exe Bobby_Carrot\train_q_learning.py --episodes 5000 --map-number 1 --no-curriculum --observation-mode local --local-view-size 3 --max-steps 500 --model-path Bobby_Carrot\q_table_level1_fixed.pkl
```

## Recommended working command (phase-aware state)
```powershell
.venv\Scripts\python.exe Bobby_Carrot\train_q_learning.py --episodes 800 --map-number 1 --no-curriculum --observation-mode local --local-view-size 3 --max-steps 500 --report-every 200 --model-path Bobby_Carrot\q_table_level1_inventory.pkl
```

## Evaluate command
```powershell
.venv\Scripts\python.exe Bobby_Carrot\train_q_learning.py --eval --episodes 20 --map-number 1 --observation-mode local --local-view-size 3 --max-steps 500 --model-path Bobby_Carrot\q_table_level1_inventory.pkl
```

## Current behavior
- Agent reliably learns carrot collection and finish behavior on level 1.
- Verified eval: all_collected_rate=100%, success_rate=100%, mean_steps=27.

## Cleanup status
- Temporary debug scripts were removed.
- Temporary training checkpoints can be safely deleted if they are not needed for comparison.
