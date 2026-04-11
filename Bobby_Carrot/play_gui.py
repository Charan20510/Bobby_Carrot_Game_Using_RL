import time
import argparse
from pathlib import Path
import sys

# Add paths so we can find Game_Python and train_dqn_fixed
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent
GAME_PYTHON_DIR = ROOT / "Game_Python"
if str(GAME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_PYTHON_DIR))

import torch
import numpy as np

# Import from both scripts
import train_dqn_fixed
import train_dqn
from bobby_carrot.rl_env import BobbyCarrotEnv, RewardConfig
from bobby_carrot.game import Map

def main():
    parser = argparse.ArgumentParser(description="Play Bobby Carrot with ANY trained DQN and GUI.")
    parser.add_argument("--level", type=int, default=8, help="Map level number to play (e.g., 7 or 8)")
    parser.add_argument("--model", type=str, default="dqn_level8_fixed_best.pt", help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for rendering")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Peek at the model to figure out architecture
    sd = torch.load(model_path, map_location=device, weights_only=False)
    if "policy" in sd:
        sd_policy = sd["policy"]
    else:
        sd_policy = sd

    # Determine architecture based on the first convolution layer
    weight_key = "conv.0.weight" if "conv.0.weight" in sd_policy else "conv.weight"
    if weight_key not in sd_policy:
        # Check standard sequential
        weight_key = next((k for k in sd_policy.keys() if k.endswith('.weight') and 'conv' in k), None)
        
    in_channels = sd_policy[weight_key].shape[1]
    
    print(f"Detected model architecture: {in_channels} input channels.")
    
    silent_rc = RewardConfig(
        step=0.0, carrot=0.0, egg=0.0, finish=0.0, death=0.0, invalid_move=0.0,
        distance_delta_scale=0.0, new_best_target_distance_scale=0.0,
        new_best_finish_distance_scale=0.0, post_collection_step_penalty=0.0,
        no_progress_penalty_after=999999, no_progress_penalty=0.0,
        no_progress_penalty_hard_after=999999, no_progress_penalty_hard=0.0,
        all_collected_bonus=0.0
    )
    
    env_visual = BobbyCarrotEnv(
        map_kind="normal",
        map_number=args.level,
        observation_mode="full",
        local_view_size=3,
        include_inventory=True,
        headless=False,
        max_steps=150,
        reward_config=silent_rc
    )

    if in_channels == 11:
        # Architecture from train_dqn_fixed.py
        agent = train_dqn_fixed.Agent(device)
        agent.load(model_path)
        agent.epsilon = 0.0
        agent.policy.eval()
        
        env_logic = train_dqn_fixed.FastEnvL8(max_steps=150)
        env_logic._map_obj = Map("normal", args.level)
        env_logic._fresh = None
        
        for ep in range(1, args.episodes + 1):
            env_logic.reset()
            env_visual.set_map("normal", args.level)
            env_visual.reset()
            
            g, v = env_logic.get_obs()
            done = False; steps = 0
            
            print(f"\n--- Starting Episode {ep} on Level {args.level} (Fixed Env) ---")
            while not done:
                env_visual.render()
                time.sleep(1.0 / args.fps)
                
                action = agent.act(g, v)
                _, done_logic, _ = env_logic.step(action)
                _, _, done_visual, _ = env_visual.step(action)
                
                g, v = env_logic.get_obs()
                steps += 1
                done = done_logic or done_visual

            env_visual.render(); time.sleep(0.5)
            print(f"Ep {ep}: {'WIN' if env_logic.finished else 'FAIL'} steps={steps}")

    else:
        # Architecture from train_dqn.py (14 or 15 channels)
        # We need to temporarily mock GRID_CHANNELS to load legacy 14-channel models safely
        original_channels = train_dqn.GRID_CHANNELS
        train_dqn.GRID_CHANNELS = in_channels
        
        cfg = train_dqn.DQNConfig()
        agent = train_dqn.DQNAgent(env_visual.action_space_n, cfg, device)
        agent.load(model_path)
        agent.epsilon = 0.0
        agent.policy_net.eval()
        
        train_dqn.GRID_CHANNELS = original_channels # Restore
        
        proxy = train_dqn.FastBobbyEnv("normal", args.level, 150)
        
        for ep in range(1, args.episodes + 1):
            env_visual.set_map("normal", args.level)
            env_visual.reset()
            
            proxy.map_info = env_visual.map_info
            proxy.bobby = env_visual.bobby
            proxy._expected_carrots = env_visual.map_info.carrot_total if env_visual.map_info else 0
            proxy._expected_eggs = env_visual.map_info.egg_total if env_visual.map_info else 0
            proxy._visited_tiles = set()
            if env_visual.map_info: 
                np.copyto(proxy._md_np, np.array(env_visual.map_info.data, dtype=np.uint8))
            if proxy.bobby:
                si = proxy.bobby.coord_src[0] + proxy.bobby.coord_src[1] * 16
                proxy._cache.refresh(proxy._md_np, 0, si)
                
            grid, inv = train_dqn._semantic_channels(proxy)
            done = False; steps = 0; info = {}
            
            print(f"\n--- Starting Episode {ep} on Level {args.level} (Standard Env, {in_channels} channels) ---")
            while not done:
                env_visual.render()
                time.sleep(1.0 / args.fps)
                
                # Trim grid channels if model expects 14 but generator makes 15
                g_tensor = torch.from_numpy(grid[:in_channels]).unsqueeze(0).to(device).float()
                v_tensor = torch.from_numpy(inv).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    action = agent.policy_net(g_tensor, v_tensor).argmax(1).item()
                
                _, _, done, info = env_visual.step(action)
                
                # Sync proxy
                proxy.map_info = env_visual.map_info
                proxy.bobby = env_visual.bobby
                if env_visual.map_info: 
                    np.copyto(proxy._md_np, np.array(env_visual.map_info.data, dtype=np.uint8))
                if proxy.bobby:
                    si = proxy.bobby.coord_src[0] + proxy.bobby.coord_src[1] * 16
                    proxy._cache.refresh(proxy._md_np, steps, si)
                    
                grid, inv = train_dqn._semantic_channels(proxy)
                steps += 1

            env_visual.render(); time.sleep(0.5)
            print(f"Ep {ep}: {'WIN' if info.get('level_completed') else 'FAIL'} steps={steps}")

    env_visual.close()

if __name__ == "__main__":
    main()
