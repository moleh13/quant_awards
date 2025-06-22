"""
Train Baseline RL Agent (PPO) on PortfolioEnv

This script trains a baseline PPO agent with a simple MLP policy on the custom
PortfolioEnv. The agent does not use graph or regime/Kalman features in the policy
network, but these features are present in the observation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.portfolio_env import PortfolioEnv


def main():
    # Training parameters - reduced for testing
    total_timesteps = 1_000  # Very small for testing
    save_path = '../models/baseline_ppo'
    os.makedirs(save_path, exist_ok=True)

    # Create environment directly (not vectorized)
    env = PortfolioEnv(asset_list_path='data_cache/assets_2019.csv')

    # Test environment first
    print("Testing environment...")
    obs, _ = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
            print(f"{key} min: {np.min(value)}, max: {np.max(value)}")
            print(f"{key} has NaN: {np.any(np.isnan(value))}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}")
        if done:
            break

    # Try with a much simpler policy and add action clipping
    policy_kwargs = dict(
        net_arch=[32],  # Very small network
        activation_fn=torch.nn.Tanh,  # Use Tanh instead of ReLU
    )

    # Instantiate PPO agent with very conservative settings
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=1e-5,  # Very low learning rate
        n_steps=64,  # Very small batch
        batch_size=16,  # Very small batch
        n_epochs=2,  # Very few epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Smaller clip range
        ent_coef=0.1,  # More exploration
        vf_coef=0.5,
        max_grad_norm=0.1,  # Smaller gradient norm
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="../tensorboard/baseline_ppo/"
    )

    # Train
    print("Starting training...")
    try:
        model.learn(total_timesteps=total_timesteps)
        print("Training completed successfully!")
        
        # Save final model
        model.save(os.path.join(save_path, 'ppo_baseline_final'))
        print(f"Model saved to {save_path}/ppo_baseline_final.zip")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 