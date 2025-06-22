"""
Train Graph-Aware RL Agent (PPO) on PortfolioEnv

This script trains a PPO agent with a custom policy that uses GNN embeddings
as part of its observation space.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stable_baselines3 import PPO
from environment.portfolio_env import PortfolioEnv
from agents.graph_policy import GnnFeatureExtractor


def main():
    # --- Configuration ---
    total_timesteps = 20_000  # Longer training time
    model_save_path = 'models/graph_ppo'
    embedding_path = 'data_cache/gnn_embeddings.csv'
    asset_list_path = 'data_cache/assets_2019.csv'
    embedding_dim = 128
    
    os.makedirs(model_save_path, exist_ok=True)

    # --- Environment ---
    env = PortfolioEnv(asset_list_path=asset_list_path)

    # --- Policy Kwargs for Custom Feature Extractor ---
    policy_kwargs = {
        "features_extractor_class": GnnFeatureExtractor,
        "features_extractor_kwargs": {
            "embedding_path": embedding_path,
            "embedding_dim": embedding_dim,
            "asset_list_path": asset_list_path,
        },
        "net_arch": [256, 256], # Deeper network for more complex features
        "activation_fn": torch.nn.ReLU,
    }

    # --- PPO Agent ---
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=1e-4,  # Slightly higher learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="tensorboard/graph_ppo/"
    )

    # --- Train the Agent ---
    print("Starting training for Graph-Aware PPO model...")
    try:
        model.learn(total_timesteps=total_timesteps)
        print("Training completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(model_save_path, 'ppo_graph_final')
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}.zip")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 