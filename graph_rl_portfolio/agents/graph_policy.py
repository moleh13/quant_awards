"""
Custom Policy for Graph-Aware RL Agent

This module defines a custom feature extractor that integrates pre-trained GNN
embeddings into the observation space for the RL agent.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import pandas as pd
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GnnFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines GNN embeddings with other features.

    :param observation_space: The observation space of the environment.
    :param embedding_path: Path to the GNN embeddings CSV file.
    :param embedding_dim: The dimension of the GNN embeddings.
    :param asset_list_path: Path to the asset list to ensure order consistency.
    """
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 embedding_path: str,
                 embedding_dim: int,
                 asset_list_path: str):
        
        # Calculate the total feature dimension
        # assets * (price_features + embedding_dim) + cash + positions + signals
        n_assets = observation_space['features'].shape[0]
        n_features = observation_space['features'].shape[1]
        n_signals = observation_space['signals'].shape[0]
        
        features_dim = n_assets * (n_features + embedding_dim) + 1 + n_assets + n_signals
        
        super().__init__(observation_space, features_dim=features_dim)

        # --- Load and align GNN embeddings ---
        print("Loading GNN embeddings...")
        
        # --- Correct paths to be relative to the project root ---
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corrected_asset_list_path = os.path.join(base_path, asset_list_path)
        corrected_embedding_path = os.path.join(base_path, embedding_path)
        
        # Load assets from the environment's list
        env_assets = pd.read_csv(corrected_asset_list_path)['asset'].tolist()
        
        # Load embeddings from CSV
        embedding_df = pd.read_csv(corrected_embedding_path)
        
        # Ensure the 'asset' column exists
        if 'asset' not in embedding_df.columns:
            # The saved embeddings from our script have assets as the index, not a column
            # Let's reset the index to create the 'asset' column
            embedding_df = embedding_df.rename(columns={'Unnamed: 0': 'asset'})

        # Align embeddings to the order of assets in the environment
        embedding_df = embedding_df.set_index('asset').reindex(env_assets).reset_index()
        
        # Check for any assets that might be missing from the embedding file
        if embedding_df.isnull().values.any():
            missing_assets = embedding_df[embedding_df.isnull().any(axis=1)]['asset'].tolist()
            print(f"Warning: No GNN embeddings found for the following assets: {missing_assets}")
            embedding_df = embedding_df.fillna(0) # Fill missing with zeros
            
        # Convert to tensor
        self.gnn_embeddings = torch.tensor(
            embedding_df.drop('asset', axis=1).values,
            dtype=torch.float32
        )
        
        print("GNN embeddings loaded and aligned successfully.")


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: The observation from the environment.
            
        Returns:
            The processed features.
        """
        batch_size = observations["features"].shape[0]

        # Expand GNN embeddings to match the batch size
        gnn_embeddings_batch = self.gnn_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_embeddings_batch = gnn_embeddings_batch.to(observations["features"].device)

        # Concatenate asset features with their corresponding GNN embeddings
        # Shape: (batch_size, n_assets, n_features + embedding_dim)
        combined_asset_features = torch.cat([observations["features"], gnn_embeddings_batch], dim=2)
        
        # Flatten the combined asset features to a single vector per batch item
        flattened_asset_features = combined_asset_features.reshape(batch_size, -1)

        # Concatenate with the other parts of the observation (cash, positions, signals)
        # to form the final feature vector for the policy network.
        final_features = torch.cat([
            flattened_asset_features,
            observations["cash"],
            observations["positions"],
            observations["signals"]
        ], dim=1)

        return final_features 