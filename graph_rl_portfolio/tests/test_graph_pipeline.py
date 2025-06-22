import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import torch

CACHE_DIR = os.path.join(os.path.dirname(__file__), '../data_cache')

# File paths to test
GRAPH_PATH = os.path.join(CACHE_DIR, 'correlation_graph.pkl')
FEATURES_PATH = os.path.join(CACHE_DIR, 'graph_features.csv')
FEATURES_NORM_PATH = os.path.join(CACHE_DIR, 'graph_features_normalized.csv')
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'gnn_embeddings.csv')
MODEL_PATH = os.path.join(CACHE_DIR, 'gnn_model.pth')

def test_files_exist():
    """Test that all graph pipeline files exist"""
    files = [GRAPH_PATH, FEATURES_PATH, FEATURES_NORM_PATH, EMBEDDINGS_PATH, MODEL_PATH]
    for file_path in files:
        assert os.path.exists(file_path), f"Missing file: {file_path}"
    print("✅ All graph pipeline files exist")

def test_graph_structure():
    """Test graph structure and properties"""
    with open(GRAPH_PATH, 'rb') as f:
        graph = pickle.load(f)
    
    # Basic graph properties
    assert isinstance(graph, nx.Graph), "Graph should be NetworkX Graph"
    assert graph.number_of_nodes() > 0, "Graph should have nodes"
    assert graph.number_of_edges() > 0, "Graph should have edges"
    
    # Check edge weights
    for u, v, data in graph.edges(data=True):
        assert 'weight' in data, f"Edge {u}-{v} missing weight"
        assert 0 <= data['weight'] <= 1, f"Edge weight {data['weight']} not in [0,1]"
    
    print(f"✅ Graph structure valid: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

def test_features_consistency():
    """Test feature files consistency"""
    # Load features
    features = pd.read_csv(FEATURES_PATH, index_col='asset')
    features_norm = pd.read_csv(FEATURES_NORM_PATH, index_col='asset')
    
    # Check dimensions
    assert features.shape[0] > 0, "Features should have rows"
    assert features.shape[1] > 0, "Features should have columns"
    assert features.shape == features_norm.shape, "Raw and normalized features should have same shape"
    
    # Check for NaNs
    assert not features.isna().any().any(), "Raw features should not contain NaNs"
    assert not features_norm.isna().any().any(), "Normalized features should not contain NaNs"
    
    # Check that normalized features are actually normalized (mean close to 0, std close to 1)
    for col in features_norm.columns:
        if col not in ['community', 'triangles']:  # Skip categorical and integer features
            mean_val = features_norm[col].mean()
            std_val = features_norm[col].std()
            assert abs(mean_val) < 1e-6, f"Normalized feature {col} mean should be ~0, got {mean_val}"
            # Skip std check for features with zero variance (constant features)
            if std_val > 1e-10:  # Only check std if feature has some variance
                assert 0.9 < std_val < 1.1, f"Normalized feature {col} std should be ~1, got {std_val}"
    
    print(f"✅ Features consistent: {features.shape[0]} nodes, {features.shape[1]} features")

def test_embeddings_quality():
    """Test GNN embeddings quality and dimensions"""
    embeddings = pd.read_csv(EMBEDDINGS_PATH, index_col=0)  # Use first column as index
    
    # Check dimensions
    expected_dim = 128
    assert embeddings.shape[1] == expected_dim, f"Embeddings should have {expected_dim} dimensions, got {embeddings.shape[1]}"
    assert embeddings.shape[0] > 0, "Embeddings should have nodes"
    
    # Check for NaNs
    assert not embeddings.isna().any().any(), "Embeddings should not contain NaNs"
    
    # Check that embeddings are not all identical
    for col in embeddings.columns:
        assert embeddings[col].std() > 0, f"Embedding dimension {col} has zero variance"
    
    # Check embedding statistics
    for col in embeddings.columns:
        assert not np.isinf(embeddings[col]).any(), f"Embedding dimension {col} contains infinite values"
    
    print(f"✅ Embeddings valid: {embeddings.shape[0]} nodes, {embeddings.shape[1]} dimensions")

def test_model_loading():
    """Test that the trained model can be loaded"""
    assert torch.load(MODEL_PATH, map_location='cpu'), "Model should be loadable"
    print("✅ Model can be loaded successfully")

def test_pipeline_consistency():
    """Test consistency across the entire pipeline"""
    # Load all components
    with open(GRAPH_PATH, 'rb') as f:
        graph = pickle.load(f)
    
    features = pd.read_csv(FEATURES_PATH, index_col='asset')
    embeddings = pd.read_csv(EMBEDDINGS_PATH, index_col=0)  # Use first column as index
    
    # Check that all components have the same nodes
    graph_nodes = set(graph.nodes())
    feature_nodes = set(features.index)
    embedding_nodes = set(embeddings.index)
    
    assert graph_nodes == feature_nodes, "Graph and features should have same nodes"
    assert feature_nodes == embedding_nodes, "Features and embeddings should have same nodes"
    
    print(f"✅ Pipeline consistency: {len(graph_nodes)} nodes across all components")

def run_all_tests():
    """Run all graph pipeline tests"""
    print("Running graph pipeline tests...")
    
    test_files_exist()
    test_graph_structure()
    test_features_consistency()
    test_embeddings_quality()
    test_model_loading()
    test_pipeline_consistency()
    
    print("\n🎉 All graph pipeline tests passed!")

if __name__ == "__main__":
    run_all_tests() 