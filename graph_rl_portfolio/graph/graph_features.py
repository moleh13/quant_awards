import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

class GraphFeatureExtractor:
    """
    Extracts node-level features from correlation graph.
    Features include: centrality measures, clustering, communities, etc.
    """
    
    def __init__(self, graph_path=None):
        if graph_path is None:
            graph_path = os.path.join(os.path.dirname(__file__), '../data_cache/correlation_graph.pkl')
        
        import pickle
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        self.features_df = None
    
    def extract_all_features(self):
        """Extract all node-level features"""
        nodes = list(self.graph.nodes())
        features = {}
        
        # 1. Basic centrality measures
        print("Computing centrality measures...")
        features['degree'] = dict(self.graph.degree())
        features['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
        features['closeness_centrality'] = nx.closeness_centrality(self.graph)
        features['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(self.graph)
        
        # 2. Clustering and local structure
        print("Computing clustering coefficients...")
        features['clustering_coefficient'] = nx.clustering(self.graph)
        features['triangles'] = nx.triangles(self.graph)
        
        # 3. PageRank
        print("Computing PageRank...")
        features['pagerank'] = nx.pagerank(self.graph)
        
        # 4. Community detection
        print("Detecting communities...")
        features['community'] = self._detect_communities()
        
        # 5. Additional structural features
        print("Computing structural features...")
        features.update(self._compute_structural_features())
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(features)
        self.features_df.index.name = 'asset'
        
        return self.features_df
    
    def _detect_communities(self):
        """Detect communities using spectral clustering"""
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        
        # Use spectral clustering
        n_clusters = min(4, len(self.graph.nodes()))  # Max 4 communities
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        
        # Use absolute correlation as similarity
        similarity_matrix = np.abs(adj_matrix)
        communities = clustering.fit_predict(similarity_matrix)
        
        return dict(zip(self.graph.nodes(), communities))
    
    def _compute_structural_features(self):
        """Compute additional structural features"""
        features = {}
        
        # Average neighbor degree
        features['avg_neighbor_degree'] = nx.average_neighbor_degree(self.graph)
        
        # Core number (k-core decomposition)
        features['core_number'] = nx.core_number(self.graph)
        
        # Local efficiency
        features['local_efficiency'] = nx.local_efficiency(self.graph)
        
        # Load centrality
        features['load_centrality'] = nx.load_centrality(self.graph)
        
        # Harmonic centrality
        features['harmonic_centrality'] = nx.harmonic_centrality(self.graph)
        
        return features
    
    def normalize_features(self, method='standard'):
        """Normalize features for ML models"""
        if self.features_df is None:
            raise ValueError("Features not extracted yet. Call extract_all_features() first.")
        
        if method == 'standard':
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(self.features_df)
            normalized_df = pd.DataFrame(
                normalized_features, 
                columns=self.features_df.columns,
                index=self.features_df.index
            )
        else:
            normalized_df = self.features_df.copy()
        
        return normalized_df
    
    def save_features(self, output_path=None, normalized=False):
        """Save features to CSV"""
        if output_path is None:
            if normalized:
                output_path = os.path.join(os.path.dirname(__file__), '../data_cache/graph_features_normalized.csv')
            else:
                output_path = os.path.join(os.path.dirname(__file__), '../data_cache/graph_features.csv')
        
        if normalized:
            df_to_save = self.normalize_features()
        else:
            df_to_save = self.features_df
        
        df_to_save.to_csv(output_path)
        print(f"Features saved to {output_path}")
    
    def get_feature_summary(self):
        """Get summary statistics of features"""
        if self.features_df is None:
            raise ValueError("Features not extracted yet. Call extract_all_features() first.")
        
        summary = {
            'num_nodes': len(self.features_df),
            'num_features': len(self.features_df.columns),
            'feature_names': list(self.features_df.columns),
            'summary_stats': self.features_df.describe()
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    extractor = GraphFeatureExtractor()
    features_df = extractor.extract_all_features()
    
    print(f"\nExtracted {len(features_df.columns)} features for {len(features_df)} nodes")
    print(f"Feature names: {list(features_df.columns)}")
    
    # Show summary
    summary = extractor.get_feature_summary()
    print(f"\nSummary statistics:")
    print(summary['summary_stats'])
    
    # Save features
    extractor.save_features()  # Raw features
    extractor.save_features(normalized=True)  # Normalized features
    
    # Show top assets by different metrics
    print(f"\nTop 5 assets by degree centrality:")
    print(features_df['degree'].nlargest(5))
    
    print(f"\nTop 5 assets by betweenness centrality:")
    print(features_df['betweenness_centrality'].nlargest(5))
    
    print(f"\nTop 5 assets by PageRank:")
    print(features_df['pagerank'].nlargest(5)) 