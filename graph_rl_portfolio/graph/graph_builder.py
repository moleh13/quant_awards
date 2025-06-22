import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

class CorrelationGraphBuilder:
    """
    Builds correlation-based graph from asset returns.
    - Rolling 30-day correlation window
    - Correlation threshold: |correlation| > 0.35
    - Uses log returns
    """
    
    def __init__(self, data_path=None, correlation_threshold=0.35, window_days=30):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '../data_cache/preprocessed_all.csv')
        self.data = pd.read_csv(data_path, parse_dates=['datetime'])
        self.correlation_threshold = correlation_threshold
        self.window_days = window_days
        self.graph = None
        
    def build_graph(self, start_date=None, end_date=None):
        """Build correlation graph for the specified date range"""
        # Filter data by date range
        if start_date is None:
            start_date = self.data['datetime'].min()
        if end_date is None:
            end_date = self.data['datetime'].max()
            
        mask = (self.data['datetime'] >= start_date) & (self.data['datetime'] <= end_date)
        filtered_data = self.data[mask].copy()
        
        # Pivot to get log returns matrix (assets as columns, dates as rows)
        returns_matrix = filtered_data.pivot(index='datetime', columns='asset', values='log_return')
        
        # Remove assets with too much missing data
        min_obs = len(returns_matrix) * 0.8  # At least 80% of observations
        returns_matrix = returns_matrix.dropna(axis=1, thresh=min_obs)
        
        # Forward fill remaining NaNs
        returns_matrix = returns_matrix.fillna(method='ffill')
        
        # Compute rolling correlation matrix
        rolling_corr = returns_matrix.rolling(window=self.window_days, min_periods=self.window_days//2).corr()
        
        # Get the latest correlation matrix
        latest_corr = rolling_corr.xs(rolling_corr.index.get_level_values(0)[-1], level=0)
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes
        for asset in latest_corr.columns:
            self.graph.add_node(asset)
        
        # Add edges based on correlation threshold
        for i, asset1 in enumerate(latest_corr.columns):
            for j, asset2 in enumerate(latest_corr.columns):
                if i < j:  # Avoid duplicate edges
                    corr = latest_corr.loc[asset1, asset2]
                    if abs(corr) > self.correlation_threshold:
                        self.graph.add_edge(asset1, asset2, weight=abs(corr))
        
        return self.graph
    
    def compute_graph_metrics(self):
        """Compute basic graph metrics"""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'avg_shortest_path': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else float('inf'),
            'num_connected_components': nx.number_connected_components(self.graph),
        }
        
        # Node-level metrics
        node_metrics = {}
        for node in self.graph.nodes():
            node_metrics[node] = {
                'degree': self.graph.degree(node),
                'betweenness_centrality': nx.betweenness_centrality(self.graph)[node],
                'clustering_coefficient': nx.clustering(self.graph, node),
                'eigenvector_centrality': nx.eigenvector_centrality_numpy(self.graph)[node] if nx.is_connected(self.graph) else 0.0
            }
        
        return metrics, node_metrics
    
    def save_graph(self, output_path=None):
        """Save graph to file"""
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), '../data_cache/correlation_graph.pkl')
        
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {output_path}")
    
    def load_graph(self, graph_path=None):
        """Load graph from file"""
        if graph_path is None:
            graph_path = os.path.join(os.path.dirname(__file__), '../data_cache/correlation_graph.pkl')
        
        import pickle
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded from {graph_path}")

if __name__ == "__main__":
    # Example usage
    builder = CorrelationGraphBuilder()
    graph = builder.build_graph()
    
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    metrics, node_metrics = builder.compute_graph_metrics()
    print(f"Graph metrics: {metrics}")
    
    # Save graph
    builder.save_graph()
    
    # Print top 5 assets by degree
    sorted_nodes = sorted(node_metrics.items(), key=lambda x: x[1]['degree'], reverse=True)
    print("\nTop 5 assets by degree:")
    for asset, metrics in sorted_nodes[:5]:
        print(f"{asset}: degree={metrics['degree']}, centrality={metrics['betweenness_centrality']:.3f}") 