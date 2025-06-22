import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
import pickle

class GraphSAGE(torch.nn.Module):
    """GraphSAGE model for node embeddings"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels if _ < num_layers - 1 else out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x

class GNNEmbeddingPipeline:
    """Pipeline for generating GNN embeddings from graph and features"""
    
    def __init__(self, graph_path=None, features_path=None):
        if graph_path is None:
            graph_path = os.path.join(os.path.dirname(__file__), '../data_cache/correlation_graph.pkl')
        if features_path is None:
            features_path = os.path.join(os.path.dirname(__file__), '../data_cache/graph_features_normalized.csv')
        
        # Load graph and features
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        self.features_df = pd.read_csv(features_path, index_col='asset')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embeddings = None
    
    def prepare_data(self):
        """Convert graph and features to PyTorch Geometric format"""
        # Create node mapping
        nodes = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Node features
        x = torch.tensor(self.features_df.loc[nodes].values, dtype=torch.float32)
        
        # Edge index
        edge_list = list(self.graph.edges())
        edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edge_list], dtype=torch.long).t()
        
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create PyTorch Geometric Data object
        self.data = Data(x=x, edge_index=edge_index)
        self.data = self.data.to(self.device)
        
        # Store node mapping
        self.node_to_idx = node_to_idx
        self.idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        return self.data
    
    def train_model(self, hidden_channels=128, out_channels=128, num_layers=2, epochs=100, lr=0.01):
        """Train GraphSAGE model with unsupervised loss"""
        # Prepare data
        data = self.prepare_data()
        
        # Initialize model
        self.model = GraphSAGE(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(data.x, data.edge_index)
            
            # Unsupervised loss: reconstruct adjacency matrix
            adj_pred = torch.sigmoid(torch.mm(embeddings, embeddings.t()))
            adj_true = torch.zeros_like(adj_pred)
            
            # Create adjacency matrix
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[:, i]
                adj_true[src, dst] = 1.0
                adj_true[dst, src] = 1.0  # Undirected graph
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy(adj_pred, adj_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        print(f"Training completed. Final loss: {loss.item():.4f}")
    
    def generate_embeddings(self):
        """Generate embeddings for all nodes"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.data.x, self.data.edge_index)
        
        # Convert to numpy and create DataFrame
        embeddings_np = embeddings.cpu().numpy()
        self.embeddings = pd.DataFrame(
            embeddings_np,
            index=[self.idx_to_node[i] for i in range(len(self.idx_to_node))],
            columns=[f'embedding_{i}' for i in range(embeddings_np.shape[1])]
        )
        
        return self.embeddings
    
    def save_embeddings(self, output_path=None):
        """Save embeddings to CSV"""
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), '../data_cache/gnn_embeddings.csv')
        
        if self.embeddings is not None:
            self.embeddings.to_csv(output_path)
            print(f"Embeddings saved to {output_path}")
        else:
            raise ValueError("Embeddings not generated yet. Call generate_embeddings() first.")
    
    def save_model(self, model_path=None):
        """Save trained model"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../data_cache/gnn_model.pth')
        
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            raise ValueError("Model not trained yet. Call train_model() first.")

if __name__ == "__main__":
    # Example usage
    pipeline = GNNEmbeddingPipeline()
    
    print("Training GraphSAGE model...")
    pipeline.train_model(hidden_channels=128, out_channels=128, num_layers=2, epochs=100)
    
    print("Generating embeddings...")
    embeddings = pipeline.generate_embeddings()
    
    print(f"Generated {embeddings.shape[1]}-dimensional embeddings for {embeddings.shape[0]} nodes")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Save embeddings and model
    pipeline.save_embeddings()
    pipeline.save_model()
    
    # Show some embedding statistics
    print(f"\nEmbedding statistics:")
    print(embeddings.describe())
    
    # Show top 5 assets by first embedding dimension
    print(f"\nTop 5 assets by first embedding dimension:")
    print(embeddings.iloc[:, 0].nlargest(5)) 