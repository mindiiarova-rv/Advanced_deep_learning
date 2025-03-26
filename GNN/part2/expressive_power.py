import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
cycle_graphs = [nx.cycle_graph(n) for n in range(10, 20)]
##################

############## Task 5
        
##################

adj_batch = list()
idx_batch = list()
y_batch = list()

for i in range(len(cycle_graphs)):
    adj = nx.adjacency_matrix(cycle_graphs[i])
    adj_matrix = adj + sp.eye(adj.shape[0])

    idx_batch.extend([i] * adj.shape[0])
    adj_batch.append(adj_matrix)

adj_batch = sp.block_diag(adj_batch)
adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)

num_nodes = adj_batch.shape[0]
features_batch = torch.ones((num_nodes, 1), dtype=torch.float32)
idx_batch = torch.tensor(idx_batch, dtype=torch.long)


##################


############## Task 8
        
##################
input_dim = 1
hidden_dim = 16
output_dim = 8
gnn = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr='mean', readout='mean', dropout=0.2)

print("GNN Representations with Mean Aggregation and Mean Readout:")
representation = gnn(features_batch, adj_batch, idx_batch)
print(f"Cycle Graph: {representation.detach().numpy()}")

gnn = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr='sum', readout='mean', dropout=0.2)

print("GNN Representations with Sum Aggregation and Mean Readout:")
representation = gnn(features_batch, adj_batch, idx_batch)
print(f"Cycle Graph: {representation.detach().numpy()}")

##################




############## Task 9
        
##################
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])  


G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]) 



##################




############## Task 10
        
##################
adj_batch = list()
idx_batch = list()
y_batch = list()

adj = nx.adjacency_matrix(G1)
adj_matrix = adj + sp.eye(adj.shape[0])

idx_batch.extend([0] * adj.shape[0])
adj_batch.append(adj_matrix)


adj = nx.adjacency_matrix(G2)
adj_matrix = adj + sp.eye(adj.shape[0])

idx_batch.extend([1] * adj.shape[0])
adj_batch.append(adj_matrix)


adj_batch = sp.block_diag(adj_batch)
adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)

num_nodes = adj_batch.shape[0]
features_batch = torch.ones((num_nodes, 1), dtype=torch.float32)

idx_batch = torch.tensor(idx_batch, dtype=torch.long)


##################


############## Task 11
        
##################
input_dim = 1
hidden_dim = 16
output_dim = 8
gnn = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr='sum', readout='sum', dropout=0.2)

print("GNN Representations with SUM Aggregation and SUM Readout:")
representation = gnn(features_batch, adj_batch, idx_batch)
print(f"Cycle Graph: {representation.detach().numpy()}")
##################
