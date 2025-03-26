import networkx as nx
import numpy as np
import torch
from random import randint


def create_dataset():
    Gs = [] 
    y = []   

    for _ in range(50): 
        n = randint(10, 20)

        G_low = nx.fast_gnp_random_graph(n=n, p=0.1)
        Gs.append(G_low)
        y.append(0) 

        G_high = nx.fast_gnp_random_graph(n=n, p=0.4)
        Gs.append(G_high)
        y.append(1) 

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
