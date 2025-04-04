import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 2
    
        ##################
        adj = adj.to_dense()
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

        x = torch.mm(adj_hat, x_in)
        x = self.fc1(x)
        x = self.relu(x)

        x = torch.mm(adj_hat, x)
        x = self.fc2(x)
        x = self.relu(x)

        ##################
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), device=x_in.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        ##################

        return F.log_softmax(out, dim=1)
