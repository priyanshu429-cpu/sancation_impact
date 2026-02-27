import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------------- Spatial GCN ----------------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        return h


# ---------------- Full Model ----------------
class SanctionImpactGNN(nn.Module):

    def __init__(self, in_dim, india_index=0):
        super().__init__()

        self.india_index = india_index
        self.gcn = GCNEncoder(in_dim)

        # temporal learning
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        # economic prediction heads
        self.heads = nn.ModuleDict({
            "gdp": nn.Linear(64, 1),
            "cpi": nn.Linear(64, 1),
            "fx": nn.Linear(64, 1),
            "trade": nn.Linear(64, 1),
            "fdi": nn.Linear(64, 1),
            "res": nn.Linear(64, 1),
            "score": nn.Linear(64, 1),
            "duration": nn.Linear(64, 1),
        })

    def forward(self, data_list):

        # collect India's embedding per year
        india_embeddings = []

        for data in data_list:
            h = self.gcn(data.x, data.edge_index, data.edge_weight)
            india_embeddings.append(h[self.india_index])

        # [time_steps, 64] → [1, time_steps, 64]
        seq = torch.stack(india_embeddings).unsqueeze(0)

        gru_out, _ = self.gru(seq)
        final_state = gru_out[:, -1, :]  # last year

        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = torch.sigmoid(head(final_state))

        return outputs