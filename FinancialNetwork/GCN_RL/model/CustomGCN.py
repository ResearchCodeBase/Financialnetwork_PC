import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, **kwargs):
        super(CustomGCNConv, self).__init__(in_channels, out_channels, improved, cached, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        # Check if edge weights are provided, otherwise use equal weights
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)

        return super().forward(x, edge_index, edge_weight)
