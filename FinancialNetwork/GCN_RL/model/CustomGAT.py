import torch
from torch_geometric.nn import GATConv


class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super(CustomGATConv, self).__init__(in_channels, out_channels, heads, **kwargs)

    def forward(self, x, edge_index, size=None, return_attention_weights=None):
        # Assuming x[:, 0] indicates whether a node is a defaulting bank or not
        default_status = x[:, 0:1]

        # Perform the standard forward pass
        x = torch.cat([x[:, 1:], default_status], dim=1)
        x, attn = super().forward(x, edge_index, size, return_attention_weights)

        # Modify the attention weights based on default status
        # Here, you can implement your custom logic. For example:
        # Increase the attention weights for edges where the source is a defaulting bank
        defaulting_nodes = default_status[edge_index[0]] == 1
        attn[defaulting_nodes] *= 1.5  # Increase by 50%

        return x if return_attention_weights is None else (x, attn)
