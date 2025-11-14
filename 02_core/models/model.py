import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

EDGE_EMB_DIM = 32

class EdgeEmbedding(nn.Module):
    """MLP-based encoder for raw edge features."""

    def __init__(self, in_dim, out_dim=EDGE_EMB_DIM):
        """
        Initialize the edge embedding module.

        Args:
            in_dim (int): Input edge feature dimension.
            out_dim (int): Output embedding dimension (default = EDGE_EMB_DIM).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, e):
        """
        Compute edge embeddings.

        Args:
            e (Tensor): Raw edge feature tensor of shape [E, in_dim].

        Returns:
            Tensor: Embedded edge features of shape [E, out_dim].
        """
        return self.mlp(e)


class GraphTransformer(nn.Module):
    """Transformer-based GNN with edge-aware attention."""

    def __init__(
        self,
        in_dim,
        edge_in_dim,
        hidden=64,
        heads=4,
        layers=3,
        dropout=0.3
    ):
        """
        Initialize the GraphTransformer model.

        Args:
            in_dim (int): Input node feature dimension.
            edge_in_dim (int): Input edge feature dimension.
            hidden (int): Hidden size per attention head.
            heads (int): Number of attention heads.
            layers (int): Number of TransformerConv layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.edge_emb = EdgeEmbedding(edge_in_dim, EDGE_EMB_DIM)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dim = in_dim
        for _ in range(layers):
            conv = TransformerConv(
                dim,
                hidden,
                heads=heads,
                dropout=dropout,
                edge_dim=EDGE_EMB_DIM
            )
            self.layers.append(conv)
            self.norms.append(LayerNorm(hidden * heads))
            dim = hidden * heads

        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GraphTransformer.

        Args:
            x (Tensor): Node feature matrix [N, in_dim].
            edge_index (Tensor): Graph edges [2, E].
            edge_attr (Tensor): Raw edge features [E, edge_in_dim].

        Returns:
            Tensor: Logits of shape [N, 2] for binary classification.
        """
        e = self.edge_emb(edge_attr)
        for conv, norm in zip(self.layers, self.norms):
            x = F.relu(norm(conv(x, edge_index, edge_attr=e)))
        return self.mlp(x)

