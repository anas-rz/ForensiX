"""The physdf graph prior: a spatio-temporal GNN trained on real media only.

Given the graph states of a clip with some nodes hidden, it predicts a Gaussian
(mean and variance) for every hidden feature from the visible nodes, along the
physical edges and over time. Trained by masked modelling on real clips, it
learns how real physical states evolve together; at test time the gap between
prediction and observation (the negative log-likelihood) is the physics
residual.

Layers: typed message passing over the fixed physical edges (one weight per edge
type, R-GCN style: Schlichtkrull et al., 2018), then a dilated depthwise
temporal convolution per node (ST-GCN style: Yan et al., 2018). Pure PyTorch.
"""

import torch
from torch import nn

from forensix.physics.graph.schema import EDGE_TYPES, EDGES, NODE_NAMES, feature_dims


def adjacency() -> torch.Tensor:
    """(R, N, N) row-normalized adjacency per edge type (undirected edges)."""
    index = {n: i for i, n in enumerate(NODE_NAMES)}
    a = torch.zeros(len(EDGE_TYPES), len(NODE_NAMES), len(NODE_NAMES))
    for u, v, r in EDGES:
        k = EDGE_TYPES.index(r)
        a[k, index[u], index[v]] = a[k, index[v], index[u]] = 1.0
    deg = a.sum(-1, keepdim=True)
    return a / deg.clamp(min=1.0)


class STLayer(nn.Module):
    def __init__(self, dim: int, n_relations: int, dilation: int, kernel: int = 5):
        super().__init__()
        self.rel = nn.Linear(dim, dim * n_relations, bias=False)
        self.self_loop = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.temporal = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=dim)
        self.point = nn.Linear(dim, dim)
        self.n_relations = n_relations

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, D); adj: (R, N, N)
        b, t, n, d = h.shape
        x = self.norm1(h)
        rel = self.rel(x).view(b, t, n, self.n_relations, d)
        msg = torch.einsum("rnm,btmrd->btnd", adj, rel)
        h = h + self.mlp(msg + self.self_loop(x))
        x = self.norm2(h).permute(0, 2, 3, 1).reshape(b * n, d, t)
        x = self.temporal(x).view(b, n, d, t).permute(0, 3, 1, 2)
        return h + self.point(torch.nn.functional.gelu(x))


class PhysicsGNN(nn.Module):
    def __init__(self, dim: int = 64, layers: int = 6, kernel: int = 5):
        super().__init__()
        dims = feature_dims()
        self.config = {"dim": dim, "layers": layers, "kernel": kernel}
        self.encoders = nn.ModuleDict({n: nn.Linear(dims[n], dim) for n in NODE_NAMES})
        self.decoders = nn.ModuleDict({n: nn.Linear(dim, 2 * dims[n]) for n in NODE_NAMES})
        self.node_embedding = nn.Parameter(torch.randn(len(NODE_NAMES), dim) * 0.02)
        self.hidden_token = nn.Parameter(torch.randn(len(NODE_NAMES), dim) * 0.02)
        self.layers = nn.ModuleList(STLayer(dim, len(EDGE_TYPES), 2 ** (i % 4), kernel) for i in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("adj", adjacency(), persistent=False)

    def forward(self, x: dict[str, torch.Tensor], visible: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``x[node]``: (B, T, d_node) standardized features; ``visible``: (B, T, N)
        bool, False where a node is hidden or invalid. Returns ``{node: (mean,
        log_var)}``, each (B, T, d_node)."""
        h = torch.stack([self.encoders[n](x[n]) for n in NODE_NAMES], dim=2)  # (B, T, N, D)
        vis = visible.unsqueeze(-1).to(h.dtype)
        h = vis * h + (1 - vis) * self.hidden_token + self.node_embedding
        for layer in self.layers:
            h = layer(h, self.adj)
        h = self.norm(h)
        out = {}
        for i, n in enumerate(NODE_NAMES):
            mean, log_var = self.decoders[n](h[:, :, i]).chunk(2, dim=-1)
            out[n] = (mean, log_var.clamp(-6.0, 6.0))
        return out


def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Per-element negative log-likelihood (constant dropped)."""
    return 0.5 * (log_var + (x - mean) ** 2 * torch.exp(-log_var))
