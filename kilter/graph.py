"""Grade predictor #2: a graph neural network over the holds of a climb.

Graph
  nodes  = holds. Each node knows its position (x, y), its role (start, hand,
           finish, foot) and which board position it is. The position ID goes
           through a learned embedding, so the model can learn on its own that
           a given hold is a bad crimp or a jug.
  edges  = pairs of holds within reach of each other (<= 56 board units,
           i.e. 7 holes), plus each hold's 2 nearest neighbours so nothing is
           isolated. Each edge carries the move: dx, dy and distance.

Model (a basic message-passing network, written in plain PyTorch)
  for each of L layers:
      message  m_ij = MLP([h_i, h_j, edge_ij])        "what hold j tells hold i"
      gather   a_i  = mean of m_ij over neighbours j
      update   h_i  = LayerNorm(h_i + MLP([h_i, a_i]))
  readout: mean and max over all holds, plus the wall angle -> MLP -> difficulty

All graphs in a batch are stacked into one big disconnected graph; `batch`
says which climb each node belongs to. This is what PyTorch Geometric does
internally, done by hand here so the only dependency is torch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import ROLES, Climb

REACH = 56.0     # board units; holes are 8 apart
MIN_NEIGHBOURS = 2
X_CENTER, Y_CENTER, SCALE = 72.0, 88.0, 80.0
MAX_ANGLE = 70.0


@dataclass
class ClimbGraph:
    placement: np.ndarray  # [n] index into the placement embedding
    role: np.ndarray       # [n] 0..3
    pos: np.ndarray        # [n, 2] normalised x, y
    edges: np.ndarray      # [2, e] source, target
    edge_attr: np.ndarray  # [e, 3] dx, dy, distance (normalised)
    angle: float           # normalised to 0..1
    y: float               # difficulty


def build_graph(c: Climb, placement_index: dict[int, int]) -> ClimbGraph:
    xy = np.array([[h.x, h.y] for h in c.holds], dtype=np.float32)
    n = len(xy)
    diff = xy[None, :, :] - xy[:, None, :]             # diff[i, j] = pos_j - pos_i
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    adj = dist <= REACH
    k = min(MIN_NEIGHBOURS, n - 1)                     # tiny hand-made climbs have < 3 holds
    nearest = np.argsort(dist, axis=1)[:, :k]
    adj[np.repeat(np.arange(n), k), nearest.ravel()] = True
    adj = adj | adj.T                                  # moves go both ways
    src, dst = np.nonzero(adj)
    d = diff[dst, src]                                 # vector from target to source
    edge_attr = np.stack([d[:, 0], d[:, 1], dist[dst, src]], axis=1) / SCALE
    return ClimbGraph(
        placement=np.array([placement_index[h.placement] for h in c.holds], dtype=np.int64),
        role=np.array([ROLES.index(h.role) for h in c.holds], dtype=np.int64),
        pos=(xy - [X_CENTER, Y_CENTER]) / SCALE,
        edges=np.stack([src, dst]).astype(np.int64),
        edge_attr=edge_attr.astype(np.float32),
        angle=c.angle / MAX_ANGLE,
        y=c.difficulty,
    )


def collate(graphs: list[ClimbGraph]) -> dict[str, torch.Tensor]:
    """Stack several climbs into one disconnected graph."""
    sizes = np.array([len(g.role) for g in graphs])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return {
        "placement": torch.from_numpy(np.concatenate([g.placement for g in graphs])),
        "role": torch.from_numpy(np.concatenate([g.role for g in graphs])),
        "pos": torch.from_numpy(np.concatenate([g.pos for g in graphs])).float(),
        "edges": torch.from_numpy(np.concatenate([g.edges + o for g, o in zip(graphs, offsets)], axis=1)),
        "edge_attr": torch.from_numpy(np.concatenate([g.edge_attr for g in graphs])),
        "batch": torch.from_numpy(np.repeat(np.arange(len(graphs)), sizes)),
        "angle": torch.tensor([g.angle for g in graphs], dtype=torch.float32),
        "y": torch.tensor([g.y for g in graphs], dtype=torch.float32),
    }


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MessagePassing(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 3):
        super().__init__()
        self.message = mlp(2 * dim + edge_dim, dim, dim)
        self.update = mlp(2 * dim, dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edges, edge_attr):
        src, dst = edges
        m = self.message(torch.cat([h[dst], h[src], edge_attr], dim=1))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(1)
        return self.norm(h + self.update(torch.cat([h, agg], dim=1)))


class ClimbGNN(nn.Module):
    def __init__(self, n_placements: int, dim: int = 96, layers: int = 3,
                 placement_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.placement_emb = nn.Embedding(n_placements, placement_dim)
        self.role_emb = nn.Embedding(len(ROLES), 8)
        self.inp = mlp(placement_dim + 8 + 2 + 1, dim, dim)    # + x, y + angle
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(MessagePassing(dim) for _ in range(layers))
        self.readout = mlp(2 * dim + 1, dim, 1)

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        batch, n_graphs = b["batch"], len(b["angle"])
        node_angle = b["angle"][batch].unsqueeze(1)
        h = self.inp(torch.cat([self.drop(self.placement_emb(b["placement"])),
                                self.role_emb(b["role"]), b["pos"], node_angle], dim=1))
        for layer in self.layers:
            h = layer(h, b["edges"], b["edge_attr"])
        counts = torch.zeros(n_graphs).index_add_(0, batch, torch.ones(len(batch)))
        mean = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h) / counts.unsqueeze(1)
        mx = torch.full((n_graphs, h.shape[1]), -1e9).scatter_reduce(
            0, batch.unsqueeze(1).expand_as(h), h, reduce="amax")
        return self.readout(torch.cat([mean, mx, b["angle"].unsqueeze(1)], dim=1)).squeeze(1)
