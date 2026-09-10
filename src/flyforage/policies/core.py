from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flyforage.connectome.graph import ConnectomeGraph


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    observation_dim: int = 6
    action_dim: int = 4
    propagation_steps: int = 4
    input_neurons: int = 64
    readout_neurons: int = 64


class BioPolicy(nn.Module):
    """Sparse recurrent policy constrained by a measured connectome topology."""

    def __init__(self, graph: ConnectomeGraph, config: PolicyConfig | None = None) -> None:
        super().__init__()
        if graph.num_nodes == 0 or graph.num_edges == 0:
            raise ValueError("connectome graph cannot be empty")

        self.graph = graph
        self.config = config or PolicyConfig()
        sources, targets, weights = graph.indexed_edges()
        self.register_buffer("source_index", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("target_index", torch.tensor(targets, dtype=torch.long))

        base = torch.log1p(torch.tensor(weights, dtype=torch.float32))
        base = base / base.mean().clamp_min(1e-6)
        self.register_buffer("base_weight", base)

        input_count = min(self.config.input_neurons, graph.num_nodes)
        readout_count = min(self.config.readout_neurons, graph.num_nodes)
        self.register_buffer("input_index", torch.arange(input_count, dtype=torch.long))
        self.register_buffer(
            "readout_index",
            torch.arange(graph.num_nodes - readout_count, graph.num_nodes, dtype=torch.long),
        )

        self.sensory_encoder = nn.Linear(self.config.observation_dim, input_count)
        self.edge_gain = nn.Parameter(torch.ones(graph.num_edges))
        self.node_bias = nn.Parameter(torch.zeros(graph.num_nodes))
        self.leak_logit = nn.Parameter(torch.tensor(0.0))
        self.actor = nn.Linear(readout_count, self.config.action_dim)
        self.critic = nn.Linear(readout_count, 1)

    def _sparse_matrix(self) -> torch.Tensor:
        values = self.base_weight * self.edge_gain
        indices = torch.stack([self.target_index, self.source_index], dim=0)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.graph.num_nodes, self.graph.num_nodes),
            device=values.device,
        ).coalesce()

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = observation.shape[0]
        state = observation.new_zeros((batch, self.graph.num_nodes))
        sensory = torch.tanh(self.sensory_encoder(observation))
        state[:, self.input_index] = sensory

        adjacency = self._sparse_matrix()
        leak = torch.sigmoid(self.leak_logit)
        for _ in range(self.config.propagation_steps):
            propagated = torch.sparse.mm(adjacency, state.transpose(0, 1)).transpose(0, 1)
            state = torch.tanh(leak * state + propagated + self.node_bias)

        readout = state[:, self.readout_index]
        return self.actor(readout), self.critic(readout).squeeze(-1)


class MLPPolicy(nn.Module):
    def __init__(self, config: PolicyConfig | None = None, hidden_dim: int = 128) -> None:
        super().__init__()
        config = config or PolicyConfig()
        self.backbone = nn.Sequential(
            nn.Linear(config.observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, config.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observation)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class GRUPolicy(nn.Module):
    def __init__(self, config: PolicyConfig | None = None, hidden_dim: int = 128) -> None:
        super().__init__()
        config = config or PolicyConfig()
        self.gru = nn.GRU(config.observation_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, config.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = observation.unsqueeze(1)
        output, _ = self.gru(sequence)
        hidden = output[:, -1]
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
