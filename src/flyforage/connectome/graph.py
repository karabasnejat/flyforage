from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Edge:
    source: int
    target: int
    weight: float


@dataclass(slots=True)
class ConnectomeGraph:
    neuron_ids: list[int]
    edges: list[Edge]
    index_by_neuron: dict[int, int]

    @classmethod
    def from_csv(cls, path: str | Path) -> "ConnectomeGraph":
        path = Path(path)
        edges: list[Edge] = []
        neurons: set[int] = set()

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "target", "weight"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"connectome CSV must contain {sorted(required)}")

            for row in reader:
                source = int(row["source"])
                target = int(row["target"])
                weight = float(row["weight"])
                if source == target or weight <= 0:
                    continue
                neurons.add(source)
                neurons.add(target)
                edges.append(Edge(source, target, weight))

        neuron_ids = sorted(neurons)
        return cls(neuron_ids, edges, {n: i for i, n in enumerate(neuron_ids)})

    @property
    def num_nodes(self) -> int:
        return len(self.neuron_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def indexed_edges(self) -> tuple[list[int], list[int], list[float]]:
        sources, targets, weights = [], [], []
        for edge in self.edges:
            sources.append(self.index_by_neuron[edge.source])
            targets.append(self.index_by_neuron[edge.target])
            weights.append(edge.weight)
        return sources, targets, weights
