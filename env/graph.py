import networkx as nx
import numpy as np
import random


class UrbanGraph:
    def __init__(self, num_nodes=50, num_warehouses=5, edge_prob=0.2, seed=42):

        random.seed(seed)
        np.random.seed(seed)

        self.num_nodes = num_nodes
        self.num_warehouses = num_warehouses
        self.edge_prob = edge_prob

        self.graph = nx.DiGraph()
        self.time = 0

        self._create_nodes()
        self._create_edges()

        # Precompute adjacency list
        self.adjacency = {
            node: list(self.graph.successors(node))
            for node in self.graph.nodes
        }

        # Precompute shortest path matrix (NO runtime calls)
        self.distance_matrix = dict(
            nx.all_pairs_shortest_path_length(self.graph)
        )

    # -------------------------------------------------
    # Node Creation
    # -------------------------------------------------
    def _create_nodes(self):
        for i in range(self.num_nodes):
            node_type = "warehouse" if i < self.num_warehouses else "delivery"
            self.graph.add_node(i, type=node_type)

    # -------------------------------------------------
    # Edge Creation
    # -------------------------------------------------
    def _create_edges(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and random.random() < self.edge_prob:
                    self.graph.add_edge(
                        i,
                        j,
                        base_time=random.uniform(1.0, 5.0),
                        congestion=0.0
                    )

    # -------------------------------------------------
    # Congestion Update (Smooth)
    # -------------------------------------------------
    def update_congestion(self, edge_usage):

        for u, v, data in self.graph.edges(data=True):

            usage = edge_usage.get((u, v), 0)

            # Congestion proportional to usage
            data["congestion"] = min(0.1 * usage, 0.8)

    # -------------------------------------------------
    # Fast Travel Time
    # -------------------------------------------------
    def get_travel_time(self, u, v):
        edge = self.graph[u][v]
        return edge["base_time"] * (1 + edge["congestion"])

    # -------------------------------------------------
    # Fast Neighbor Access
    # -------------------------------------------------
    def get_neighbors(self, node):
        return self.adjacency[node]

    # -------------------------------------------------
    # Fast Distance Lookup
    # -------------------------------------------------
    def get_distance(self, u, v):
        return self.distance_matrix.get(u, {}).get(v, 100)
