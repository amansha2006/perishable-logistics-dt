# =============================================================
# env/graph_osm.py
# Real-World OpenStreetMap Graph — Chennai, India
# Extracts a ~500-node subgraph so all computations stay fast.
# Public API identical to UrbanGraph.
# Requirements: pip install osmnx
# =============================================================

import os
import pickle
import random

import networkx as nx
import numpy as np

CACHE_DIR    = "data/osm_cache"
TARGET_NODES = 500

FALLBACK_AREAS = [
    "Chennai, Tamil Nadu, India",
    "Coimbatore, Tamil Nadu, India",
    "Madurai, Tamil Nadu, India",
    "Bengaluru, Karnataka, India",
    "Hyderabad, Telangana, India",
]


class OSMUrbanGraph:
    def __init__(self, area="Chennai, Tamil Nadu, India",
                 num_warehouses=5, seed=42, use_cache=True, min_nodes=40):
        random.seed(seed)
        np.random.seed(seed)
        self.num_warehouses = num_warehouses
        self.seed           = seed
        self.time           = 0
        self.graph          = nx.DiGraph()
        self._area_used     = area
        os.makedirs(CACHE_DIR, exist_ok=True)

        raw = self._get_raw_graph(area, use_cache, min_nodes)
        sub = self._extract_subgraph(raw)
        self._build_digraph(sub)
        self._assign_warehouses()
        self._precompute()

        print(f"\n[OSMUrbanGraph] Ready  |  Area: {self._area_used}")
        print(f"  Nodes: {self.num_nodes}  Edges: {self.graph.number_of_edges()}")
        print(f"  Warehouses: {self.num_warehouses}  Delivery: {len(self.delivery_nodes)}")

    # ------------------------------------------------------------------
    def _get_raw_graph(self, requested_area, use_cache, min_nodes):
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("Run:  pip install osmnx")

        areas_to_try = [requested_area] + [a for a in FALLBACK_AREAS if a != requested_area]

        for area in areas_to_try:
            safe      = area.replace(",", "").replace(" ", "_")
            cache_p   = os.path.join(CACHE_DIR, f"{safe}_raw.pkl")

            if use_cache and os.path.exists(cache_p):
                print(f"[OSMUrbanGraph] Loading cache: {safe}")
                with open(cache_p, "rb") as f:
                    G = pickle.load(f)
                if G.number_of_nodes() >= min_nodes:
                    self._area_used = area
                    print(f"  Full graph: {G.number_of_nodes()} nodes")
                    return G
                continue

            print(f"[OSMUrbanGraph] Downloading: {area} ...")
            try:
                G = ox.graph_from_place(area, network_type="drive", retain_all=False)
                # Compatible with osmnx v1 and v2
                try:
                    G = ox.utils_graph.get_largest_component(G, strongly=True)
                except AttributeError:
                    if not nx.is_strongly_connected(G):
                        largest = max(nx.strongly_connected_components(G), key=len)
                        G = G.subgraph(largest).copy()

                print(f"  Downloaded: {G.number_of_nodes()} nodes")
                if use_cache:
                    with open(cache_p, "wb") as f:
                        pickle.dump(G, f)
                if G.number_of_nodes() >= min_nodes:
                    self._area_used = area
                    return G
            except Exception as e:
                print(f"  Failed: {e}  ->  trying next...")
                continue

        raise RuntimeError("[OSMUrbanGraph] Could not get a valid OSM graph.")

    # ------------------------------------------------------------------
    def _extract_subgraph(self, G_raw, target=TARGET_NODES):
        n_full = G_raw.number_of_nodes()
        if n_full <= target:
            return G_raw

        print(f"[OSMUrbanGraph] Extracting {target}-node subgraph from {n_full} nodes (BFS)...")
        # Start from highest-degree node (city centre proxy)
        start   = max(dict(G_raw.degree()).items(), key=lambda x: x[1])[0]
        visited = []
        queue   = [start]
        seen    = {start}

        while queue and len(visited) < target:
            node = queue.pop(0)
            visited.append(node)
            for nb in list(G_raw.successors(node)) + list(G_raw.predecessors(node)):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

        sub = G_raw.subgraph(visited).copy()
        if not nx.is_strongly_connected(sub):
            largest = max(nx.strongly_connected_components(sub), key=len)
            sub = sub.subgraph(largest).copy()

        print(f"  Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
        return sub

    # ------------------------------------------------------------------
    def _build_digraph(self, G_raw):
        mapping = {old: new for new, old in enumerate(G_raw.nodes())}
        G_re    = nx.relabel_nodes(G_raw, mapping)

        for n, d in G_re.nodes(data=True):
            self.graph.add_node(n, lat=d.get("y", 0.0), lon=d.get("x", 0.0), type="delivery")

        for u, v, d in G_re.edges(data=True):
            length    = d.get("length", 100.0)
            speed     = d.get("speed_kph", 30.0)
            if isinstance(speed, list):
                speed = float(speed[0])
            speed     = max(float(speed), 5.0)
            base_time = max((length / 1000.0) / speed * 3600.0, 1.0)
            self.graph.add_edge(u, v,
                                base_time=round(base_time, 3),
                                length_m=round(length, 1),
                                congestion=0.0)
        self.num_nodes = self.graph.number_of_nodes()

    # ------------------------------------------------------------------
    def _assign_warehouses(self):
        """Use out-degree centrality — O(E), always fast."""
        print(f"[OSMUrbanGraph] Assigning warehouses ({self.num_nodes} nodes)...")
        degree = dict(self.graph.out_degree())
        top_k  = set(sorted(degree, key=degree.get, reverse=True)[: self.num_warehouses])
        for node in self.graph.nodes:
            self.graph.nodes[node]["type"] = "warehouse" if node in top_k else "delivery"
        self.warehouse_nodes = sorted(top_k)
        self.delivery_nodes  = [n for n in self.graph.nodes if n not in top_k]

    # ------------------------------------------------------------------
    def _precompute(self):
        print(f"[OSMUrbanGraph] Precomputing shortest paths ({self.num_nodes} nodes)...")
        self.adjacency       = {n: list(self.graph.successors(n)) for n in self.graph.nodes}
        self.distance_matrix = dict(nx.all_pairs_shortest_path_length(self.graph))
        print(f"[OSMUrbanGraph] Precompute done.")

    # ------------------------------------------------------------------
    # Public API (identical to UrbanGraph)
    # ------------------------------------------------------------------
    def update_congestion(self, edge_usage):
        for u, v, data in self.graph.edges(data=True):
            data["congestion"] = min(0.1 * edge_usage.get((u, v), 0), 0.8)

    def get_travel_time(self, u, v):
        e = self.graph[u][v]
        return e["base_time"] * (1 + e["congestion"])

    def get_neighbors(self, node):
        return self.adjacency.get(node, [])

    def get_distance(self, u, v):
        return self.distance_matrix.get(u, {}).get(v, 9999)

    def print_summary(self):
        edges = self.graph.number_of_edges()
        tts   = [d["base_time"] for _, _, d in self.graph.edges(data=True)]
        lens  = [d["length_m"]  for _, _, d in self.graph.edges(data=True)]
        print("=" * 55)
        print(f"  OSM GRAPH  |  {self._area_used}")
        print(f"  Nodes: {self.num_nodes}  Edges: {edges}  Avg degree: {edges/self.num_nodes:.1f}")
        print(f"  Warehouses: {self.num_warehouses}  Delivery: {len(self.delivery_nodes)}")
        print(f"  Avg seg: {np.mean(lens):.0f}m  Avg travel: {np.mean(tts):.1f}s")
        print("=" * 55)
