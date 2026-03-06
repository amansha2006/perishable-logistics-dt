# =============================================================
# env/logistics_env.py
# Updated to accept any graph object (synthetic OR real-world OSM).
# All original simulation logic is preserved exactly.
# =============================================================

import random

import numpy as np

from env.agents import VehicleAgent, WarehouseAgent


class LogisticsEnvironment:

    def __init__(
        self,
        max_steps=100,
        num_nodes=50,
        num_warehouses=5,
        num_vehicles=10,
        decay_rate=0.03,
        seed=42,
        dock_capacity=3,
        service_time=4,
        graph=None,          # Pass any graph here — synthetic or OSM
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.max_steps = max_steps
        self.num_vehicles = num_vehicles
        self.decay_rate = decay_rate
        self.dock_capacity = dock_capacity
        self.service_time = service_time

        # Use injected graph or build synthetic default
        if graph is not None:
            self.graph = graph
        else:
            from env.graph import UrbanGraph
            self.graph = UrbanGraph(
                num_nodes=num_nodes,
                num_warehouses=num_warehouses,
                seed=seed,
            )

        self._init_agents()
        self.time = 0

    # ------------------------------------------------------------------

    def _init_agents(self):
        self.warehouses = {}
        for node, data in self.graph.graph.nodes(data=True):
            if data["type"] == "warehouse":
                self.warehouses[node] = WarehouseAgent(
                    node_id=node,
                    dock_capacity=self.dock_capacity,
                    service_time=self.service_time,
                )

        self.delivery_nodes = [
            n for n, data in self.graph.graph.nodes(data=True)
            if data["type"] == "delivery"
        ]

        self.vehicles = []
        warehouse_nodes = list(self.warehouses.keys())
        for i in range(self.num_vehicles):
            start = random.choice(warehouse_nodes)
            target = random.choice(self.delivery_nodes)
            v = VehicleAgent(vehicle_id=i, start_node=start, decay_rate=self.decay_rate)
            v.target_node = target
            self.vehicles.append(v)

    # ------------------------------------------------------------------

    def reset(self):
        self.time = 0
        warehouse_nodes = list(self.warehouses.keys())
        for vehicle in self.vehicles:
            start = random.choice(warehouse_nodes)
            target = random.choice(self.delivery_nodes)
            vehicle.reset(start, target)
        for warehouse in self.warehouses.values():
            warehouse.reset()
        return self._get_observation()

    # ------------------------------------------------------------------

    def step(self, actions):
        rewards = np.zeros(self.num_vehicles)
        violations = 0
        edge_usage = {}
        deliveries = 0
        freshness_sum = 0.0

        for warehouse in self.warehouses.values():
            warehouse.update()

        self.graph.update_congestion(edge_usage)

        for vehicle in self.vehicles:
            vid = vehicle.vehicle_id
            current = vehicle.current_node
            next_node = actions.get(vid, current)

            if next_node != current:
                edge = (current, next_node)
                edge_usage[edge] = edge_usage.get(edge, 0) + 1

            neighbors = self.graph.get_neighbors(current)
            if next_node not in neighbors:
                next_node = current

            old_dist = self.graph.get_distance(current, vehicle.target_node)
            new_dist = self.graph.get_distance(next_node, vehicle.target_node)

            vehicle.move_to(next_node)

            if vehicle.waiting:
                vehicle.update_freshness(delta_t=2.0)
            else:
                vehicle.update_freshness(delta_t=1.0)

            rewards[vid] += (old_dist - new_dist)

            if next_node != current:
                travel_time = self.graph.get_travel_time(current, next_node)
                rewards[vid] -= 0.01 * travel_time

            if next_node in self.warehouses:
                success = self.warehouses[next_node].allocate_dock(vid)
                if not success:
                    vehicle.set_waiting(True)
                    rewards[vid] -= 5.0
                    violations += 1
                else:
                    vehicle.set_waiting(False)

            rewards[vid] -= 0.01

            if vehicle.freshness < 0.15:
                rewards[vid] -= 5.0

            if vehicle.current_node == vehicle.target_node:
                rewards[vid] += 20 * vehicle.freshness
                deliveries += 1
                freshness_sum += vehicle.freshness
                new_start = random.choice(list(self.warehouses.keys()))
                new_target = random.choice(self.delivery_nodes)
                vehicle.reset(new_start, new_target)

        self.time += 1
        done = self.time >= self.max_steps

        info = {
            "violations": violations,
            "deliveries": deliveries,
            "avg_freshness": freshness_sum / deliveries if deliveries > 0 else 0.0,
        }
        return self._get_observation(), rewards, done, info

    # ------------------------------------------------------------------

    def _get_observation(self):
        N = self.graph.num_nodes
        vehicle_states = np.array(
            [[v.current_node / N, v.freshness, v.target_node / N] for v in self.vehicles],
            dtype=np.float32,
        )
        warehouse_states = np.array(
            [[w.current_occupancy / w.dock_capacity] for w in self.warehouses.values()],
            dtype=np.float32,
        )
        return {"vehicles": vehicle_states, "warehouses": warehouse_states}
