# agents.py

import numpy as np


# ====================================
# Vehicle Agent
# ====================================

class VehicleAgent:

    def __init__(self, vehicle_id, start_node, decay_rate=0.03):
        self.vehicle_id = vehicle_id
        self.decay_rate = decay_rate
        self.waiting = False
        self.current_node = start_node
        self.target_node = None
        self.freshness = 1.0

    def move_to(self, node):
        self.current_node = node

    def update_freshness(self, delta_t=1.0):
        self.freshness *= np.exp(-self.decay_rate * delta_t)
        self.freshness = max(self.freshness, 0.0)

    def reset(self, start_node, new_target):
        self.current_node = start_node
        self.target_node = new_target
        self.freshness = 1.0
    
    def set_waiting(self, status):
        self.waiting = status


# ====================================
# Warehouse Agent
# ====================================

class WarehouseAgent:

    def __init__(self, node_id, dock_capacity=3, service_time=4):

        self.node_id = node_id
        self.dock_capacity = dock_capacity
        self.service_time = service_time

        self.current_occupancy = 0
        self.service_timers = {}

    def allocate_dock(self, vehicle_id):

        if self.current_occupancy < self.dock_capacity:
            self.current_occupancy += 1
            self.service_timers[vehicle_id] = self.service_time
            return True

        return False

    def update(self):

        completed = []

        for vid in list(self.service_timers.keys()):
            self.service_timers[vid] -= 1

            if self.service_timers[vid] <= 0:
                completed.append(vid)
                del self.service_timers[vid]
                self.current_occupancy -= 1

        return completed

    def reset(self):
        self.current_occupancy = 0
        self.service_timers = {}