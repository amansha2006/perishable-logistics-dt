# =============================================================
# evaluation/policies.py
# FOUR baseline policies — Fix 1: added constraint_aware_greedy
# =============================================================

import numpy as np


def random_policy(env):
    actions = {}
    for v in env.vehicles:
        nbs = env.graph.get_neighbors(v.current_node)
        pool = nbs + [v.current_node]
        actions[v.vehicle_id] = int(np.random.choice(pool))
    return actions


def greedy_policy(env):
    actions = {}
    for v in env.vehicles:
        cur = v.current_node
        nbs = env.graph.get_neighbors(cur)
        if not nbs:
            actions[v.vehicle_id] = cur
            continue
        best = cur
        best_dist = env.graph.get_distance(cur, v.target_node)
        for nb in nbs:
            d = env.graph.get_distance(nb, v.target_node)
            if d < best_dist:
                best_dist = d
                best = nb
            elif d == best_dist and best != cur:
                if env.graph.get_travel_time(cur, nb) < env.graph.get_travel_time(cur, best):
                    best = nb
        actions[v.vehicle_id] = best
    return actions


def nearest_dock_policy(env):
    actions = {}
    for v in env.vehicles:
        cur = v.current_node
        nbs = env.graph.get_neighbors(cur)
        if not nbs:
            actions[v.vehicle_id] = cur
            continue
        if v.waiting:
            free_whs = [
                (wid, env.graph.get_distance(cur, wid))
                for wid, wa in env.warehouses.items()
                if wa.current_occupancy < wa.dock_capacity
            ]
            if free_whs:
                target_w = min(free_whs, key=lambda x: x[1])[0]
                best = cur
                best_dist = env.graph.get_distance(cur, target_w)
                for nb in nbs:
                    d = env.graph.get_distance(nb, target_w)
                    if d < best_dist:
                        best_dist = d
                        best = nb
                actions[v.vehicle_id] = best
                continue
        best = cur
        best_dist = env.graph.get_distance(cur, v.target_node)
        for nb in nbs:
            d = env.graph.get_distance(nb, v.target_node)
            if d < best_dist:
                best_dist = d
                best = nb
            elif d == best_dist and best != cur:
                if env.graph.get_travel_time(cur, nb) < env.graph.get_travel_time(cur, best):
                    best = nb
        actions[v.vehicle_id] = best
    return actions


def constraint_aware_greedy_policy(env):
    GAMMA     = 2.0
    THRESHOLD = 0.66

    dock_pressure = {
        wid: wa.current_occupancy / max(wa.dock_capacity, 1)
        for wid, wa in env.warehouses.items()
    }

    actions = {}
    for v in env.vehicles:
        cur = v.current_node
        nbs = env.graph.get_neighbors(cur)
        if not nbs:
            actions[v.vehicle_id] = cur
            continue

        best      = cur
        best_cost = float('inf')
        candidates = nbs + [cur]

        for nb in candidates:
            base_dist    = env.graph.get_distance(nb, v.target_node)
            dock_penalty = 0.0
            for wid, pressure in dock_pressure.items():
                if pressure > THRESHOLD:
                    dist_to_w = env.graph.get_distance(nb, wid)
                    if dist_to_w < 5:
                        dock_penalty += GAMMA * (pressure - THRESHOLD) * (5 - dist_to_w)
            cost = base_dist + dock_penalty
            if cost < best_cost:
                best_cost = cost
                best = nb
            elif cost == best_cost and nb != cur and best != cur:
                if env.graph.get_travel_time(cur, nb) < env.graph.get_travel_time(cur, best):
                    best = nb

        actions[v.vehicle_id] = best
    return actions
