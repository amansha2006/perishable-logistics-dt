import os
import time

import numpy as np
import pandas as pd

from env.logistics_env import LogisticsEnvironment
from evaluation.policies import random_policy


def run_scalability(node_sizes=None, graph=None,
                    save_dir="results/logs", suffix=""):
    if node_sizes is None:
        node_sizes = [50, 100, 200]

    network = "OSM" if graph is not None else "Synthetic"
    rows = []
    print(f"\n[Scalability]  Network={network}")

    # For OSM graph, only one size is available (the graph is fixed)
    sizes_to_run = [None] if graph is not None else node_sizes

    for n in sizes_to_run:
        timings = []
        label = graph.num_nodes if graph is not None else n

        for run in range(3):
            seed = 42 + run

            if graph is None:
                env = LogisticsEnvironment(
                    max_steps=100, num_nodes=n,
                    num_warehouses=5, num_vehicles=10,
                    decay_rate=0.03, dock_capacity=3, seed=seed,
                )
            else:
                env = LogisticsEnvironment(
                    max_steps=100, num_vehicles=10,
                    decay_rate=0.03, dock_capacity=3,
                    seed=seed, graph=graph,
                )

            total_steps = 0
            t0 = time.time()
            for _ in range(20):
                env.reset()
                for _ in range(100):
                    env.step(random_policy(env))
                    total_steps += 1
            timings.append(total_steps / (time.time() - t0))

        mean_tput = np.mean(timings)
        std_tput  = np.std(timings)
        print(f"  |V|={label:<5}  {mean_tput:.0f} ± {std_tput:.0f} steps/sec")

        rows.append({
            "num_nodes":         label,
            "steps_per_sec":     round(mean_tput, 2),
            "steps_per_sec_std": round(std_tput, 2),
            "timing_runs":       3,
        })

    df   = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"scalability{suffix}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ------------------------------------------------------------------

if __name__ == "__main__":
    run_scalability()

    try:
        from env.graph_osm import OSMUrbanGraph
        g = OSMUrbanGraph(num_warehouses=5, seed=42)
        run_scalability(graph=g, suffix="_osm")
    except Exception as e:
        print(f"\n[OSM scalability skipped]: {e}")

    print("\n✓ scalability.py complete")
