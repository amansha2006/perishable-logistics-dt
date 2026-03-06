# evaluation/sensitivity.py — UPDATED: 4 policies, 100 nodes
import os
import numpy as np
import pandas as pd
from evaluation.evaluate import POLICY_MAP, SEEDS, run_seed


def run_sensitivity(param_name, param_values, seeds=SEEDS,
                    graph=None, save_dir="results/logs", suffix=""):
    rows = []
    network = "OSM" if graph is not None else "Synthetic"
    print(f"\n[Sensitivity: {param_name}]  Network={network}")

    for pol_name, pol_fn in POLICY_MAP.items():
        for val in param_values:
            print(f"  {pol_name:<28}  {param_name}={val:<6}", end="", flush=True)
            cfg = dict(
                episodes=50, max_steps=100, num_nodes=100,
                num_warehouses=5, num_vehicles=10,
                decay_rate=0.03, dock_capacity=3,
                graph=graph, save_dir=save_dir,
            )
            cfg[param_name] = val
            tag = f"{pol_name}_{param_name}{val}{suffix}"
            summaries = [run_seed(s, pol_fn, tag, **cfg) for s in seeds]

            rm = np.mean([s["reward_mean"]     for s in summaries])
            rs = np.std( [s["reward_mean"]     for s in summaries])
            vm = np.mean([s["violations_mean"] for s in summaries])
            vs = np.std( [s["violations_mean"] for s in summaries])
            fm = np.mean([s["freshness_mean"]  for s in summaries])
            fs = np.std( [s["freshness_mean"]  for s in summaries])
            dm = np.mean([s["deliveries_mean"] for s in summaries])
            ds = np.std( [s["deliveries_mean"] for s in summaries])

            print(f"  R={rm:.1f}±{rs:.1f}  V={vm:.2f}  F={fm:.4f}")

            rows.append({
                param_name:        val,
                "policy":          pol_name,
                "reward_mean":     round(rm, 4),
                "reward_std":      round(rs, 4),
                "deliveries_mean": round(dm, 4),
                "deliveries_std":  round(ds, 4),
                "violations_mean": round(vm, 4),
                "violations_std":  round(vs, 4),
                "freshness_mean":  round(fm, 6),
                "freshness_std":   round(fs, 6),
            })

    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, f"sensitivity_{param_name}{suffix}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


if __name__ == "__main__":
    PARAMS = [
        ("num_vehicles",  [5, 10, 20]),
        ("decay_rate",    [0.01, 0.03, 0.05]),
        ("dock_capacity", [1, 3, 5]),
    ]
    for param, vals in PARAMS:
        run_sensitivity(param, vals)
    print("\n✓ sensitivity.py complete")
