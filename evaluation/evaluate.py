# =============================================================
# evaluation/evaluate.py  — UPDATED: Fix 1 (4th policy), Fix 2 (100 nodes), Fix 3 (t-tests)
# =============================================================

import csv
import os
import time

import numpy as np
from scipy import stats

from env.logistics_env import LogisticsEnvironment
from evaluation.metrics import MetricsLogger
from evaluation.policies import (greedy_policy, nearest_dock_policy,
                                  random_policy, constraint_aware_greedy_policy)

SEEDS = [42, 100, 999, 7, 21, 77, 123, 256, 512, 2024]

POLICY_MAP = {
    "random":                   random_policy,
    "greedy":                   greedy_policy,
    "nearest_dock":             nearest_dock_policy,
    "constraint_aware_greedy":  constraint_aware_greedy_policy,
}


def run_seed(seed, policy_fn, tag, episodes=100, max_steps=100,
             num_nodes=100, num_warehouses=5, num_vehicles=10,
             decay_rate=0.03, dock_capacity=3, graph=None,
             save_dir="results/logs"):

    env = LogisticsEnvironment(
        max_steps=max_steps, num_nodes=num_nodes,
        num_warehouses=num_warehouses, num_vehicles=num_vehicles,
        decay_rate=decay_rate, dock_capacity=dock_capacity,
        seed=seed, graph=graph,
    )
    logger = MetricsLogger(save_dir=save_dir)

    for ep in range(episodes):
        env.reset()
        total_r = deliveries = violations = 0
        fsum = fcnt = 0
        for _ in range(max_steps):
            _, rewards, done, info = env.step(policy_fn(env))
            total_r    += rewards.sum()
            deliveries += info["deliveries"]
            violations += info["violations"]
            if info["deliveries"] > 0:
                fsum += info["avg_freshness"] * info["deliveries"]
                fcnt += info["deliveries"]
            if done:
                break
        logger.log_episode(ep, total_r, deliveries, violations,
                           fsum / fcnt if fcnt > 0 else 0.0)

    logger.save_csv(f"{tag}_seed{seed}.csv")
    return logger.compute_summary()


def run_policy(policy_name, seeds=SEEDS, label=None, **kwargs):
    fn  = POLICY_MAP[policy_name]
    tag = label or policy_name
    print(f"\n  {tag.upper()}  ({len(seeds)} seeds)")
    summaries = []
    # Collect per-seed raw reward lists for t-tests
    per_seed_rewards = []

    for seed in seeds:
        print(f"    seed {seed:>4}...", end="", flush=True)
        s = run_seed(seed, fn, tag, **kwargs)
        summaries.append(s)
        per_seed_rewards.append(s["reward_mean"])
        print(f"  R={s['reward_mean']:>8.1f}  V={s['violations_mean']:.2f}  F={s['freshness_mean']:.4f}")

    rm  = np.mean([s["reward_mean"]     for s in summaries])
    rs  = np.std( [s["reward_mean"]     for s in summaries])
    dm  = np.mean([s["deliveries_mean"] for s in summaries])
    ds  = np.std( [s["deliveries_mean"] for s in summaries])
    vm  = np.mean([s["violations_mean"] for s in summaries])
    vs  = np.std( [s["violations_mean"] for s in summaries])
    fm  = np.mean([s["freshness_mean"]  for s in summaries])
    fs  = np.std( [s["freshness_mean"]  for s in summaries])

    print(f"\n  FINAL  R={rm:.2f}±{rs:.2f}  V={vm:.2f}±{vs:.2f}  F={fm:.4f}±{fs:.4f}")

    return dict(
        policy=tag,
        reward_mean=rm,      reward_std=rs,
        deliveries_mean=dm,  deliveries_std=ds,
        violations_mean=vm,  violations_std=vs,
        freshness_mean=fm,   freshness_std=fs,
        _per_seed_rewards=per_seed_rewards,   # for t-tests
    )


def run_ttests(rows):
    """Fix 3: t-tests between all policy pairs. Returns dict of results."""
    print("\n  ── Fix 3: Statistical t-tests (pairwise) ──")
    results = {}
    names = [r["policy"] for r in rows]
    seed_rewards = {r["policy"]: r["_per_seed_rewards"] for r in rows}

    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            n1, n2 = names[i], names[j]
            a, b = seed_rewards[n1], seed_rewards[n2]
            t_stat, p_val = stats.ttest_ind(a, b)
            key = f"{n1}_vs_{n2}"
            results[key] = {"t": round(t_stat, 2), "p": round(p_val, 4)}
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"    {n1} vs {n2}:  t={t_stat:.2f}  p={p_val:.4f}  {sig}")

    return results


def save_csv(rows, path):
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=clean[0].keys())
        w.writeheader()
        w.writerows(clean)
    print(f"  Saved: {path}")


def save_ttest_csv(ttest_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "t_stat", "p_value", "significant"])
        w.writeheader()
        for k, v in ttest_results.items():
            w.writerow({
                "comparison": k,
                "t_stat": v["t"],
                "p_value": v["p"],
                "significant": "yes" if v["p"] < 0.05 else "no"
            })
    print(f"  Saved: {path}")


def print_table(rows, title):
    print(f"\n  {title}")
    print(f"  {'Policy':<28} {'Reward':>16} {'Deliveries':>14} {'Violations':>14} {'Freshness':>14}")
    print("  " + "-" * 90)
    for r in rows:
        print(
            f"  {r['policy']:<28}"
            f" {r['reward_mean']:>8.1f}±{r['reward_std']:<6.1f}"
            f" {r['deliveries_mean']:>7.2f}±{r['deliveries_std']:<5.2f}"
            f" {r['violations_mean']:>7.2f}±{r['violations_std']:<5.2f}"
            f" {r['freshness_mean']:>7.4f}±{r['freshness_std']:<5.4f}"
        )
    if len(rows) > 1:
        br = rows[0]["reward_mean"]
        print("\n  Improvement over Random:")
        for r in rows[1:]:
            rw = (r["reward_mean"] - br) / abs(br) * 100 if br else 0
            print(f"    {r['policy']:<28}  reward {rw:+.1f}%")


# ── Experiment A: Synthetic (100 nodes — Fix 2) ───────────────
def experiment_synthetic():
    print("\n" + "=" * 65)
    print("  EXPERIMENT A — Synthetic (100 nodes) | 4 Policies | 10 Seeds")
    print("=" * 65)

    base = dict(
        episodes=100, max_steps=100, num_nodes=100,   # Fix 2: 100 nodes
        num_warehouses=5, num_vehicles=10, decay_rate=0.03,
        dock_capacity=3, save_dir="results/logs",
    )

    rows = [
        run_policy("random",                  **base),
        run_policy("greedy",                  **base),
        run_policy("nearest_dock",            **base),
        run_policy("constraint_aware_greedy", **base),   # Fix 1
    ]

    print_table(rows, "Synthetic Network — 4-Policy Comparison")

    # Fix 3: t-tests
    ttest_results = run_ttests(rows)

    save_csv(rows, "results/logs/experiment_A_synthetic.csv")
    save_ttest_csv(ttest_results, "results/logs/ttests_synthetic.csv")
    return rows, ttest_results


# ── Experiment B: Real-World OSM ──────────────────────────────
def experiment_osm():
    print("\n" + "=" * 65)
    print("  EXPERIMENT B — Real-World OSM | 4 Policies | 10 Seeds")
    print("=" * 65)

    try:
        from env.graph_osm import OSMUrbanGraph
        g = OSMUrbanGraph(num_warehouses=5, seed=42)
        g.print_summary()
    except ImportError:
        print("  [SKIP] Run: pip install osmnx")
        return None
    except Exception as e:
        print(f"  [SKIP] OSM graph failed: {e}")
        return None

    base = dict(
        episodes=100, max_steps=100, num_vehicles=10,
        decay_rate=0.03, dock_capacity=3,
        graph=g, save_dir="results/logs",
    )

    rows = [
        run_policy("random",                  label="random_osm",                  **base),
        run_policy("greedy",                  label="greedy_osm",                  **base),
        run_policy("nearest_dock",            label="nearest_dock_osm",            **base),
        run_policy("constraint_aware_greedy", label="constraint_aware_greedy_osm", **base),
    ]

    print_table(rows, "Real-World OSM — 4-Policy Comparison")
    ttest_results = run_ttests(rows)

    save_csv(rows, "results/logs/experiment_B_osm.csv")
    save_ttest_csv(ttest_results, "results/logs/ttests_osm.csv")
    return rows, ttest_results


if __name__ == "__main__":
    t0 = time.time()
    experiment_synthetic()
    experiment_osm()
    print(f"\n✓ evaluate.py complete — {time.time() - t0:.1f}s")
