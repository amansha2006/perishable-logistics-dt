"""
Fast experiment runner — reuses graph across seeds, 20 episodes per seed.
"""
import sys, os, csv, time
sys.path.insert(0, '.')
import numpy as np
from scipy import stats
from env.graph import UrbanGraph
from env.logistics_env import LogisticsEnvironment
from evaluation.policies import (random_policy, greedy_policy,
    nearest_dock_policy, constraint_aware_greedy_policy)

SEEDS    = [42, 100, 999, 7, 21, 77, 123, 256, 512, 2024]
EPISODES = 20   # per seed (sufficient for statistical comparison)
STEPS    = 100

POLICIES = [
    ("random",                   random_policy),
    ("greedy",                   greedy_policy),
    ("nearest_dock",             nearest_dock_policy),
    ("constraint_aware_greedy",  constraint_aware_greedy_policy),
]

os.makedirs("results/logs",  exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# ── Build shared graph once per seed ──────────────────────────
def run_experiment(num_nodes=100, label="synthetic", graph_obj=None):
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}  |  nodes={num_nodes}  |  4 policies × {len(SEEDS)} seeds")
    print(f"{'='*60}")

    all_rows = []
    per_policy_rewards = {}   # for t-tests

    for pol_name, pol_fn in POLICIES:
        seed_rewards = []
        seed_results = []

        for seed in SEEDS:
            np.random.seed(seed)
            if graph_obj is None:
                env = LogisticsEnvironment(num_nodes=num_nodes,
                    num_warehouses=5, num_vehicles=10,
                    decay_rate=0.03, dock_capacity=3,
                    max_steps=STEPS, seed=seed)
            else:
                env = LogisticsEnvironment(num_vehicles=10,
                    decay_rate=0.03, dock_capacity=3,
                    max_steps=STEPS, seed=seed, graph=graph_obj)

            ep_rewards, ep_deliv, ep_viol, ep_fresh = [], [], [], []
            for ep in range(EPISODES):
                env.reset()
                tot_r = tot_d = tot_v = 0
                fsum = fcnt = 0
                for _ in range(STEPS):
                    _, rews, done, info = env.step(pol_fn(env))
                    tot_r += rews.sum()
                    tot_d += info["deliveries"]
                    tot_v += info["violations"]
                    if info["deliveries"] > 0:
                        fsum += info["avg_freshness"] * info["deliveries"]
                        fcnt += info["deliveries"]
                    if done: break
                ep_rewards.append(tot_r)
                ep_deliv.append(tot_d)
                ep_viol.append(tot_v)
                ep_fresh.append(fsum/fcnt if fcnt > 0 else 0.0)

            seed_rewards.append(np.mean(ep_rewards))
            seed_results.append({
                "r": np.mean(ep_rewards), "d": np.mean(ep_deliv),
                "v": np.mean(ep_viol),    "f": np.mean(ep_fresh),
            })

        rm = np.mean([s["r"] for s in seed_results])
        rs = np.std( [s["r"] for s in seed_results])
        dm = np.mean([s["d"] for s in seed_results])
        ds = np.std( [s["d"] for s in seed_results])
        vm = np.mean([s["v"] for s in seed_results])
        vs = np.std( [s["v"] for s in seed_results])
        fm = np.mean([s["f"] for s in seed_results])
        fs = np.std( [s["f"] for s in seed_results])

        print(f"  {pol_name:<28}  R={rm:>8.1f}±{rs:<6.1f}  V={vm:.2f}±{vs:.2f}  F={fm:.4f}±{fs:.4f}")
        all_rows.append(dict(policy=pol_name,
            reward_mean=round(rm,2), reward_std=round(rs,2),
            deliveries_mean=round(dm,2), deliveries_std=round(ds,2),
            violations_mean=round(vm,2), violations_std=round(vs,2),
            freshness_mean=round(fm,5), freshness_std=round(fs,5),
        ))
        per_policy_rewards[pol_name] = seed_rewards

    # Save CSV
    outfile = f"results/logs/experiment_{'B_osm' if label=='osm' else 'A_synthetic'}.csv"
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  Saved: {outfile}")

    # t-tests (Fix 3)
    print("\n  ── Fix 3: Pairwise t-tests ──")
    names = [r["policy"] for r in all_rows]
    ttest_rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            t_stat, p_val = stats.ttest_ind(
                per_policy_rewards[n1], per_policy_rewards[n2])
            sig = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns"))
            print(f"    {n1} vs {n2}:  t={t_stat:.2f}  p={p_val:.4f}  {sig}")
            ttest_rows.append(dict(
                comparison=f"{n1} vs {n2}",
                t_stat=round(t_stat,3), p_value=round(p_val,4),
                significant="yes" if p_val<0.05 else "no"
            ))

    ttfile = f"results/logs/ttests_{'osm' if label=='osm' else 'synthetic'}.csv"
    with open(ttfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ttest_rows[0].keys())
        w.writeheader(); w.writerows(ttest_rows)
    print(f"  Saved: {ttfile}")

    return all_rows, per_policy_rewards


# ── Sensitivity analysis ──────────────────────────────────────
def run_sensitivity_fast(param_name, param_values, num_nodes=100, suffix=""):
    print(f"\n[Sensitivity: {param_name}]  nodes={num_nodes}")
    rows = []
    for pol_name, pol_fn in POLICIES:
        for val in param_values:
            ep_rm, ep_vm, ep_fm, ep_dm = [], [], [], []
            for seed in SEEDS:
                cfg = dict(num_nodes=num_nodes, num_warehouses=5,
                    num_vehicles=10, decay_rate=0.03, dock_capacity=3,
                    max_steps=STEPS, seed=seed)
                cfg[param_name] = val
                env = LogisticsEnvironment(**cfg)
                seed_r, seed_v, seed_f, seed_d = [], [], [], []
                for ep in range(10):  # 10 eps per seed for sensitivity
                    env.reset()
                    tot_r=tot_d=tot_v=0; fsum=fcnt=0
                    for _ in range(STEPS):
                        _, rews, done, info = env.step(pol_fn(env))
                        tot_r+=rews.sum(); tot_d+=info["deliveries"]
                        tot_v+=info["violations"]
                        if info["deliveries"]>0:
                            fsum+=info["avg_freshness"]*info["deliveries"]; fcnt+=info["deliveries"]
                        if done: break
                    seed_r.append(tot_r); seed_d.append(tot_d)
                    seed_v.append(tot_v); seed_f.append(fsum/fcnt if fcnt>0 else 0.)
                ep_rm.append(np.mean(seed_r)); ep_vm.append(np.mean(seed_v))
                ep_fm.append(np.mean(seed_f)); ep_dm.append(np.mean(seed_d))
            rows.append({
                param_name: val, "policy": pol_name,
                "reward_mean": round(np.mean(ep_rm),2), "reward_std": round(np.std(ep_rm),2),
                "violations_mean": round(np.mean(ep_vm),2), "violations_std": round(np.std(ep_vm),2),
                "freshness_mean": round(np.mean(ep_fm),5), "freshness_std": round(np.std(ep_fm),5),
                "deliveries_mean": round(np.mean(ep_dm),2), "deliveries_std": round(np.std(ep_dm),2),
            })
    import pandas as pd
    df = pd.DataFrame(rows)
    path = f"results/logs/sensitivity_{param_name}{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ── Scalability ───────────────────────────────────────────────
def run_scalability_fast(node_sizes=[100, 150, 200], suffix=""):
    import time as tm
    rows = []
    print(f"\n[Scalability]  nodes={node_sizes}")
    for n in node_sizes:
        timings = []
        for run in range(3):
            env = LogisticsEnvironment(num_nodes=n, seed=42+run)
            total_steps = 0
            t0 = tm.time()
            for _ in range(5):
                env.reset()
                for _ in range(100):
                    env.step(random_policy(env)); total_steps+=1
            timings.append(total_steps/(tm.time()-t0))
        print(f"  |V|={n}  {np.mean(timings):.0f}±{np.std(timings):.0f} steps/sec")
        rows.append(dict(num_nodes=n,
            steps_per_sec=round(np.mean(timings),1),
            steps_per_sec_std=round(np.std(timings),1), timing_runs=3))
    import pandas as pd
    df = pd.DataFrame(rows)
    path = f"results/logs/scalability{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    # Synthetic — 4 policies, 100 nodes
    print("\n" + "█"*60)
    print("  STEP 1: Synthetic (100 nodes, 4 policies, 10 seeds)")
    print("█"*60)
    run_experiment(num_nodes=100, label="synthetic")

    # Sensitivity
    print("\n" + "█"*60)
    print("  STEP 2: Sensitivity Analysis")
    print("█"*60)
    for param, vals in [("num_vehicles",[5,10,20]),
                        ("decay_rate",[0.01,0.03,0.05]),
                        ("dock_capacity",[1,3,5])]:
        run_sensitivity_fast(param, vals, num_nodes=100)

    # Scalability (synthetic)
    print("\n" + "█"*60)
    print("  STEP 3: Scalability")
    print("█"*60)
    run_scalability_fast([100, 150, 200])

    print(f"\n{'█'*60}")
    print(f"  DONE — {(time.time()-t0)/60:.1f} minutes")
    print("█"*60)
