# Perishable Urban Logistics — Digital Twin Simulation Framework

**Code repository for:**

> Sharma, A. & Jeya Mala, D. (2026). *A Graph-Structured Digital Twin-Inspired Simulation Framework for Constraint-Aware Perishable Urban Logistics: Real-World Validation and Four-Policy Benchmark Analysis with Pareto-Dominance Result.* Digital Engineering, Elsevier. *(under review)*

---

## Overview

This repository contains the complete simulation framework and experimental code for a graph-structured digital twin-inspired environment that jointly models:

- **Endogenous vehicle-density-dependent congestion** — the fleet itself generates the traffic it must navigate
- **Waiting-amplified freshness decay** — `F_k(t+Δt) = F_k(t) · exp(−λΔt(1 + β·w_k(t)))` — freshness decays faster when vehicles queue at saturated docks
- **Explicit dock capacity constraints** — enforced by construction within the simulation dynamics

Four routing policies are benchmarked across 10 random seeds on both a 100-node synthetic network and a 482-node real-world Chennai OpenStreetMap network.

---

## Key Results

| Policy | Reward | Violations | Freshness |
|---|---|---|---|
| Random | −971 ± 74 | 1.37 ± 0.72 | 0.422 |
| Greedy | 11,424 ± 162 | 3.03 ± 5.93 | 0.945 |
| Nearest-Dock | 11,381 ± 210 | 3.40 ± 6.21 | 0.947 |
| **CA-Greedy ★** | **11,470 ± 153** | **0.12 ± 0.20** | **0.947** |

★ **CA-Greedy Pareto-dominates all baselines**: statistically equivalent reward to Greedy (t=−0.62, p=0.54, ns) with **96% fewer dock violations** across 10 seeds.

---

## Repository Structure

```
perishable-logistics-dt/
│
├── env/                          # Core simulation environment
│   ├── logistics_env.py          # Main environment (step loop, rewards)
│   ├── agents.py                 # VehicleAgent and WarehouseAgent classes
│   ├── graph.py                  # Synthetic Erdős–Rényi road network
│   └── graph_osm.py              # Real-world OpenStreetMap network (Chennai)
│
├── evaluation/                   # Experiments and analysis
│   ├── policies.py               # Four routing policies (Random, Greedy, ND, CA-Greedy)
│   ├── evaluate.py               # Main experiment runner (synthetic + OSM)
│   ├── sensitivity.py            # Sensitivity analysis (fleet size, decay, dock cap)
│   ├── scalability.py            # Throughput vs. network size
│   ├── metrics.py                # Episode logging and CSV export
│   └── plot_results.py           # Figure generation (matplotlib)
│
├── results/
│   └── plots/                    # Pre-generated figures (PDF + PNG)
│
├── run_fast.py                   # Quick runner (20 episodes/seed, ~5 min)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/aman-sharma/perishable-logistics-dt.git
cd perishable-logistics-dt

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.9 or higher

**Note:** `osmnx` is required only for the real-world OSM experiment. The synthetic experiment runs without it.

---

## Quick Start

### Run the full benchmark (synthetic network, ~5 minutes)

```bash
python run_fast.py
```

This runs all 4 policies × 10 seeds × 20 episodes on the 100-node synthetic network and saves results to `results/logs/`.

### Run the complete experiment (synthetic + OSM, ~45 minutes)

```bash
python -m evaluation.evaluate
```

### Run sensitivity analysis

```bash
python -m evaluation.sensitivity
```

### Generate all figures

```bash
python -m evaluation.plot_results
```

---

## Environment Details

### Simulation Parameters (baseline)

| Parameter | Symbol | Value |
|---|---|---|
| Network nodes | \|V\| | 100 (synthetic) / 482 (OSM) |
| Fleet size | K | 10 |
| Dock capacity | C_w | 3 |
| Freshness decay rate | λ | 0.03 |
| Waiting amplification | β | 1.0 |
| Congestion sensitivity | α | 0.1 |
| Service time | T_w | 4 steps |
| Episode horizon | T | 100 steps |

### Reward Function

```
R_k(t) = +20·F_k(t)          [delivery reward]
        − 0.01·τ_ij(t)        [travel cost]
        − 5·1[violation]      [dock violation penalty]
        − 5·1[F_k < 0.15]     [spoilage penalty]
```

### Four Routing Policies

1. **Random** — uniformly random neighbour selection (lower-bound baseline)
2. **Greedy** — minimises graph distance to delivery target (constraint-oblivious upper bound)
3. **Nearest-Dock** — reactive: reroutes to nearest free dock only *after* a violation occurs
4. **CA-Greedy** — proactive: composite cost penalises routes toward near-saturated docks *before* violation

CA-Greedy cost function:
```
cost(nb) = d(nb, target) + γ · Σ_w max(0, n_w(t)/C_w − θ) · (H − d(nb, w))
```
where γ=2.0, θ=0.66, H=5.

---

## Reproducing Paper Results

All results in the paper use seeds: `[42, 100, 999, 7, 21, 77, 123, 256, 512, 2024]`

```python
from env.graph import UrbanGraph
from env.logistics_env import LogisticsEnvironment
from evaluation.policies import constraint_aware_greedy_policy

# Single run example
graph = UrbanGraph(num_nodes=100, num_warehouses=5, seed=42)
env = LogisticsEnvironment(
    max_steps=100, num_vehicles=10,
    decay_rate=0.03, dock_capacity=3,
    seed=42, graph=graph
)

env.reset()
for step in range(100):
    actions = constraint_aware_greedy_policy(env)
    obs, rewards, done, info = env.step(actions)
    if done:
        break

print(f"Violations: {info['violations']}, Freshness: {info['avg_freshness']:.4f}")
```

---

## Real-World OSM Network

The Chennai road network is downloaded automatically via OSMnx on first run and cached locally:

```python
from env.graph_osm import OSMUrbanGraph

graph = OSMUrbanGraph(
    area="Chennai, Tamil Nadu, India",
    num_warehouses=5,
    seed=42,
    use_cache=True      # cached after first download
)
graph.print_summary()
```

The 482-node subgraph is extracted by BFS from the highest out-degree node (primary urban junction proxy in T. Nagar).

---

## Statistical Testing

All pairwise t-tests between policies are in `results/logs/ttests_synthetic.csv` after running the full experiment. Key results:

| Comparison | t-stat | p-value | Significance |
|---|---|---|---|
| Random vs. Greedy | −208.4 | ≈0.000 | *** |
| Greedy vs. CA-Greedy | −0.62 | 0.544 | ns |
| Greedy vs. Nearest-Dock | +0.48 | 0.634 | ns |

---

## Citation

```bibtex
@article{sharma2026digital,
  author  = {Sharma, Aman and {Jeya Mala}, D.},
  title   = {A Graph-Structured Digital Twin-Inspired Simulation Framework
             for Constraint-Aware Perishable Urban Logistics:
             Real-World Validation and Four-Policy Benchmark Analysis
             with Pareto-Dominance Result},
  journal = {Digital Engineering},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

**Aman Sharma** — School of Computer Science and Engineering, Vellore Institute of Technology, Chennai  
**D. Jeya Mala** — jeyamala.d@vit.ac.in (Corresponding author)
