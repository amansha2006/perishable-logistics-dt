import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

LOG_DIR  = "results/logs"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

COLORS = {
    "random":                   "#1f77b4",
    "greedy":                   "#d62728",
    "nearest_dock":             "#2ca02c",
    "constraint_aware_greedy":  "#ff7f0e",
}
LABELS = {
    "random":                   "Random",
    "greedy":                   "Greedy",
    "nearest_dock":             "Nearest-Dock",
    "constraint_aware_greedy":  "CA-Greedy (Ours)",
}
MARKERS = {
    "random": "o", "greedy": "s",
    "nearest_dock": "^", "constraint_aware_greedy": "D"
}
POLICY_ORDER = ["random", "greedy", "nearest_dock", "constraint_aware_greedy"]


def _load(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path)


def _save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def _get_color(policy_str):
    for k in COLORS:
        if policy_str.startswith(k[:8]):
            return COLORS[k]
    return "#888888"


def _get_label(policy_str):
    for k in LABELS:
        if policy_str.startswith(k[:8]):
            return LABELS[k]
    return policy_str


# ── Figure 1: 4-policy bar chart ─────────────────────────────
def fig_policy_comparison():
    df = _load(f"{LOG_DIR}/experiment_A_synthetic.csv")
    if df is None:
        return

    metrics = [
        ("reward_mean",     "reward_std",     "Cumulative Reward"),
        ("deliveries_mean", "deliveries_std", "Deliveries / Episode"),
        ("violations_mean", "violations_std", "Dock Violations / Episode"),
        ("freshness_mean",  "freshness_std",  "Freshness at Delivery"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(len(df))

    for ax, (m, s, title) in zip(axes, metrics):
        cols = [_get_color(p) for p in df["policy"]]
        bars = ax.bar(x, df[m], yerr=df[s], capsize=5, width=0.6,
                      color=cols, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_get_label(p) for p in df["policy"]],
                           fontsize=9, rotation=12)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("4-Policy Comparison — Synthetic Network (100 nodes, 10 seeds, mean ± std)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save("fig1_policy_comparison.png")


# ── Figures 2–4 / 7–9: sensitivity ───────────────────────────
def fig_sensitivity(param_name, x_label, suffix="", fig_num=2):
    path = f"{LOG_DIR}/sensitivity_{param_name}{suffix}.csv"
    df   = _load(path)
    if df is None:
        return

    policies = [p for p in POLICY_ORDER if p in df["policy"].values]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for pol in policies:
        sub = df[df["policy"] == pol]
        c, m, lbl = COLORS[pol], MARKERS[pol], LABELS[pol]
        axes[0].errorbar(sub[param_name], sub["violations_mean"],
                         yerr=sub["violations_std"], marker=m, color=c,
                         label=lbl, capsize=4, linewidth=1.8, markersize=6)
        axes[1].errorbar(sub[param_name], sub["reward_mean"],
                         yerr=sub["reward_std"], marker=m, color=c,
                         label=lbl, capsize=4, linewidth=1.8, markersize=6)
        axes[2].errorbar(sub[param_name], sub["freshness_mean"],
                         yerr=sub["freshness_std"], marker=m, color=c,
                         label=lbl, capsize=4, linewidth=1.8, markersize=6)

    net = "Real-World OSM" if suffix == "_osm" else "Synthetic"
    ylabels = ["Dock Violations (mean ± std)",
               "Cumulative Reward (mean ± std)",
               "Freshness at Delivery (mean ± std)"]

    for ax, ylabel in zip(axes, ylabels):
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sensitivity: {x_label}  [{net}]  (10 seeds, mean ± std)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    net_tag = "osm" if suffix == "_osm" else "syn"
    _save(f"fig{fig_num}_sensitivity_{param_name}_{net_tag}.png")


# ── Figure 5/10: scalability ─────────────────────────────────
def fig_scalability(suffix="", fig_num=5):
    path = f"{LOG_DIR}/scalability{suffix}.csv"
    df   = _load(path)
    if df is None:
        return

    net = "Real-World OSM" if suffix == "_osm" else "Synthetic"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df["num_nodes"], df["steps_per_sec"],
                yerr=df["steps_per_sec_std"],
                marker="o", color=COLORS["random"],
                capsize=5, linewidth=2, markersize=7)
    for _, row in df.iterrows():
        ax.annotate(f"{row['steps_per_sec']:.0f}",
                    (row["num_nodes"], row["steps_per_sec"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=9)
    ax.set_xlabel("Number of Nodes |V|")
    ax.set_ylabel("Simulation Throughput (steps/sec)")
    ax.set_title(f"Scalability Analysis  [{net}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    net_tag = "osm" if suffix == "_osm" else "syn"
    _save(f"fig{fig_num}_scalability_{net_tag}.png")


# ── Figure 6: OSM vs Synthetic ───────────────────────────────
def fig_osm_vs_synthetic():
    syn = _load(f"{LOG_DIR}/experiment_A_synthetic.csv")
    osm = _load(f"{LOG_DIR}/experiment_B_osm.csv")
    if syn is None or osm is None:
        print("  [skip] Need both results for comparison figure.")
        return

    base_pols = POLICY_ORDER
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x, w = np.arange(len(base_pols)), 0.35

    metrics = [
        ("reward_mean",     "reward_std",     "Cumulative Reward"),
        ("violations_mean", "violations_std", "Dock Violations / Episode"),
        ("freshness_mean",  "freshness_std",  "Freshness at Delivery"),
    ]

    for ax, (m, s, title) in zip(axes, metrics):
        for offset, df_, alpha in [(-w/2, syn, 0.6), (w/2, osm, 1.0)]:
            vals, errs, cols_ = [], [], []
            for pol in base_pols:
                row = df_[df_["policy"].str.contains(pol[:8])]
                vals.append(row[m].values[0] if not row.empty else 0)
                errs.append(row[s].values[0] if not row.empty else 0)
                cols_.append(COLORS[pol])
            label = "Synthetic" if alpha < 1 else "Real-World OSM"
            ax.bar(x + offset, vals, yerr=errs, width=w, capsize=5,
                   color=cols_, edgecolor="black", linewidth=0.7,
                   alpha=alpha, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in base_pols],
                           fontsize=8, rotation=12)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle("Synthetic vs Real-World OSM — 4-Policy Comparison  (10 seeds, mean ± std)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save("fig6_osm_vs_synthetic.png")


if __name__ == "__main__":
    print("Generating all figures...\n")
    fig_policy_comparison()
    fig_sensitivity("num_vehicles",  "Number of Vehicles K",  fig_num=2)
    fig_sensitivity("decay_rate",    "Decay Rate λ",           fig_num=3)
    fig_sensitivity("dock_capacity", "Dock Capacity Cw",       fig_num=4)
    fig_scalability(fig_num=5)
    fig_osm_vs_synthetic()
    fig_sensitivity("num_vehicles",  "Number of Vehicles K",  suffix="_osm", fig_num=7)
    fig_sensitivity("decay_rate",    "Decay Rate λ",           suffix="_osm", fig_num=8)
    fig_sensitivity("dock_capacity", "Dock Capacity Cw",       suffix="_osm", fig_num=9)
    fig_scalability(suffix="_osm", fig_num=10)
    print(f"\n✓ All figures saved to {PLOT_DIR}/")
