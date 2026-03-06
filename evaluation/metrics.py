# evaluation/metrics.py

import os
import csv
import numpy as np


class MetricsLogger:

    def __init__(self, save_dir="results/logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_data = []

    def log_episode(self, episode, total_reward,
                    deliveries, violations, avg_freshness):

        self.episode_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "deliveries": deliveries,
            "violations": violations,
            "avg_freshness": avg_freshness
        })

    def save_csv(self, filename):

        file_path = os.path.join(self.save_dir, filename)

        keys = self.episode_data[0].keys()

        with open(file_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.episode_data)

        print(f"Saved results to {file_path}")

    def compute_summary(self):

        rewards = [e["total_reward"] for e in self.episode_data]
        deliveries = [e["deliveries"] for e in self.episode_data]
        violations = [e["violations"] for e in self.episode_data]
        freshness = [e["avg_freshness"] for e in self.episode_data]

        summary = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "deliveries_mean": np.mean(deliveries),
            "violations_mean": np.mean(violations),
            "freshness_mean": np.mean(freshness)
        }

        return summary