"""
dataset_builder.py
Builds and manages the behavioral dataset CSV.
Also generates synthetic training data for cold-start ML training.
"""

import csv
import os
import random
import numpy as np
from datetime import datetime
from feature_engine import FeatureEngine

DATASET_PATH = "attention_dataset.csv"
FEATURE_NAMES = FeatureEngine.FEATURE_NAMES


def get_csv_header():
    return ["timestamp"] + FEATURE_NAMES + ["label"]


def append_sample(features: dict, label: str, path: str = DATASET_PATH):
    """Append a single labeled feature row to the dataset CSV."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_header())
        if not file_exists:
            writer.writeheader()
        row = {"timestamp": datetime.now().isoformat()}
        row.update(features)
        row["label"] = label
        writer.writerow(row)


def load_dataset(path: str = DATASET_PATH):
    """Load dataset CSV. Returns (X, y) as lists."""
    if not os.path.isfile(path):
        return [], []
    X, y = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[fn]) for fn in FEATURE_NAMES]
                X.append(features)
                y.append(row["label"])
            except (KeyError, ValueError):
                continue
    return X, y


def generate_synthetic_dataset(n_samples: int = 600, path: str = DATASET_PATH,
                                overwrite: bool = False):
    """
    Generate a realistic synthetic dataset for bootstrapping ML models.
    Simulates High / Medium / Low attention behavioral patterns.
    """
    if os.path.isfile(path) and not overwrite:
        existing_X, _ = load_dataset(path)
        if len(existing_X) >= 100:
            print(f"[DatasetBuilder] Dataset already has {len(existing_X)} rows. Skipping generation.")
            return

    random.seed(42)
    np.random.seed(42)

    rows = []
    labels_dist = {"High": n_samples // 3, "Medium": n_samples // 3,
                   "Low": n_samples - 2 * (n_samples // 3)}

    def clip(val, lo, hi):
        return max(lo, min(hi, val))

    for label, count in labels_dist.items():
        for _ in range(count):
            if label == "High":
                typing_speed    = clip(np.random.normal(4.5, 1.2), 1.0, 12.0)
                pause_time      = clip(np.random.normal(0.8, 0.4), 0.1, 5.0)
                mouse_speed     = clip(np.random.normal(250, 80), 50, 600)
                click_rate      = clip(np.random.normal(0.4, 0.15), 0.05, 2.0)
                idle_time       = clip(np.random.normal(1.5, 1.0), 0.0, 8.0)
                app_switch_freq = clip(np.random.normal(0.5, 0.3), 0.0, 2.0)
                error_rate      = clip(np.random.normal(0.05, 0.03), 0.0, 0.3)
                cursor_velocity = clip(np.random.normal(300, 100), 50, 800)
                movement_smooth = clip(np.random.normal(0.85, 0.08), 0.5, 1.0)

            elif label == "Medium":
                typing_speed    = clip(np.random.normal(2.5, 1.0), 0.5, 8.0)
                pause_time      = clip(np.random.normal(2.5, 1.0), 0.5, 15.0)
                mouse_speed     = clip(np.random.normal(150, 70), 20, 400)
                click_rate      = clip(np.random.normal(0.2, 0.12), 0.0, 1.0)
                idle_time       = clip(np.random.normal(5.0, 2.5), 0.5, 20.0)
                app_switch_freq = clip(np.random.normal(1.5, 0.7), 0.0, 5.0)
                error_rate      = clip(np.random.normal(0.12, 0.06), 0.0, 0.5)
                cursor_velocity = clip(np.random.normal(160, 80), 20, 500)
                movement_smooth = clip(np.random.normal(0.65, 0.12), 0.3, 0.9)

            else:  # Low
                typing_speed    = clip(np.random.normal(0.5, 0.4), 0.0, 3.0)
                pause_time      = clip(np.random.normal(8.0, 3.0), 2.0, 30.0)
                mouse_speed     = clip(np.random.normal(40, 30), 0.0, 200)
                click_rate      = clip(np.random.normal(0.05, 0.04), 0.0, 0.3)
                idle_time       = clip(np.random.normal(20.0, 8.0), 5.0, 120.0)
                app_switch_freq = clip(np.random.normal(3.0, 1.2), 0.5, 10.0)
                error_rate      = clip(np.random.normal(0.25, 0.10), 0.05, 0.8)
                cursor_velocity = clip(np.random.normal(40, 30), 0.0, 200)
                movement_smooth = clip(np.random.normal(0.35, 0.12), 0.1, 0.7)

            row = {
                "timestamp": datetime.now().isoformat(),
                "typing_speed": round(typing_speed, 4),
                "pause_time": round(pause_time, 4),
                "mouse_speed": round(mouse_speed, 4),
                "click_rate": round(click_rate, 4),
                "idle_time": round(idle_time, 4),
                "app_switch_freq": round(app_switch_freq, 4),
                "error_rate": round(error_rate, 4),
                "cursor_velocity": round(cursor_velocity, 4),
                "movement_smoothness": round(movement_smooth, 4),
                "label": label,
            }
            rows.append(row)

    random.shuffle(rows)

    mode = "w" if overwrite or not os.path.isfile(path) else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_header())
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    print(f"[DatasetBuilder] Generated {len(rows)} synthetic samples → {path}")


def get_dataset_stats(path: str = DATASET_PATH) -> dict:
    """Return basic stats about the dataset."""
    X, y = load_dataset(path)
    from collections import Counter
    counts = Counter(y)
    return {
        "total": len(y),
        "High": counts.get("High", 0),
        "Medium": counts.get("Medium", 0),
        "Low": counts.get("Low", 0),
    }


if __name__ == "__main__":
    generate_synthetic_dataset(600, overwrite=True)
    print(get_dataset_stats())
