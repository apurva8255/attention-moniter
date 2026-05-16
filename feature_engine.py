"""
feature_engine.py
Converts raw activity snapshots into normalized ML feature vectors.
"""

import math


class FeatureEngine:
    """Transforms raw behavioral metrics into structured feature vectors."""

    FEATURE_NAMES = [
        "typing_speed",        # keys per second
        "pause_time",          # average pause between keys (seconds)
        "mouse_speed",         # average mouse pixels per second
        "click_rate",          # clicks per second
        "idle_time",           # seconds since last activity
        "app_switch_freq",     # switches per minute
        "error_rate",          # backspaces / total keystrokes
        "cursor_velocity",     # normalized mouse distance / time
        "movement_smoothness", # 0–1 (1 = very smooth)
    ]

    def extract(self, snapshot: dict) -> dict:
        """
        Extract and normalize features from a raw snapshot.

        Args:
            snapshot: dict from ActivityCollector.get_snapshot()

        Returns:
            dict of feature_name → float value
        """
        elapsed = max(snapshot.get("elapsed", 5.0), 0.001)
        keystrokes = snapshot.get("keystrokes", 0)
        backspaces = snapshot.get("backspaces", 0)
        clicks = snapshot.get("clicks", 0)
        mouse_distance = snapshot.get("mouse_distance", 0.0)
        avg_mouse_speed = snapshot.get("avg_mouse_speed", 0.0)
        idle_time = snapshot.get("idle_time", 0.0)
        app_switches = snapshot.get("app_switches", 0)
        avg_pause = snapshot.get("avg_pause", 0.0)
        smoothness = snapshot.get("smoothness", 1.0)

        typing_speed = keystrokes / elapsed
        pause_time = avg_pause if avg_pause > 0 else (elapsed if keystrokes == 0 else 0.0)
        mouse_speed = avg_mouse_speed
        click_rate = clicks / elapsed
        app_switch_freq = (app_switches / elapsed) * 60  # per minute
        error_rate = backspaces / max(keystrokes, 1)
        cursor_velocity = mouse_distance / elapsed

        features = {
            "typing_speed": round(typing_speed, 4),
            "pause_time": round(min(pause_time, 60.0), 4),
            "mouse_speed": round(mouse_speed, 4),
            "click_rate": round(click_rate, 4),
            "idle_time": round(min(idle_time, 300.0), 4),
            "app_switch_freq": round(app_switch_freq, 4),
            "error_rate": round(error_rate, 4),
            "cursor_velocity": round(cursor_velocity, 4),
            "movement_smoothness": round(smoothness, 4),
        }
        return features

    def to_vector(self, features: dict) -> list:
        """Convert feature dict to ordered list for ML input."""
        return [features.get(f, 0.0) for f in self.FEATURE_NAMES]

    def compute_attention_score(self, features: dict, label: str = None) -> float:
        """
        Heuristic attention score 0–100 based on features.
        Used as fallback when model not yet trained.
        """
        ts = min(features.get("typing_speed", 0), 10) / 10.0      # 0–1
        ms = min(features.get("mouse_speed", 0), 500) / 500.0      # 0–1
        idle = features.get("idle_time", 0)
        idle_factor = max(0.0, 1.0 - idle / 30.0)                  # penalize idle
        err = features.get("error_rate", 0)
        err_factor = max(0.0, 1.0 - err * 2)
        smooth = features.get("movement_smoothness", 1.0)
        cr = min(features.get("click_rate", 0), 2) / 2.0

        score = (ts * 0.25 + ms * 0.15 + idle_factor * 0.30 +
                 err_factor * 0.10 + smooth * 0.10 + cr * 0.10) * 100

        # Label override
        if label == "High":
            score = max(score, 65)
        elif label == "Low":
            score = min(score, 35)

        return round(min(max(score, 0.0), 100.0), 2)
