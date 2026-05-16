"""
attention_engine.py
Core pipeline: snapshot → features → ML prediction → attention score.
"""

import time
from feature_engine import FeatureEngine
from ml_model import MLModelTrainer

LABEL_SCORE_MAP = {
    "High": (75, 100),
    "Medium": (40, 74),
    "Low": (0, 39),
}

SCORE_WEIGHTS = {
    "High": 90.0,
    "Medium": 57.0,
    "Low": 18.0,
}


class AttentionEngine:
    """Maps behavioral features → attention label → score 0–100."""

    def __init__(self, model_trainer: MLModelTrainer):
        self.feature_engine = FeatureEngine()
        self.model = model_trainer
        self._history = []  # (timestamp, score, label)

    def process(self, snapshot: dict) -> dict:
        """
        Full pipeline for one snapshot window.

        Returns:
            {
                features: dict,
                label: str,
                score: float,
                confidence: float,
                timestamp: float
            }
        """
        features = self.feature_engine.extract(snapshot)
        vector = self.feature_engine.to_vector(features)

        if self.model.is_trained():
            label, confidence = self.model.predict(vector)
        else:
            # Heuristic fallback
            heuristic = self.feature_engine.compute_attention_score(features)
            if heuristic >= 65:
                label = "High"
            elif heuristic >= 35:
                label = "Medium"
            else:
                label = "Low"
            confidence = 0.5

        score = self._compute_score(label, confidence, features)

        result = {
            "features": features,
            "label": label,
            "score": score,
            "confidence": confidence,
            "timestamp": snapshot.get("timestamp", time.time()),
        }
        self._history.append((result["timestamp"], score, label))
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return result

    def _compute_score(self, label: str, confidence: float,
                       features: dict) -> float:
        """
        Blend ML label with heuristic features for a smooth 0–100 score.
        """
        base = SCORE_WEIGHTS.get(label, 57.0)

        # Fine-tune with features
        idle = features.get("idle_time", 0)
        idle_penalty = min(idle * 1.5, 25.0)

        ts = min(features.get("typing_speed", 0), 10)
        ts_bonus = ts * 1.5

        err = features.get("error_rate", 0)
        err_penalty = err * 10

        smooth = features.get("movement_smoothness", 0.5)
        smooth_bonus = smooth * 5

        raw = base + ts_bonus - idle_penalty - err_penalty + smooth_bonus

        # Confidence weighting
        heuristic = self.feature_engine.compute_attention_score(features)
        blended = confidence * raw + (1 - confidence) * heuristic

        return round(min(max(blended, 0.0), 100.0), 2)

    def get_history(self) -> list:
        """Return list of (timestamp, score, label) tuples."""
        return list(self._history)

    def get_summary(self) -> dict:
        """Compute session summary stats."""
        if not self._history:
            return {}

        scores = [s for _, s, _ in self._history]
        labels = [l for _, _, l in self._history]

        high_count = labels.count("High")
        med_count = labels.count("Medium")
        low_count = labels.count("Low")
        total = len(labels)

        return {
            "avg_score": round(sum(scores) / len(scores), 2),
            "max_score": round(max(scores), 2),
            "min_score": round(min(scores), 2),
            "high_pct": round(100 * high_count / max(total, 1), 1),
            "medium_pct": round(100 * med_count / max(total, 1), 1),
            "low_pct": round(100 * low_count / max(total, 1), 1),
            "total_readings": total,
            "focus_windows": high_count,
            "distracted_windows": low_count,
        }
