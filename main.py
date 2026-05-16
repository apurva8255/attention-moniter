"""
main.py
Privacy-Preserving Human Attention Monitoring System
Using Behavioral Analytics

Entry point and central controller.
Orchestrates: ActivityCollector → FeatureEngine → Dataset → ML → Alert → UI
"""

import os
import sys
import threading
import time
import csv
from datetime import datetime

# Ensure project dir is on path
sys.path.insert(0, os.path.dirname(__file__))

from activity_collector import ActivityCollector
from feature_engine import FeatureEngine
from dataset_builder import (
    generate_synthetic_dataset,
    append_sample,
    load_dataset,
    get_dataset_stats,
)
from ml_model import MLModelTrainer
from attention_engine import AttentionEngine
from alert_system import AlertSystem


class AttentionController:
    """
    Central controller that wires all subsystems together.
    The UI talks to this controller; the controller runs the monitoring loop.
    """

    MONITOR_INTERVAL = 5  # seconds per monitoring window

    def __init__(self):
        self.collector = ActivityCollector()
        self.feature_engine = FeatureEngine()
        self.model_trainer = MLModelTrainer()
        self.attention_engine = AttentionEngine(self.model_trainer)
        self.alert_system = AlertSystem(threshold=40.0)

        self._ui = None
        self._monitor_thread = None
        self._running = False
        self._latest_result = None

        # Try loading an existing model
        self._try_load_model()

        # Generate synthetic dataset for cold-start training
        generate_synthetic_dataset(n_samples=600)

    # ── Setup ──────────────────────────────────────────────────────────────

    def set_ui(self, ui):
        """Attach the UI; set up alert callback."""
        self._ui = ui
        self.alert_system.set_alert_callback(self._alert_callback)

    def _try_load_model(self):
        loaded = self.model_trainer.load_model()
        if loaded:
            print("[Controller] Pre-trained model loaded.")

    # ── Monitoring Loop ────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self.collector.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[Controller] Monitoring started.")

    def stop(self):
        self._running = False
        self.collector.stop()
        print("[Controller] Monitoring stopped.")

    def _monitor_loop(self):
        """Main 5-second loop: capture → extract → predict → update UI → alert."""
        while self._running:
            loop_start = time.time()

            try:
                snapshot = self.collector.get_snapshot()
                result = self.attention_engine.process(snapshot)
                self._latest_result = result

                # Save to dataset
                append_sample(result["features"],
                              label=result["label"])

                # Update UI
                if self._ui:
                    self._ui.after(0, lambda r=result: self._ui.update_metrics(r))

                # Alert check
                self.alert_system.check(result["score"], result["label"])

            except Exception as e:
                print(f"[Controller] Loop error: {e}")

            # Sleep for remainder of interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.MONITOR_INTERVAL - elapsed)
            time.sleep(sleep_time)

    # ── Training ──────────────────────────────────────────────────────────

    def train_models(self, progress_cb=None, done_cb=None):
        """Load dataset and train all ML models. Runs in calling thread."""
        try:
            X, y = load_dataset()
            if len(X) < 20:
                print("[Controller] Not enough data; using synthetic dataset.")
                generate_synthetic_dataset(600, overwrite=True)
                X, y = load_dataset()

            def _callback(name, metrics):
                if progress_cb:
                    progress_cb(name, metrics)

            results = self.model_trainer.train(X, y, callback=_callback)

            if done_cb:
                done_cb(results)

        except Exception as e:
            print(f"[Controller] Training error: {e}")
            if done_cb:
                done_cb({})

    # ── Alert ──────────────────────────────────────────────────────────────

    def _alert_callback(self, score: float, label: str):
        if self._ui:
            self._ui.show_alert_popup(score, label)

    def set_threshold(self, value: float):
        self.alert_system.set_threshold(value)

    def set_cooldown(self, value: float):
        self.alert_system.set_cooldown(value)

    def get_alert_count(self) -> int:
        return self.alert_system.get_alert_count()

    # ── Report ─────────────────────────────────────────────────────────────

    def save_report(self) -> str:
        """Save session summary CSV and return file path."""
        summary = self.attention_engine.get_summary()
        history = self.attention_engine.get_history()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"session_report_{timestamp}.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Session Report", datetime.now().isoformat()])
            writer.writerow([])
            writer.writerow(["Summary"])
            for k, v in summary.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(["Timestamp", "Score", "Label"])
            for ts, score, label in history:
                dt = datetime.fromtimestamp(ts).isoformat()
                writer.writerow([dt, score, label])

        print(f"[Controller] Report saved → {path}")
        return path

    # ── Info ───────────────────────────────────────────────────────────────

    def get_dataset_stats(self) -> dict:
        return get_dataset_stats()

    def get_best_model_name(self) -> str:
        return self.model_trainer.best_model_name or ""


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Privacy-Preserving Human Attention Monitoring System")
    print("  Using Behavioral Analytics")
    print("=" * 60)

    # Import UI here to avoid circular imports
    from ui import AttentionMonitorUI

    controller = AttentionController()
    app = AttentionMonitorUI(controller)
    controller.set_ui(app)

    # Auto-train on startup in background if model not loaded
    if not controller.model_trainer.is_trained():
        print("[Main] Auto-training models on startup...")

        def auto_train():
            time.sleep(1.5)  # Let UI finish loading
            controller.train_models(
                done_cb=lambda r: app.after(
                    0, lambda: app._on_train_done(r)) if r else None
            )

        threading.Thread(target=auto_train, daemon=True).start()

    app.mainloop()


if __name__ == "__main__":
    main()
