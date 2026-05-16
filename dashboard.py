"""
dashboard.py
Generates matplotlib charts embedded in the CustomTkinter UI.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime


DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#0f3460"
HIGH_COLOR = "#00d4aa"
MED_COLOR = "#f5a623"
LOW_COLOR = "#e94560"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"


class AttentionDashboard:
    """Embeddable matplotlib dashboard panel for the CustomTkinter window."""

    def __init__(self, parent_frame, width: int = 900, height: int = 380):
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)
        self._setup_subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._times = []
        self._scores = []
        self._idle_times = []
        self._typing_speeds = []
        self._labels = []

    def _setup_subplots(self):
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            ax.spines[:].set_color(GRID_COLOR)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
        self.fig.tight_layout(pad=2.0)

    def update(self, timestamp: float, score: float, idle: float,
               typing_speed: float, label: str):
        """Add a new data point and redraw charts."""
        self._times.append(datetime.fromtimestamp(timestamp))
        self._scores.append(score)
        self._idle_times.append(idle)
        self._typing_speeds.append(typing_speed)
        self._labels.append(label)

        # Keep last 60 readings
        maxlen = 60
        if len(self._times) > maxlen:
            self._times = self._times[-maxlen:]
            self._scores = self._scores[-maxlen:]
            self._idle_times = self._idle_times[-maxlen:]
            self._typing_speeds = self._typing_speeds[-maxlen:]
            self._labels = self._labels[-maxlen:]

        self._draw()

    def _draw(self):
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.cla()
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

        times = self._times
        if not times:
            self.canvas.draw()
            return

        # ── Chart 1: Attention Score ───────────────────────────────────────
        colors = [HIGH_COLOR if l == "High" else
                  (MED_COLOR if l == "Medium" else LOW_COLOR)
                  for l in self._labels]

        self.ax1.plot(times, self._scores, color=HIGH_COLOR,
                      linewidth=1.5, alpha=0.8, zorder=2)
        self.ax1.scatter(times, self._scores, c=colors, s=18, zorder=3)
        self.ax1.fill_between(times, self._scores, alpha=0.12, color=HIGH_COLOR)
        self.ax1.axhline(y=40, color=LOW_COLOR, linewidth=1, linestyle="--",
                         alpha=0.7, label="Low threshold")
        self.ax1.axhline(y=70, color=HIGH_COLOR, linewidth=1, linestyle="--",
                         alpha=0.7, label="High threshold")
        self.ax1.set_ylabel("Score", color=TEXT_COLOR, fontsize=8)
        self.ax1.set_title("Attention Score Over Time", color=TEXT_COLOR,
                           fontsize=9, fontweight="bold")
        self.ax1.set_ylim(0, 105)
        self.ax1.yaxis.label.set_color(TEXT_COLOR)

        # ── Chart 2: Idle Time ─────────────────────────────────────────────
        self.ax2.bar(times, self._idle_times, color=MED_COLOR, alpha=0.7,
                     width=0.00015)
        self.ax2.set_ylabel("Idle (s)", color=TEXT_COLOR, fontsize=8)
        self.ax2.set_title("Idle Time Per Window", color=TEXT_COLOR,
                           fontsize=9, fontweight="bold")
        self.ax2.yaxis.label.set_color(TEXT_COLOR)

        # ── Chart 3: Typing Speed ──────────────────────────────────────────
        self.ax3.plot(times, self._typing_speeds, color="#7f5af0",
                      linewidth=1.5, alpha=0.9)
        self.ax3.fill_between(times, self._typing_speeds, alpha=0.15,
                              color="#7f5af0")
        self.ax3.set_ylabel("Keys/s", color=TEXT_COLOR, fontsize=8)
        self.ax3.set_title("Typing Speed Over Time", color=TEXT_COLOR,
                           fontsize=9, fontweight="bold")
        self.ax3.yaxis.label.set_color(TEXT_COLOR)

        # Format x-axis for all
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20,
                     ha="right", fontsize=6, color=TEXT_COLOR)

        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()


class ModelComparisonChart:
    """Bar chart for ML model comparison results."""

    def __init__(self, parent_frame, width: int = 860, height: int = 300):
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, results: list):
        """
        Args:
            results: list of dicts with keys: model, accuracy, precision, recall, f1_score
        """
        if not results:
            return

        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.4, axis="y")

        models = [r.get("model", "?") for r in results]
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        metric_colors = [HIGH_COLOR, "#7f5af0", MED_COLOR, LOW_COLOR]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

        x = np.arange(len(models))
        bar_w = 0.18

        for i, (metric, color, mlabel) in enumerate(
                zip(metrics, metric_colors, metric_labels)):
            vals = []
            for r in results:
                try:
                    vals.append(float(r.get(metric, 0)))
                except (ValueError, TypeError):
                    vals.append(0.0)
            bars = self.ax.bar(x + i * bar_w, vals, bar_w, label=mlabel,
                               color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    self.ax.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.01,
                                 f"{val:.2f}", ha="center", va="bottom",
                                 color=TEXT_COLOR, fontsize=6)

        self.ax.set_xticks(x + bar_w * 1.5)
        self.ax.set_xticklabels(models, color=TEXT_COLOR, fontsize=8)
        self.ax.set_ylim(0, 1.15)
        self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
        self.ax.set_title("ML Model Comparison", color=TEXT_COLOR,
                          fontsize=10, fontweight="bold")
        legend = self.ax.legend(fontsize=7, facecolor=ACCENT,
                                labelcolor=TEXT_COLOR)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
