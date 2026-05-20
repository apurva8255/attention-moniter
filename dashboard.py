"""
dashboard.py
Generates matplotlib charts embedded in the CustomTkinter UI.
Graphs are shown one at a time and can be switched via buttons.
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
from sklearn.metrics import ConfusionMatrixDisplay
import customtkinter as ctk


DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
HIGH_COLOR  = "#00d4aa"
MED_COLOR   = "#f5a623"
LOW_COLOR   = "#e94560"
TEXT_COLOR  = "#e0e0e0"
GRID_COLOR  = "#2a2a4a"
DIM_COLOR   = "#8888aa"


class AttentionDashboard:
    """Embeddable dashboard: one graph visible at a time, switchable via tabs."""

    GRAPH_NAMES = [
        "📈 Attention Score",
        "💤 Idle Time",
        "⌨ Typing Speed",
    ]

    def __init__(self, parent_frame, width: int = 900, height: int = 380):
        self._current_graph = 0  # index of visible graph

        self._times         = []
        self._scores        = []
        self._idle_times    = []
        self._typing_speeds = []
        self._labels        = []

        self._parent = parent_frame
        self._width  = width
        self._height = height

        self._build_layout(parent_frame, width, height)

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_layout(self, parent, width, height):
        """Build tab-switcher row + single-graph canvas."""

        # ── Top nav bar ───────────────────────────────────────────────────
        nav = ctk.CTkFrame(parent, fg_color=DARK_BG)
        nav.pack(fill="x", pady=(4, 0))

        self._tab_btns = []
        for i, name in enumerate(self.GRAPH_NAMES):
            btn = ctk.CTkButton(
                nav,
                text=name,
                width=160,
                height=30,
                corner_radius=8,
                fg_color=ACCENT if i != 0 else HIGH_COLOR,
                hover_color="#00b894",
                text_color=DARK_BG if i == 0 else TEXT_COLOR,
                font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda idx=i: self._switch_graph(idx),
            )
            btn.pack(side="left", padx=4, pady=4)
            self._tab_btns.append(btn)

        # ── Matplotlib figure (single subplot) ────────────────────────────
        fig_h = (height - 50) / 100  # leave room for nav bar
        fig_w = width / 100

        self.fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._draw_empty()

    # ── Graph switching ────────────────────────────────────────────────────

    def _switch_graph(self, idx: int):
        self._current_graph = idx
        # Update button styles
        for i, btn in enumerate(self._tab_btns):
            if i == idx:
                btn.configure(fg_color=HIGH_COLOR, text_color=DARK_BG)
            else:
                btn.configure(fg_color=ACCENT, text_color=TEXT_COLOR)
        self._draw()

    # ── Data ingestion ─────────────────────────────────────────────────────

    def update(self, timestamp: float, score: float, idle: float,
               typing_speed: float, label: str):
        """Add a new data point and redraw the active chart."""
        self._times.append(datetime.fromtimestamp(timestamp))
        self._scores.append(score)
        self._idle_times.append(idle)
        self._typing_speeds.append(typing_speed)
        self._labels.append(label)

        # Keep last 60 readings
        maxlen = 60
        if len(self._times) > maxlen:
            self._times         = self._times[-maxlen:]
            self._scores        = self._scores[-maxlen:]
            self._idle_times    = self._idle_times[-maxlen:]
            self._typing_speeds = self._typing_speeds[-maxlen:]
            self._labels        = self._labels[-maxlen:]

        self._draw()

    # ── Drawing ────────────────────────────────────────────────────────────

    def _draw_empty(self):
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.text(
            0.5, 0.5, "Waiting for data…",
            transform=self.ax.transAxes,
            ha="center", va="center",
            color=DIM_COLOR, fontsize=12
        )
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

    def _style_ax(self):
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

    def _fmt_xaxis(self):
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(
            self.ax.xaxis.get_majorticklabels(),
            rotation=20, ha="right", fontsize=6, color=TEXT_COLOR
        )

    def _draw(self):
        if not self._times:
            self._draw_empty()
            return

        times = self._times
        g = self._current_graph

        # ── Graph 0: Attention Score ───────────────────────────────────────
        if g == 0:
            self._style_ax()
            colors = [
                HIGH_COLOR if l == "High" else (MED_COLOR if l == "Medium" else LOW_COLOR)
                for l in self._labels
            ]
            self.ax.plot(times, self._scores,
                         color=HIGH_COLOR, linewidth=1.5, alpha=0.8, zorder=2)
            self.ax.scatter(times, self._scores, c=colors, s=18, zorder=3)
            self.ax.fill_between(times, self._scores, alpha=0.12, color=HIGH_COLOR)
            self.ax.axhline(y=40, color=LOW_COLOR, linewidth=1,
                            linestyle="--", alpha=0.7, label="Low threshold (40)")
            self.ax.axhline(y=70, color=HIGH_COLOR, linewidth=1,
                            linestyle="--", alpha=0.7, label="High threshold (70)")
            self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Attention Score Over Time",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.set_ylim(0, 105)
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            legend = self.ax.legend(
                fontsize=7, facecolor=ACCENT, labelcolor=TEXT_COLOR, loc="upper right"
            )
            self._fmt_xaxis()

        # ── Graph 1: Idle Time ─────────────────────────────────────────────
        elif g == 1:
            self._style_ax()
            self.ax.bar(times, self._idle_times,
                        color=MED_COLOR, alpha=0.7, width=0.00015)
            self.ax.set_ylabel("Idle (s)", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Idle Time Per Window",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            self._fmt_xaxis()

        # ── Graph 2: Typing Speed ──────────────────────────────────────────
        elif g == 2:
            self._style_ax()
            self.ax.plot(times, self._typing_speeds,
                         color="#7f5af0", linewidth=1.5, alpha=0.9)
            self.ax.fill_between(times, self._typing_speeds,
                                 alpha=0.15, color="#7f5af0")
            self.ax.set_ylabel("Keys/s", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Typing Speed Over Time",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            self._fmt_xaxis()

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
        if not results:
            return

        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.4, axis="y")

        models        = [r.get("model", "?") for r in results]
        metrics       = ["accuracy", "precision", "recall", "f1_score"]
        metric_colors = [HIGH_COLOR, "#7f5af0", MED_COLOR, LOW_COLOR]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

        x     = np.arange(len(models))
        bar_w = 0.18

        for i, (metric, color, mlabel) in enumerate(
                zip(metrics, metric_colors, metric_labels)):
            vals = []
            for r in results:
                try:
                    vals.append(float(r.get(metric, 0)))
                except (ValueError, TypeError):
                    vals.append(0.0)
            bars = self.ax.bar(x + i * bar_w, vals, bar_w,
                               label=mlabel, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        color=TEXT_COLOR, fontsize=6
                    )

        self.ax.set_xticks(x + bar_w * 1.5)
        self.ax.set_xticklabels(models, color=TEXT_COLOR, fontsize=8)
        self.ax.set_ylim(0, 1.15)
        self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
        self.ax.set_title("ML Model Comparison",
                          color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.ax.legend(fontsize=7, facecolor=ACCENT, labelcolor=TEXT_COLOR)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


class ConfusionMatrixChart:
    """Displays the confusion matrix for the best model."""

    def __init__(self, parent_frame, width: int = 350, height: int = 280):
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, cm, model_name: str):
        if not cm:
            return

        self.ax.cla()
        cm_np  = np.array(cm)
        labels = ["Low", "Medium", "High"]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)
        disp.plot(ax=self.ax, cmap="Blues", colorbar=False)

        for text in disp.text_.ravel():
            text.set_color(TEXT_COLOR)

        self.ax.set_title(f"Confusion Matrix — {model_name}",
                          color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_COLOR)
        self.ax.yaxis.label.set_color(TEXT_COLOR)

        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()