"""
dashboard.py
Generates matplotlib charts embedded in the CustomTkinter UI.
- Light theme throughout
- Dashboard: 3 graphs switchable one at a time
- Model Comparison: Bar chart & Confusion Matrix switchable one at a time
"""

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


# ── Light Theme Palette ────────────────────────────────────────────────────
LIGHT_BG    = "#f0f4f8"
PANEL_BG    = "#ffffff"
CARD_BG     = "#e8edf5"
GRID_COLOR  = "#d0d7e3"
TEXT_COLOR  = "#1a1a2e"
DIM_COLOR   = "#5f6368"
BORDER      = "#c5ccd8"

HIGH_COLOR   = "#00897b"   # teal
MED_COLOR    = "#f57c00"   # orange
LOW_COLOR    = "#d32f2f"   # red
BLUE_COLOR   = "#1a73e8"
PURPLE_COLOR = "#6200ea"

NAV_ACTIVE   = "#1a73e8"
NAV_INACTIVE = "#e8edf5"
NAV_TXT_ON   = "#ffffff"
NAV_TXT_OFF  = "#1a1a2e"


def _style_ax_light(ax):
    """Apply light-theme styling to a matplotlib axes."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(True, color=GRID_COLOR, alpha=0.7, linewidth=0.6)


def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=20, ha="right", fontsize=6, color=TEXT_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard: Attention Score | Idle Time | Typing Speed  (one at a time)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionDashboard:
    """Real-time metrics dashboard — one graph visible at a time."""

    GRAPH_NAMES = [
        "📈 Attention Score",
        "💤 Idle Time",
        "⌨ Typing Speed",
    ]

    def __init__(self, parent_frame, width: int = 900, height: int = 380):
        self._current_graph = 0

        self._times         = []
        self._scores        = []
        self._idle_times    = []
        self._typing_speeds = []
        self._labels        = []

        self._width  = width
        self._height = height

        self._build_layout(parent_frame, width, height)

    def _build_layout(self, parent, width, height):
        # ── Nav buttons ───────────────────────────────────────────────────
        nav = ctk.CTkFrame(parent, fg_color=LIGHT_BG)
        nav.pack(fill="x", pady=(4, 0))

        self._tab_btns = []
        for i, name in enumerate(self.GRAPH_NAMES):
            active = (i == 0)
            btn = ctk.CTkButton(
                nav,
                text=name,
                width=165,
                height=30,
                corner_radius=8,
                fg_color=NAV_ACTIVE if active else NAV_INACTIVE,
                hover_color="#1558b0",
                text_color=NAV_TXT_ON if active else NAV_TXT_OFF,
                border_width=1,
                border_color=BORDER,
                font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda idx=i: self._switch_graph(idx),
            )
            btn.pack(side="left", padx=4, pady=4)
            self._tab_btns.append(btn)

        # ── Single matplotlib figure ──────────────────────────────────────
        fig_h = (height - 55) / 100
        fig_w = width / 100

        self.fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._draw_empty()

    def _switch_graph(self, idx: int):
        self._current_graph = idx
        for i, btn in enumerate(self._tab_btns):
            if i == idx:
                btn.configure(fg_color=NAV_ACTIVE, text_color=NAV_TXT_ON)
            else:
                btn.configure(fg_color=NAV_INACTIVE, text_color=NAV_TXT_OFF)
        self._draw()

    def update(self, timestamp: float, score: float, idle: float,
               typing_speed: float, label: str):
        self._times.append(datetime.fromtimestamp(timestamp))
        self._scores.append(score)
        self._idle_times.append(idle)
        self._typing_speeds.append(typing_speed)
        self._labels.append(label)

        maxlen = 60
        if len(self._times) > maxlen:
            self._times         = self._times[-maxlen:]
            self._scores        = self._scores[-maxlen:]
            self._idle_times    = self._idle_times[-maxlen:]
            self._typing_speeds = self._typing_speeds[-maxlen:]
            self._labels        = self._labels[-maxlen:]

        self._draw()

    def _draw_empty(self):
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.text(0.5, 0.5, "Waiting for data…",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     color=DIM_COLOR, fontsize=12)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

    def _draw(self):
        if not self._times:
            self._draw_empty()
            return

        times = self._times
        g = self._current_graph
        _style_ax_light(self.ax)

        if g == 0:
            # ── Attention Score ──────────────────────────────────────────
            colors = [
                HIGH_COLOR if l == "High" else (MED_COLOR if l == "Medium" else LOW_COLOR)
                for l in self._labels
            ]
            self.ax.plot(times, self._scores,
                         color=BLUE_COLOR, linewidth=1.8, alpha=0.85, zorder=2)
            self.ax.scatter(times, self._scores, c=colors, s=22, zorder=3, edgecolors="white", linewidth=0.5)
            self.ax.fill_between(times, self._scores, alpha=0.10, color=BLUE_COLOR)
            self.ax.axhline(y=40, color=LOW_COLOR, linewidth=1.2,
                            linestyle="--", alpha=0.8, label="Low threshold (40)")
            self.ax.axhline(y=70, color=HIGH_COLOR, linewidth=1.2,
                            linestyle="--", alpha=0.8, label="High threshold (70)")
            self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Attention Score Over Time",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.set_ylim(0, 105)
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            self.ax.legend(fontsize=7, facecolor=CARD_BG,
                           labelcolor=TEXT_COLOR, loc="upper right",
                           edgecolor=BORDER)
            _fmt_xaxis(self.ax)

        elif g == 1:
            # ── Idle Time ────────────────────────────────────────────────
            self.ax.bar(times, self._idle_times,
                        color=MED_COLOR, alpha=0.75, width=0.00015,
                        edgecolor="white", linewidth=0.4)
            self.ax.set_ylabel("Idle (s)", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Idle Time Per Window",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            _fmt_xaxis(self.ax)

        elif g == 2:
            # ── Typing Speed ─────────────────────────────────────────────
            self.ax.plot(times, self._typing_speeds,
                         color=PURPLE_COLOR, linewidth=1.8, alpha=0.9)
            self.ax.fill_between(times, self._typing_speeds,
                                 alpha=0.12, color=PURPLE_COLOR)
            self.ax.set_ylabel("Keys/s", color=TEXT_COLOR, fontsize=9)
            self.ax.set_title("Typing Speed Over Time",
                              color=TEXT_COLOR, fontsize=10, fontweight="bold")
            self.ax.yaxis.label.set_color(TEXT_COLOR)
            _fmt_xaxis(self.ax)

        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Model Comparison Dashboard: Bar Chart | Confusion Matrix  (one at a time)
# ══════════════════════════════════════════════════════════════════════════════

class ModelComparisonDashboard:
    """Switchable view: ML Bar Chart OR Confusion Matrix for each model."""

    GRAPH_NAMES = [
        "📊 Model Comparison",
        "🔴 Naive Bayes CM",
        "🟢 Random Forest CM",
        "🔵 SVM CM",
        "🟣 Logistic Regression CM",
    ]

    def __init__(self, parent_frame, width: int = 900, height: int = 340):
        self._current_graph = 0
        self._results       = []
        self._best_cm       = None
        self._best_name     = ""

        self._width  = width
        self._height = height

        self._build_layout(parent_frame, width, height)

    def _build_layout(self, parent, width, height):
        # ── Nav buttons ───────────────────────────────────────────────────
        nav = ctk.CTkFrame(parent, fg_color=LIGHT_BG)
        nav.pack(fill="x", pady=(4, 0), padx=10)

        self._tab_btns = []
        for i, name in enumerate(self.GRAPH_NAMES):
            active = (i == 0)
            btn = ctk.CTkButton(
                nav,
                text=name,
                width=155,
                height=30,
                corner_radius=8,
                fg_color=NAV_ACTIVE if active else NAV_INACTIVE,
                hover_color="#1558b0",
                text_color=NAV_TXT_ON if active else NAV_TXT_OFF,
                border_width=1,
                border_color=BORDER,
                font=ctk.CTkFont(size=10, weight="bold"),
                command=lambda idx=i: self._switch_graph(idx),
            )
            btn.pack(side="left", padx=4, pady=4)
            self._tab_btns.append(btn)

        # ── Single matplotlib figure ──────────────────────────────────────
        fig_h = (height - 55) / 100
        fig_w = width / 100

        self.fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self._draw_empty()

    def _switch_graph(self, idx: int):
        self._current_graph = idx
        for i, btn in enumerate(self._tab_btns):
            if i == idx:
                btn.configure(fg_color=NAV_ACTIVE, text_color=NAV_TXT_ON)
            else:
                btn.configure(fg_color=NAV_INACTIVE, text_color=NAV_TXT_OFF)
        self._draw()

    def draw(self, results: list, best_cm=None, best_model_name: str = ""):
        """Called after training completes."""
        self._results   = results or []
        self._best_cm   = best_cm
        self._best_name = best_model_name or ""
        self._draw()

    def _draw_empty(self):
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.text(0.5, 0.5, "Train ML models to see results here.",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     color=DIM_COLOR, fontsize=11)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

    def _draw(self):
        if not self._results:
            self._draw_empty()
            return

        g = self._current_graph

        if g == 0:
            self._draw_bar()
        else:
            model_names = ["Naive Bayes", "Random Forest", "SVM", "Logistic Regression"]
            selected_model = model_names[g - 1]
            self._draw_cm(selected_model)

    # ── Bar Chart ──────────────────────────────────────────────────────────

    def _draw_bar(self):
        _style_ax_light(self.ax)

        results = self._results
        models  = [r.get("model", "?") for r in results]
        metrics       = ["accuracy", "precision", "recall", "f1_score"]
        metric_colors = [HIGH_COLOR, BLUE_COLOR, MED_COLOR, LOW_COLOR]
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
                               label=mlabel, color=color, alpha=0.85,
                               edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        color=TEXT_COLOR, fontsize=6
                    )

        self.ax.set_xticks(x + bar_w * 1.5)
        self.ax.set_xticklabels(models, color=TEXT_COLOR, fontsize=9)
        self.ax.set_ylim(0, 1.15)
        self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=9)
        self.ax.set_title("ML Model Comparison",
                          color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.ax.legend(fontsize=8, facecolor=CARD_BG,
                       labelcolor=TEXT_COLOR, edgecolor=BORDER)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    # ── Confusion Matrix ───────────────────────────────────────────────────

    def _draw_cm(self, model_name: str):
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)

        cm = None
        for r in self._results:
            if r.get("model") == model_name:
                cm = r.get("confusion_matrix")
                break

        if not cm:
            self.ax.text(0.5, 0.5, f"No confusion matrix available for {model_name}.",
                         transform=self.ax.transAxes,
                         ha="center", va="center",
                         color=DIM_COLOR, fontsize=11)
            self.fig.tight_layout(pad=1.5)
            self.canvas.draw()
            return

        if isinstance(cm, str):
            import ast
            try:
                cm = ast.literal_eval(cm)
            except Exception:
                pass

        cm_np  = np.array(cm)
        labels = ["Low", "Medium", "High"]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)
        disp.plot(ax=self.ax, cmap="Blues", colorbar=False)

        # Make text readable on light background
        for text in disp.text_.ravel():
            text.set_color(TEXT_COLOR)
            text.set_fontsize(10)

        title_suffix = " (Best Model ⭐)" if model_name == self._best_name else ""
        self.ax.set_title(f"Confusion Matrix — {model_name}{title_suffix}",
                          color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        self.ax.xaxis.label.set_color(TEXT_COLOR)
        self.ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Legacy classes kept for backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class ModelComparisonChart:
    """Legacy — use ModelComparisonDashboard instead."""
    def __init__(self, parent_frame, width=860, height=300):
        self.fig = Figure(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, results):
        pass  # Handled by ModelComparisonDashboard


class ConfusionMatrixChart:
    """Legacy — use ModelComparisonDashboard instead."""
    def __init__(self, parent_frame, width=350, height=280):
        self.fig = Figure(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, cm, model_name):
        pass  # Handled by ModelComparisonDashboard