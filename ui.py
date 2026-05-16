"""
ui.py
Full dark-themed desktop UI built with CustomTkinter.
Displays real-time attention metrics, graphs, model comparison, and alerts.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
from datetime import datetime

# ── Theme ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DARK = "#0d0d1a"
BG_PANEL = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT_BLUE = "#0f3460"
ACCENT_CYAN = "#00d4aa"
ACCENT_ORANGE = "#f5a623"
ACCENT_RED = "#e94560"
ACCENT_PURPLE = "#7f5af0"
TEXT_PRIMARY = "#e0e0e0"
TEXT_DIM = "#8888aa"

LABEL_COLORS = {
    "High": ACCENT_CYAN,
    "Medium": ACCENT_ORANGE,
    "Low": ACCENT_RED,
}


class AttentionMonitorUI(ctk.CTk):
    """Main application window."""

    def __init__(self, engine_controller):
        super().__init__()
        self.controller = engine_controller
        self.title("🧠 Privacy-Preserving Human Attention Monitoring System")
        self.geometry("1280x860")
        self.minsize(1100, 720)
        self.configure(fg_color=BG_DARK)
        self._flash_active = False
        self._alert_popup = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, height=60, corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        title_lbl = ctk.CTkLabel(
            header,
            text="🧠 Privacy-Preserving Human Attention Monitoring System",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT_CYAN
        )
        title_lbl.pack(side="left", padx=20, pady=10)

        subtitle = ctk.CTkLabel(
            header,
            text="Behavioral Analytics · No Camera Required",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM
        )
        subtitle.pack(side="left", padx=5)

        # Control buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=15)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶ Start", width=90, height=32,
            fg_color=ACCENT_CYAN, hover_color="#00b894",
            text_color=BG_DARK, font=ctk.CTkFont(weight="bold"),
            command=self._on_start
        )
        self.start_btn.pack(side="left", padx=4)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="■ Stop", width=90, height=32,
            fg_color=ACCENT_RED, hover_color="#c0392b",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)

        self.train_btn = ctk.CTkButton(
            btn_frame, text="⚙ Train ML", width=100, height=32,
            fg_color=ACCENT_PURPLE, hover_color="#6c48d9",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_train
        )
        self.train_btn.pack(side="left", padx=4)

        self.report_btn = ctk.CTkButton(
            btn_frame, text="📊 Report", width=95, height=32,
            fg_color=ACCENT_BLUE, hover_color="#0d2c52",
            command=self._on_report
        )
        self.report_btn.pack(side="left", padx=4)

    def _build_main_content(self):
        self.main_pane = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.main_pane.pack(fill="both", expand=True, padx=10, pady=5)

        # Left column (metrics) — scrollable so all panels are always visible
        left_outer = ctk.CTkFrame(self.main_pane, fg_color=BG_DARK, width=300)
        left_outer.pack(side="left", fill="y", padx=(0, 5))
        left_outer.pack_propagate(False)

        left = ctk.CTkScrollableFrame(left_outer, fg_color=BG_DARK, width=285,
                                      scrollbar_button_color=ACCENT_BLUE,
                                      scrollbar_button_hover_color=ACCENT_CYAN)
        left.pack(fill="both", expand=True)

        self._build_score_card(left)
        self._build_metrics_panel(left)
        self._build_threshold_panel(left)
        self._build_dataset_panel(left)

        # Right column (tabs: dashboard + model comparison + logs)
        right = ctk.CTkFrame(self.main_pane, fg_color=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_tabs(right)

    def _build_score_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(card, text="ATTENTION SCORE",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(12, 0))

        self.score_label = ctk.CTkLabel(
            card, text="--",
            font=ctk.CTkFont(family="Segoe UI", size=56, weight="bold"),
            text_color=ACCENT_CYAN
        )
        self.score_label.pack(pady=4)

        self.status_label = ctk.CTkLabel(
            card, text="● IDLE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_DIM
        )
        self.status_label.pack(pady=(0, 8))

        # Progress bar for score
        self.score_bar = ctk.CTkProgressBar(card, width=220, height=12,
                                            progress_color=ACCENT_CYAN)
        self.score_bar.set(0)
        self.score_bar.pack(pady=(0, 12), padx=20)

    def _build_metrics_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(card, text="LIVE METRICS",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(10, 4))

        self._metric_labels = {}
        metrics = [
            ("⌨ Typing Speed", "typing_speed", "keys/s"),
            ("⏸ Pause Time", "pause_time", "s"),
            ("🖱 Mouse Speed", "mouse_speed", "px/s"),
            ("🖱 Click Rate", "click_rate", "/s"),
            ("💤 Idle Time", "idle_time", "s"),
            ("🔀 App Switches", "app_switch_freq", "/min"),
            ("❌ Error Rate", "error_rate", ""),
            ("📏 Cursor Vel.", "cursor_velocity", "px/s"),
            ("〰 Smoothness", "movement_smoothness", ""),
        ]

        for display, key, unit in metrics:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=display, font=ctk.CTkFont(size=11),
                         text_color=TEXT_DIM, width=130, anchor="w").pack(side="left")
            val_lbl = ctk.CTkLabel(row, text="--",
                                   font=ctk.CTkFont(size=11, weight="bold"),
                                   text_color=TEXT_PRIMARY)
            val_lbl.pack(side="right")
            self._metric_labels[key] = (val_lbl, unit)

        # Spacer
        ctk.CTkLabel(card, text="").pack(pady=3)

    def _build_threshold_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(card, text="ALERT SETTINGS",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(10, 2))

        # ── Score threshold ──────────────────────────────────────────
        ctk.CTkLabel(card, text="Score Threshold",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
                     anchor="w").pack(fill="x", padx=14, pady=(6, 0))

        self._threshold_var = tk.DoubleVar(value=40.0)
        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=(2, 0))

        self.threshold_slider = ctk.CTkSlider(
            row1, from_=0, to=100, variable=self._threshold_var,
            command=self._on_threshold_change,
            button_color=ACCENT_ORANGE, progress_color=ACCENT_ORANGE
        )
        self.threshold_slider.pack(side="left", fill="x", expand=True)

        self.threshold_val_lbl = ctk.CTkLabel(
            row1, text="40", width=38,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_ORANGE
        )
        self.threshold_val_lbl.pack(side="right")

        ctk.CTkLabel(card, text="Alert fires when score drops below this",
                     font=ctk.CTkFont(size=9), text_color=TEXT_DIM,
                     anchor="w").pack(fill="x", padx=14)

        # ── Cooldown timer ───────────────────────────────────────────
        ctk.CTkLabel(card, text="Cooldown Time",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
                     anchor="w").pack(fill="x", padx=14, pady=(10, 0))

        self._cooldown_var = tk.DoubleVar(value=30.0)
        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(2, 0))

        self.cooldown_slider = ctk.CTkSlider(
            row2, from_=5, to=300, variable=self._cooldown_var,
            command=self._on_cooldown_change,
            button_color=ACCENT_PURPLE, progress_color=ACCENT_PURPLE
        )
        self.cooldown_slider.pack(side="left", fill="x", expand=True)

        self.cooldown_val_lbl = ctk.CTkLabel(
            row2, text="30s", width=38,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_PURPLE
        )
        self.cooldown_val_lbl.pack(side="right")

        ctk.CTkLabel(card, text="Minimum gap between alerts (5s – 5min)",
                     font=ctk.CTkFont(size=9), text_color=TEXT_DIM,
                     anchor="w").pack(fill="x", padx=14)

        # ── Alert count ──────────────────────────────────────────────
        self.alert_count_lbl = ctk.CTkLabel(
            card, text="Alerts fired: 0",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM
        )
        self.alert_count_lbl.pack(pady=(8, 10))

    def _build_dataset_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(card, text="DATASET",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(10, 4))

        self.dataset_lbl = ctk.CTkLabel(card, text="Samples: 0",
                                        font=ctk.CTkFont(size=11),
                                        text_color=TEXT_PRIMARY)
        self.dataset_lbl.pack()

        self.model_lbl = ctk.CTkLabel(card, text="Model: Not trained",
                                      font=ctk.CTkFont(size=11),
                                      text_color=TEXT_DIM)
        self.model_lbl.pack(pady=(2, 10))

    def _build_tabs(self, parent):
        self.tabview = ctk.CTkTabview(parent, fg_color=BG_PANEL,
                                      segmented_button_fg_color=ACCENT_BLUE,
                                      segmented_button_selected_color=ACCENT_CYAN,
                                      segmented_button_selected_hover_color="#00b894",
                                      text_color=TEXT_PRIMARY)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_dashboard = self.tabview.add("📈 Dashboard")
        self.tab_models = self.tabview.add("🤖 Model Comparison")
        self.tab_logs = self.tabview.add("📋 Activity Log")

        self._build_dashboard_tab()
        self._build_models_tab()
        self._build_log_tab()

    def _build_dashboard_tab(self):
        from dashboard import AttentionDashboard
        self.dashboard = AttentionDashboard(self.tab_dashboard, width=870, height=460)

    def _build_models_tab(self):
        # Training progress
        prog_frame = ctk.CTkFrame(self.tab_models, fg_color=BG_CARD,
                                  corner_radius=8)
        prog_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.train_status_lbl = ctk.CTkLabel(
            prog_frame, text="Train ML models using the ⚙ Train ML button.",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.train_status_lbl.pack(side="left", padx=15, pady=10)

        self.train_progress = ctk.CTkProgressBar(prog_frame, width=200,
                                                  progress_color=ACCENT_PURPLE)
        self.train_progress.set(0)
        self.train_progress.pack(side="right", padx=15)

        # Chart area
        chart_frame = ctk.CTkFrame(self.tab_models, fg_color=BG_PANEL)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=5)

        comp_frame = ctk.CTkFrame(chart_frame, fg_color="transparent")
        comp_frame.pack(side="left", fill="both", expand=True, padx=5)

        cm_frame = ctk.CTkFrame(chart_frame, fg_color="transparent")
        cm_frame.pack(side="right", fill="both", expand=True, padx=5)

        from dashboard import ModelComparisonChart, ConfusionMatrixChart
        self.model_chart = ModelComparisonChart(comp_frame, width=450, height=280)
        self.cm_chart = ConfusionMatrixChart(cm_frame, width=350, height=280)

        # Results table
        table_frame = ctk.CTkFrame(self.tab_models, fg_color=BG_CARD,
                                   corner_radius=8)
        table_frame.pack(fill="x", padx=10, pady=(5, 10))

        cols = ("Model", "Accuracy", "Precision", "Recall", "F1")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview", background=BG_CARD,
                        fieldbackground=BG_CARD, foreground=TEXT_PRIMARY,
                        font=("Segoe UI", 10))
        style.configure("Dark.Treeview.Heading", background=ACCENT_BLUE,
                        foreground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT_CYAN)],
                  foreground=[("selected", BG_DARK)])

        self.model_table = ttk.Treeview(table_frame, columns=cols,
                                        show="headings", height=5,
                                        style="Dark.Treeview")
        for col in cols:
            self.model_table.heading(col, text=col)
            self.model_table.column(col, width=150, anchor="center")
        self.model_table.pack(fill="x", padx=10, pady=8)

    def _build_log_tab(self):
        self.log_text = ctk.CTkTextbox(
            self.tab_logs, fg_color=BG_CARD,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=ACCENT_BLUE, height=28,
                           corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_bar_lbl = ctk.CTkLabel(
            bar, text="Ready · Privacy Mode Active · No Camera Used",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM
        )
        self.status_bar_lbl.pack(side="left", padx=15)

        self.time_lbl = ctk.CTkLabel(bar, text="",
                                     font=ctk.CTkFont(size=10),
                                     text_color=TEXT_DIM)
        self.time_lbl.pack(side="right", padx=15)
        self._update_clock()

    def _update_clock(self):
        self.time_lbl.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    # ── Event Handlers ─────────────────────────────────────────────────────

    def _on_start(self):
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_bar_lbl.configure(text="● Monitoring active...")
        self.controller.start()

    def _on_stop(self):
        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.status_bar_lbl.configure(text="■ Monitoring stopped")
        self.controller.stop()

    def _on_train(self):
        self.train_btn.configure(state="disabled",
                                  text="Training...")
        self.train_status_lbl.configure(text="Training ML models... please wait")
        self.train_progress.set(0.1)
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        self.controller.train_models(
            progress_cb=self._update_train_progress,
            done_cb=self._on_train_done
        )

    def _update_train_progress(self, model_name: str, metrics: dict):
        prog_map = {"Naive Bayes": 0.3, "Random Forest": 0.55,
                    "SVM": 0.75, "Logistic Regression": 1.0}
        p = prog_map.get(model_name, 0.5)
        self.after(0, lambda: self.train_progress.set(p))
        self.after(0, lambda: self.train_status_lbl.configure(
            text=f"Trained: {model_name} · F1={metrics.get('f1_score', 0):.3f}"))

    def _on_train_done(self, results: dict):
        def _update():
            self.train_btn.configure(state="normal", text="⚙ Train ML")
            self.train_progress.set(1.0)
            # Update table
            for row in self.model_table.get_children():
                self.model_table.delete(row)
            result_list = list(results.values())
            for r in result_list:
                self.model_table.insert("", "end", values=(
                    r.get("model", "?"),
                    f"{float(r.get('accuracy', 0)):.4f}",
                    f"{float(r.get('precision', 0)):.4f}",
                    f"{float(r.get('recall', 0)):.4f}",
                    f"{float(r.get('f1_score', 0)):.4f}",
                ))
            self.model_chart.draw(result_list)
            
            best_model_name = self.controller.get_best_model_name()
            best_cm = None
            for r in result_list:
                if r.get("model") == best_model_name:
                    best_cm = r.get("confusion_matrix")
                    break
            
            if best_cm:
                self.cm_chart.draw(best_cm, best_model_name)

            self._log(f"✅ ML training complete. Best: {best_model_name}")
        self.after(0, _update)

    def _on_threshold_change(self, val):
        v = round(float(val))
        self.threshold_val_lbl.configure(text=str(v))
        self.controller.set_threshold(float(v))

    def _on_cooldown_change(self, val):
        secs = round(float(val))
        # Display nicely: under 60s show seconds, over show minutes
        if secs < 60:
            display = f"{secs}s"
        else:
            mins = secs // 60
            rem  = secs % 60
            display = f"{mins}m{rem:02d}s" if rem else f"{mins}m"
        self.cooldown_val_lbl.configure(text=display)
        self.controller.set_cooldown(float(secs))

    def _on_report(self):
        path = self.controller.save_report()
        messagebox.showinfo("Report Saved",
                            f"Session report saved to:\n{path}",
                            parent=self)

    def _on_close(self):
        self.controller.stop()
        self.destroy()

    # ── Update Methods (called from controller loop) ───────────────────────

    def update_metrics(self, result: dict):
        """Called from the monitoring loop with new prediction result."""
        score = result.get("score", 0)
        label = result.get("label", "Medium")
        features = result.get("features", {})
        timestamp = result.get("timestamp", time.time())

        color = LABEL_COLORS.get(label, TEXT_PRIMARY)

        # Score card
        self.score_label.configure(text=str(int(score)), text_color=color)
        self.status_label.configure(
            text=f"● {label.upper()} ATTENTION", text_color=color)
        self.score_bar.set(score / 100.0)
        self.score_bar.configure(progress_color=color)

        # Metric rows
        for key, (lbl, unit) in self._metric_labels.items():
            val = features.get(key, 0)
            if isinstance(val, float):
                text = f"{val:.2f} {unit}".strip()
            else:
                text = f"{val} {unit}".strip()
            lbl.configure(text=text)

        # Dashboard charts
        self.dashboard.update(
            timestamp=timestamp,
            score=score,
            idle=features.get("idle_time", 0),
            typing_speed=features.get("typing_speed", 0),
            label=label
        )

        # Update alert count
        self.alert_count_lbl.configure(
            text=f"Alerts fired: {self.controller.get_alert_count()}")

        # Dataset + model info
        stats = self.controller.get_dataset_stats()
        self.dataset_lbl.configure(
            text=f"Samples: {stats.get('total', 0)} "
                 f"(H:{stats.get('High',0)} M:{stats.get('Medium',0)} L:{stats.get('Low',0)})")
        self.model_lbl.configure(
            text=f"Model: {self.controller.get_best_model_name() or 'Not trained'}")

        # Log entry
        ts = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        self._log(f"[{ts}] Score={score:.1f} | {label} | "
                  f"Typing={features.get('typing_speed',0):.2f} | "
                  f"Idle={features.get('idle_time',0):.1f}s")

    def show_alert_popup(self, score: float, label: str):
        """Display attention alert popup."""
        def _show():
            if self._alert_popup and self._alert_popup.winfo_exists():
                return
            self._flash_warning()
            self._alert_popup = ctk.CTkToplevel(self)
            self._alert_popup.title("⚠ Attention Alert")
            self._alert_popup.geometry("400x200")
            self._alert_popup.configure(fg_color=BG_PANEL)
            self._alert_popup.attributes("-topmost", True)

            ctk.CTkLabel(
                self._alert_popup,
                text="⚠ LOW ATTENTION DETECTED",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=ACCENT_RED
            ).pack(pady=(20, 5))

            ctk.CTkLabel(
                self._alert_popup,
                text=f"Score: {score:.1f}  |  Status: {label}",
                font=ctk.CTkFont(size=13),
                text_color=ACCENT_ORANGE
            ).pack(pady=5)

            ctk.CTkLabel(
                self._alert_popup,
                text="Please refocus on your task.",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_DIM
            ).pack(pady=5)

            ctk.CTkButton(
                self._alert_popup, text="Dismiss",
                fg_color=ACCENT_RED, hover_color="#c0392b",
                command=self._alert_popup.destroy
            ).pack(pady=15)

        self.after(0, _show)

    def _flash_warning(self):
        """Flash the window border red briefly."""
        original = self.cget("fg_color")

        def flash(count):
            if count <= 0:
                self.configure(fg_color=original)
                return
            color = ACCENT_RED if count % 2 == 0 else original
            self.configure(fg_color=color)
            self.after(250, lambda: flash(count - 1))

        flash(6)

    def _log(self, message: str):
        """Append to activity log tab."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        # Trim to last 300 lines
        lines = self.log_text.get("1.0", "end").split("\n")
        if len(lines) > 300:
            self.log_text.delete("1.0", f"{len(lines)-300}.0")
        self.log_text.configure(state="disabled")

    def set_status(self, text: str):
        self.status_bar_lbl.configure(text=text)
