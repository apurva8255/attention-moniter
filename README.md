# Privacy-Preserving Human Attention Monitoring System
### Using Behavioral Analytics — Final Year Major Project

> **Zero camera. Zero microphone. 100% privacy.**
> Monitors user attention through keyboard, mouse & system behavioral signals.

---

## 📁 Project Structure

```
attention_monitor/
├── main.py                # Entry point & central controller
├── activity_collector.py  # Keyboard / mouse / system event capture
├── feature_engine.py      # Feature extraction & normalization
├── dataset_builder.py     # CSV dataset management & synthetic generation
├── ml_model.py            # ML training (NB, RF, SVM, LR) & comparison
├── attention_engine.py    # Score computation pipeline
├── alert_system.py        # Threshold alerts with sound & popup
├── ui.py                  # CustomTkinter dark-themed desktop UI
├── dashboard.py           # Matplotlib embedded graphs
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
python main.py
```

---

## 🔄 Pipeline

```
User Activity
  → Event Listener       (activity_collector.py)
  → Feature Extraction   (feature_engine.py)
  → Dataset CSV          (dataset_builder.py)
  → ML Training          (ml_model.py)
  → Model Comparison     (ml_model.py)
  → Prediction           (attention_engine.py)
  → Attention Score      (attention_engine.py)
  → Alert Check          (alert_system.py)
  → UI Dashboard         (ui.py + dashboard.py)
  → Logs + Graphs        (ui.py + dashboard.py)
```

---

## 📊 Features Captured

| Feature | Description |
|---|---|
| `typing_speed` | Keystrokes per second |
| `pause_time` | Average pause between keystrokes |
| `mouse_speed` | Average mouse pixel velocity |
| `click_rate` | Mouse clicks per second |
| `idle_time` | Seconds since last activity |
| `app_switch_freq` | App/window switches per minute |
| `error_rate` | Backspace ratio to total keystrokes |
| `cursor_velocity` | Mouse distance per second |
| `movement_smoothness` | Uniformity of mouse movement (0–1) |

---

## 🤖 ML Models

| Model | Notes |
|---|---|
| Naive Bayes | Fast, probabilistic baseline |
| Random Forest | Ensemble, handles non-linearity |
| SVM (RBF kernel) | Strong with small-medium datasets |
| Logistic Regression | Linear, interpretable |

Best model selected by **weighted F1 score** and saved as `best_model.joblib`.

---

## 🎯 Attention Labels

| Label | Score Range | Meaning |
|---|---|---|
| 🟢 High | 65–100 | Focused, active typing, smooth mouse |
| 🟡 Medium | 35–64 | Partial focus, some pauses |
| 🔴 Low | 0–34 | Idle, distracted, high error rate |

---

## ⚠ Alert System

- Fires when score drops below **user-configurable threshold**
- 30-second cooldown between alerts
- Shows popup dialog + plays beep sound
- Window border flash effect

---

## 📈 Dashboard Graphs

1. **Attention Score vs Time** — colored by label
2. **Idle Time Per Window** — bar chart
3. **Typing Speed vs Time** — area line chart

---

## 📋 Reports

Click **📊 Report** to save a CSV file:
- Average attention score
- Focus / distracted time split
- Per-window history with timestamps

---

## 🔒 Privacy Guarantees

- **No camera or microphone** used at any point
- **No keystroke content** is recorded — only counts and timing
- All data stays **local on your machine**
- Dataset CSV contains only anonymized behavioral metadata
