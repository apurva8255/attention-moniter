"""
activity_collector.py
Captures keyboard, mouse, and system activity events without camera.
Privacy-preserving: only behavioral metadata is collected, no content.
"""

import time
import threading
import psutil
from collections import deque

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


class ActivityCollector:
    """Collects raw behavioral signals from input devices and OS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_counters()
        self._start_time = time.time()
        self._last_activity_time = time.time()
        self._last_key_time = None
        self._pause_durations = []
        self._mouse_positions = deque(maxlen=200)
        self._mouse_speeds = deque(maxlen=100)
        self._last_mouse_pos = None
        self._last_mouse_time = None
        self._active_window = ""
        self._app_switches = 0
        self._listeners = []
        self._running = False
        self._session_start = time.time()

    def _reset_counters(self):
        self._keystrokes = 0
        self._backspaces = 0
        self._clicks = 0
        self._double_clicks = 0
        self._mouse_distance = 0.0
        self._idle_seconds = 0.0
        self._app_switches = 0
        self._pause_durations = []
        self._mouse_speeds = deque(maxlen=100)
        self._mouse_positions = deque(maxlen=200)

    def start(self):
        """Start all listeners."""
        self._running = True
        self._start_time = time.time()
        self._last_activity_time = time.time()

        if PYNPUT_AVAILABLE:
            try:
                kb_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release
                )
                kb_listener.daemon = True
                kb_listener.start()
                self._listeners.append(kb_listener)

                m_listener = mouse.Listener(
                    on_move=self._on_mouse_move,
                    on_click=self._on_mouse_click,
                    on_scroll=self._on_mouse_scroll
                )
                m_listener.daemon = True
                m_listener.start()
                self._listeners.append(m_listener)
            except Exception as e:
                print(f"[ActivityCollector] Listener error: {e}")

        # App-switch monitor thread
        t = threading.Thread(target=self._monitor_active_window, daemon=True)
        t.start()

    def stop(self):
        """Stop all listeners."""
        self._running = False
        for lst in self._listeners:
            try:
                lst.stop()
            except Exception:
                pass
        self._listeners.clear()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key_press(self, key):
        now = time.time()
        with self._lock:
            self._last_activity_time = now
            self._keystrokes += 1
            try:
                if key == keyboard.Key.backspace:
                    self._backspaces += 1
            except Exception:
                pass
            if self._last_key_time is not None:
                gap = now - self._last_key_time
                if gap > 1.0:
                    self._pause_durations.append(gap)
            self._last_key_time = now

    def _on_key_release(self, key):
        pass

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _on_mouse_move(self, x, y):
        now = time.time()
        with self._lock:
            self._last_activity_time = now
            if self._last_mouse_pos is not None:
                lx, ly = self._last_mouse_pos
                dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                self._mouse_distance += dist
                dt = now - self._last_mouse_time
                if dt > 0:
                    speed = dist / dt
                    self._mouse_speeds.append(speed)
            self._mouse_positions.append((x, y, now))
            self._last_mouse_pos = (x, y)
            self._last_mouse_time = now

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:
            with self._lock:
                self._last_activity_time = time.time()
                self._clicks += 1

    def _on_mouse_scroll(self, x, y, dx, dy):
        with self._lock:
            self._last_activity_time = time.time()

    # ── App Switch Monitor ────────────────────────────────────────────────────

    def _monitor_active_window(self):
        """Poll for foreground process changes to detect app switching."""
        while self._running:
            try:
                # Cross-platform: get top CPU process as proxy for active app
                procs = [(p.info['name'], p.info['cpu_percent'])
                         for p in psutil.process_iter(['name', 'cpu_percent'])
                         if p.info['cpu_percent'] and p.info['cpu_percent'] > 0]
                if procs:
                    top = max(procs, key=lambda x: x[1])[0]
                    with self._lock:
                        if top != self._active_window and self._active_window:
                            self._app_switches += 1
                        self._active_window = top
            except Exception:
                pass
            time.sleep(2)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return a snapshot of all collected metrics and reset counters."""
        now = time.time()
        with self._lock:
            elapsed = max(now - self._start_time, 1.0)
            idle = now - self._last_activity_time

            avg_mouse_speed = (
                sum(self._mouse_speeds) / len(self._mouse_speeds)
                if self._mouse_speeds else 0.0
            )

            avg_pause = (
                sum(self._pause_durations) / len(self._pause_durations)
                if self._pause_durations else 0.0
            )

            # Movement smoothness: inverse of speed variance
            if len(self._mouse_speeds) > 2:
                speeds = list(self._mouse_speeds)
                mean_s = sum(speeds) / len(speeds)
                variance = sum((s - mean_s) ** 2 for s in speeds) / len(speeds)
                smoothness = 1.0 / (1.0 + variance ** 0.5)
            else:
                smoothness = 1.0

            snapshot = {
                "timestamp": now,
                "elapsed": elapsed,
                "keystrokes": self._keystrokes,
                "backspaces": self._backspaces,
                "clicks": self._clicks,
                "mouse_distance": self._mouse_distance,
                "avg_mouse_speed": avg_mouse_speed,
                "idle_time": idle,
                "app_switches": self._app_switches,
                "avg_pause": avg_pause,
                "smoothness": smoothness,
                "pause_count": len(self._pause_durations),
            }

            # Reset for next window
            self._keystrokes = 0
            self._backspaces = 0
            self._clicks = 0
            self._mouse_distance = 0.0
            self._mouse_speeds.clear()
            self._pause_durations = []
            self._app_switches = 0
            self._start_time = now

            return snapshot
