"""
alert_system.py
Manages attention threshold alerts: popup, sound, and visual flash.
"""

import threading
import time
import platform

# Try audio libraries
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

WINSOUND_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except Exception:
        pass


class AlertSystem:
    """
    Monitors attention score and fires alerts when below threshold.
    Supports cooldown to avoid alert spam.
    """

    def __init__(self, threshold: float = 40.0, cooldown: float = 30.0):
        """
        Args:
            threshold: Score below which alerts fire (0–100)
            cooldown: Minimum seconds between consecutive alerts
        """
        self.threshold = threshold
        self.cooldown = cooldown
        self._last_alert_time = 0.0
        self._alert_callback = None  # UI callback for popup/flash
        self._lock = threading.Lock()
        self._alert_count = 0

    def set_threshold(self, value: float):
        with self._lock:
            self.threshold = max(0.0, min(100.0, float(value)))

    def set_cooldown(self, seconds: float):
        """Set minimum time (in seconds) between consecutive alerts."""
        with self._lock:
            self.cooldown = max(5.0, float(seconds))

    def set_alert_callback(self, callback):
        """Register a UI callback: callback(score, label) → None"""
        self._alert_callback = callback

    def check(self, score: float, label: str) -> bool:
        """
        Check if alert should fire.

        Returns:
            True if alert was triggered.
        """
        with self._lock:
            now = time.time()
            if score < self.threshold and (now - self._last_alert_time) >= self.cooldown:
                self._last_alert_time = now
                self._alert_count += 1
                # Fire alert in background thread
                t = threading.Thread(
                    target=self._fire_alert,
                    args=(score, label),
                    daemon=True
                )
                t.start()
                return True
        return False

    def _fire_alert(self, score: float, label: str):
        """Execute all alert mechanisms."""
        self._play_sound()
        if self._alert_callback:
            try:
                self._alert_callback(score, label)
            except Exception as e:
                print(f"[AlertSystem] Callback error: {e}")

    def _play_sound(self):
        """Play alert beep using available audio library."""
        try:
            if WINSOUND_AVAILABLE:
                winsound.Beep(880, 400)
                time.sleep(0.1)
                winsound.Beep(660, 300)
            elif PYGAME_AVAILABLE:
                self._pygame_beep()
            else:
                # Terminal bell fallback
                print("\a\a", end="", flush=True)
        except Exception as e:
            print(f"[AlertSystem] Sound error: {e}")

    def _pygame_beep(self):
        """Generate a beep tone using pygame synthesizer."""
        try:
            import numpy as np
            sample_rate = 44100
            duration = 0.4
            freq = 880
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
            time.sleep(duration + 0.05)
        except Exception as e:
            print(f"[AlertSystem] Pygame beep error: {e}")

    def get_alert_count(self) -> int:
        return self._alert_count

    def reset_count(self):
        with self._lock:
            self._alert_count = 0
