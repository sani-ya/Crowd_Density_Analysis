"""
anomaly.py — Anomaly detection for crowd monitoring.

Detects:
  - Sudden spike in crowd count (>30 % increase over rolling average).
  - Overcrowding beyond the configured threshold.

When an anomaly is detected the module:
  1. Logs the event to alerts.log via AlertLogger.
  2. Prints a warning to the terminal.
  3. Plays a beep sound (Windows, best-effort).
  4. Draws a large red warning overlay on the video frame.
"""

import collections
import threading
import time

import cv2
import numpy as np

from utils import Config, AlertLogger, draw_text, play_beep


class AnomalyDetector:
    """Rolling-window anomaly detector for crowd counts."""

    def __init__(self):
        self._alert_logger = AlertLogger()
        # Deque stores (timestamp, count) tuples
        self._history: collections.deque = collections.deque()
        self._last_alert_time: float = 0.0
        self._alert_cooldown: float = 3.0  # seconds between repeated alerts

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _rolling_average(self) -> float:
        """Compute the average count over the time window."""
        if not self._history:
            return 0.0
        return sum(c for _, c in self._history) / len(self._history)

    def _prune_history(self) -> None:
        """Remove entries older than the configured time window."""
        cutoff = time.time() - Config.ANOMALY_WINDOW_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def check(self, count: int) -> list[str]:
        """
        Check the current *count* for anomalies.

        Returns:
            A (possibly empty) list of anomaly message strings.
        """
        now = time.time()
        self._prune_history()

        anomalies: list[str] = []
        avg = self._rolling_average()

        # --- Sudden spike ---
        if avg > 0 and count > avg * (1 + Config.ANOMALY_SPIKE_PERCENT):
            anomalies.append(
                f"SPIKE DETECTED: count {count} vs avg {avg:.1f} "
                f"(+{((count - avg) / avg) * 100:.0f}%)"
            )

        # --- Overcrowding ---
        if count > Config.OVERCROWDING_THRESHOLD:
            anomalies.append(
                f"OVERCROWDING: {count} people (threshold {Config.OVERCROWDING_THRESHOLD})"
            )

        # Log (with cooldown to avoid spam)
        if anomalies and (now - self._last_alert_time) > self._alert_cooldown:
            self._last_alert_time = now
            for msg in anomalies:
                self._alert_logger.log(msg)
            # Play beep in a separate thread to avoid blocking the video loop
            threading.Thread(target=play_beep, daemon=True).start()

        # Append current count to history
        self._history.append((now, count))

        return anomalies

    # --------------------------------------------------------------------- #
    #  Drawing                                                               #
    # --------------------------------------------------------------------- #

    @staticmethod
    def draw_warning(frame: np.ndarray, messages: list[str]) -> None:
        """Draw large red warning text in the centre of *frame*."""
        if not messages:
            return

        h, w = frame.shape[:2]

        # Semi-transparent red overlay bar
        overlay = frame.copy()
        bar_h = 50 + 35 * len(messages)
        y_start = h // 2 - bar_h // 2
        cv2.rectangle(overlay, (0, y_start), (w, y_start + bar_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Warning header
        draw_text(frame, "!! WARNING !!", (w // 2 - 120, y_start + 30),
                  color=(255, 255, 255), scale=1.0, thickness=3)

        # Individual messages
        for i, msg in enumerate(messages):
            draw_text(frame, msg, (30, y_start + 65 + i * 30),
                      color=(255, 255, 255), scale=0.55, thickness=2)
