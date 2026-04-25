"""
Persistence model — predicts how long a PCP deviation will persist.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

class PersistenceModel:
    """Predicts expected duration of a PCP violation based on features."""

    def __init__(self):
        self._durations: list = []

    def predict_duration(self, features: Dict) -> float:
        """Predict expected remaining duration of violation in seconds."""
        dev = features.get("max_deviation", 0)
        iv = features.get("atm_iv", 0.15)
        active_s = features.get("active_seconds", 0)
        base_duration = 60.0 + dev * 100.0
        iv_factor = 1.0 + (iv - 0.15) * 5
        remaining = max(0, base_duration * iv_factor - active_s)
        return remaining

    def update(self, features: Dict, actual_duration: float):
        self._durations.append(actual_duration)
        if len(self._durations) > 5000:
            self._durations = self._durations[-2500:]
