"""
PCP deviation prediction model using gradient boosted features.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

class PCPModel:
    """Simple online model that predicts if a PCP deviation will be profitable."""

    def __init__(self):
        self._observations: List[Dict] = []
        self._outcomes: List[float] = []

    def predict_profitability(self, features: Dict) -> float:
        """Predict probability that current deviation will be profitable after costs."""
        dev = features.get("max_deviation", 0)
        spread = features.get("avg_call_spread", 0) + features.get("avg_put_spread", 0)
        iv = features.get("atm_iv", 0.15)
        if dev < 0.2:
            return 0.1
        if spread / max(dev, 0.01) > 0.5:
            return 0.2
        base_prob = min(0.9, dev / 1.0)
        iv_adj = 1.0 - min(0.3, max(0, iv - 0.2))
        return base_prob * iv_adj

    def update(self, features: Dict, outcome: float):
        """Update model with observed outcome."""
        self._observations.append(features)
        self._outcomes.append(outcome)
        if len(self._observations) > 10000:
            self._observations = self._observations[-5000:]
            self._outcomes = self._outcomes[-5000:]
