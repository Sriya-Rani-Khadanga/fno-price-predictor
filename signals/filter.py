"""
Signal filter — applies risk and quality filters to raw signals.
"""
from __future__ import annotations
from typing import Dict, List
from signals.signal_generator import ArbSignal
from config.settings import get_settings

class SignalFilter:
    """Filters signals based on risk limits, quality thresholds, and market conditions."""

    def __init__(self):
        self.settings = get_settings()
        self.min_confidence = 0.3
        self.min_margin_over_be = 0.05
        self.max_active_seconds = 180
        self.exclude_stt_risk = False

    def filter(self, signals: List[ArbSignal]) -> List[ArbSignal]:
        """Apply all filters to a list of signals."""
        filtered = []
        for s in signals:
            if s.confidence < self.min_confidence:
                continue
            if s.margin_over_breakeven < self.min_margin_over_be:
                continue
            if s.active_seconds > self.max_active_seconds:
                continue
            if self.exclude_stt_risk and s.stt_risk:
                continue
            if s.trend == "narrowing" and s.active_seconds > 60:
                continue
            filtered.append(s)
        return filtered
