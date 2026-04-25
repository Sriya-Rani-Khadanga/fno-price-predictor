"""
Risk manager — enforces position limits, P&L limits, and time-based rules.
"""
from __future__ import annotations
from datetime import datetime, time
from typing import Dict, List, Optional
from config.settings import get_settings, RiskConfig

class RiskManager:
    """Enforces risk limits for the trading session."""

    def __init__(self, config: RiskConfig = None):
        settings = get_settings()
        self.config = config or settings.risk
        self._daily_realized_pnl = 0.0
        self._positions_count = 0
        self._capital_used = 0.0

    def check_entry(self, underlying: str, strike: float, qty: int,
                    margin_required: float, current_time: time = None) -> Dict:
        """Check if entry is allowed given current risk state."""
        reasons = []
        allowed = True
        if self._positions_count >= self.config.max_positions:
            allowed = False
            reasons.append(f"Max positions ({self.config.max_positions}) reached")
        if margin_required > self.config.max_capital_per_trade:
            allowed = False
            reasons.append(f"Margin ₹{margin_required:,.0f} exceeds limit ₹{self.config.max_capital_per_trade:,.0f}")
        if self._daily_realized_pnl < -self.config.max_daily_loss:
            allowed = False
            reasons.append(f"Daily loss limit breached")
        if current_time:
            settings = get_settings()
            inst = settings.instruments.get(underlying)
            if inst and not inst.can_open_position(current_time):
                allowed = False
                reasons.append(f"Trading restricted after {inst.no_new_positions_after}")
        return {"allowed": allowed, "reasons": reasons}

    def on_entry(self, margin: float):
        self._positions_count += 1
        self._capital_used += margin

    def on_exit(self, pnl: float, margin: float):
        self._positions_count -= 1
        self._capital_used -= margin
        self._daily_realized_pnl += pnl

    def should_force_close(self, current_time: time, underlying: str) -> bool:
        settings = get_settings()
        inst = settings.instruments.get(underlying)
        if inst:
            return inst.must_close_all(current_time)
        return False

    def reset(self):
        self._daily_realized_pnl = 0.0
        self._positions_count = 0
        self._capital_used = 0.0
