"""
Signal generator — detects actionable PCP arbitrage signals.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from data.processors.options_chain import OptionChain
from data.processors.pcp_calculator import PCPCalculator, PCPViolation
from data.processors.cost_calculator import TransactionCostCalculator
from config.settings import get_settings

@dataclass
class ArbSignal:
    underlying: str
    strike: float
    expiry: str
    direction: str
    gross_deviation_pct: float
    net_deviation_pct: float
    breakeven_pct: float
    margin_over_breakeven: float
    estimated_net_profit: float
    confidence: float
    trend: str
    active_seconds: float
    stt_risk: bool
    timestamp: datetime
    action_type: str

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else
                    v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.__dict__.items()}

class SignalGenerator:
    """Generates actionable arbitrage signals from PCP violations."""

    def __init__(self):
        self.settings = get_settings()
        lots = {s: i.lot_size for s, i in self.settings.instruments.items()}
        self.pcp_calc = PCPCalculator(lots)
        self.cost_calc = TransactionCostCalculator()
        self._signal_history: List[ArbSignal] = []

    def scan(self, chain: OptionChain, T: float, min_pct: float = 0.1) -> List[ArbSignal]:
        """Scan an option chain for actionable arbitrage signals."""
        violations = self.pcp_calc.get_active_violations(chain, T, min_pct)
        signals = []
        for v in violations:
            inst = self.settings.instruments.get(v.underlying)
            lot = inst.lot_size if inst else 50
            arb_result = self.cost_calc.calculate_full_arb_costs(
                v.underlying, v.strike, v.spot, int(T * 365), 1, v.deviation_pct)
            action = "enter_long_call_short_put" if v.direction == "put_rich" else "enter_short_call_long_put"
            days_to_exp = max(1, int(T * 365))
            stt_risk = days_to_exp <= 3 and abs(v.spot - v.strike) / v.spot < 0.02
            signal = ArbSignal(
                underlying=v.underlying, strike=v.strike, expiry=v.expiry,
                direction=v.direction, gross_deviation_pct=v.deviation_pct,
                net_deviation_pct=arb_result.margin_over_breakeven_pct,
                breakeven_pct=arb_result.breakeven_violation_pct,
                margin_over_breakeven=arb_result.margin_over_breakeven_pct,
                estimated_net_profit=arb_result.net_profit_per_lot,
                confidence=v.confidence, trend=v.trend,
                active_seconds=v.active_seconds, stt_risk=stt_risk,
                timestamp=v.timestamp, action_type=action)
            signals.append(signal)
            self._signal_history.append(signal)
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]
        return sorted(signals, key=lambda s: s.margin_over_breakeven, reverse=True)

    def get_best_signal(self, chain: OptionChain, T: float) -> Optional[ArbSignal]:
        """Get the single best actionable signal."""
        signals = self.scan(chain, T)
        profitable = [s for s in signals if s.margin_over_breakeven > 0.05 and s.confidence > 0.5]
        return profitable[0] if profitable else None
