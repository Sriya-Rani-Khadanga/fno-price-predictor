"""
Put-Call Parity calculator.
Computes theoretical PCP relationship and detects violations.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data.processors.options_chain import OptionChain, StrikeData

@dataclass
class PCPViolation:
    underlying: str
    strike: float
    expiry: str
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    spot: float
    theoretical_pcp: float
    actual_pcp: float
    deviation_pct: float
    deviation_rupees: float
    deviation_rupees_per_lot: float
    direction: str
    active_seconds: float
    trend: str
    confidence: float
    timestamp: datetime
    lot_size: int

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else
                    v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.__dict__.items()}

class PCPCalculator:
    def __init__(self, lot_sizes: Dict[str, int]):
        self.lot_sizes = lot_sizes
        self._violation_history: Dict[str, List[Tuple[float, float, datetime]]] = {}
        self._violation_start_times: Dict[str, datetime] = {}
        self._min_deviation_pct = 0.05

    def compute_theoretical_pcp(self, spot, strike, r, T):
        return spot - strike * math.exp(-r * T)

    def compute_deviation(self, chain: OptionChain, sd: StrikeData, T: float) -> PCPViolation:
        theoretical = self.compute_theoretical_pcp(chain.spot_price, sd.strike, chain.risk_free_rate, T)
        actual_mid = sd.call_mid - sd.put_mid
        deviation = actual_mid - theoretical
        deviation_pct = (deviation / chain.spot_price) * 100.0 if chain.spot_price > 0 else 0.0
        direction = "call_rich" if deviation > 0 else "put_rich"
        lot_size = self.lot_sizes.get(chain.underlying, 50)
        key = f"{chain.underlying}_{sd.strike}_{chain.expiry}"
        now = chain.timestamp
        if key not in self._violation_history:
            self._violation_history[key] = []
        self._violation_history[key].append((abs(deviation_pct), deviation, now))
        if len(self._violation_history[key]) > 20:
            self._violation_history[key] = self._violation_history[key][-20:]
        if abs(deviation_pct) > self._min_deviation_pct:
            if key not in self._violation_start_times:
                self._violation_start_times[key] = now
            active_seconds = (now - self._violation_start_times[key]).total_seconds()
        else:
            self._violation_start_times.pop(key, None)
            active_seconds = 0.0
        trend = self._compute_trend(key)
        confidence = self._compute_confidence(chain, sd)
        return PCPViolation(
            underlying=chain.underlying, strike=sd.strike, expiry=chain.expiry,
            call_bid=sd.call_bid, call_ask=sd.call_ask, put_bid=sd.put_bid, put_ask=sd.put_ask,
            spot=chain.spot_price, theoretical_pcp=theoretical, actual_pcp=actual_mid,
            deviation_pct=abs(deviation_pct), deviation_rupees=abs(deviation),
            deviation_rupees_per_lot=abs(deviation) * lot_size, direction=direction,
            active_seconds=active_seconds, trend=trend, confidence=confidence,
            timestamp=now, lot_size=lot_size)

    def compute_all_deviations(self, chain: OptionChain, T: float) -> List[PCPViolation]:
        violations = []
        for s in chain.strikes:
            v = self.compute_deviation(chain, s, T)
            s.pcp_deviation_pct = v.deviation_pct
            s.pcp_deviation_rupees = v.deviation_rupees
            violations.append(v)
        return violations

    def get_active_violations(self, chain, T, min_pct=0.1):
        return [v for v in self.compute_all_deviations(chain, T) if v.deviation_pct >= min_pct]

    def _compute_trend(self, key):
        h = self._violation_history.get(key, [])
        if len(h) < 3:
            return "stable"
        recent = [x[0] for x in h[-5:]]
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.01: return "widening"
        elif slope < -0.01: return "narrowing"
        return "stable"

    def _compute_confidence(self, chain, strike):
        c = 1.0
        if chain.last_update_seconds_ago > 5:
            c *= max(0.3, 1.0 - (chain.last_update_seconds_ago - 5) / 30.0)
        oi = min(strike.call_oi, strike.put_oi)
        if oi < 10: c *= 0.2
        elif oi < 100: c *= 0.5
        elif oi < 1000: c *= 0.8
        if strike.call_ask > 0 and strike.put_ask > 0:
            cs = strike.call_spread / strike.call_mid if strike.call_mid > 0 else 1.0
            ps = strike.put_spread / strike.put_mid if strike.put_mid > 0 else 1.0
            if (cs + ps) / 2 > 0.05:
                c *= max(0.3, 1.0 - (cs + ps) / 2)
        return round(max(0.0, min(1.0, c)), 3)

    def reset(self):
        self._violation_history.clear()
        self._violation_start_times.clear()
