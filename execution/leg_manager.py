"""
Leg manager — manages multi-leg option positions for PCP arbitrage.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class ArbLeg:
    leg_id: str
    instrument: str
    option_type: str  # "call" or "put"
    direction: str  # "long" or "short"
    strike: float
    qty: int
    entry_price: float
    current_price: float
    entry_time: datetime

class LegManager:
    """Manages multi-leg arbitrage positions."""

    def __init__(self):
        self._legs: Dict[str, List[ArbLeg]] = {}  # position_id -> legs

    def add_position(self, position_id: str, underlying: str, strike: float,
                     qty: int, action_type: str, call_price: float, put_price: float):
        """Add a new arb position with two legs."""
        if action_type == "enter_long_call_short_put":
            legs = [
                ArbLeg(f"{position_id}_c", underlying, "call", "long", strike, qty,
                       call_price, call_price, datetime.now()),
                ArbLeg(f"{position_id}_p", underlying, "put", "short", strike, qty,
                       put_price, put_price, datetime.now()),
            ]
        else:
            legs = [
                ArbLeg(f"{position_id}_c", underlying, "call", "short", strike, qty,
                       call_price, call_price, datetime.now()),
                ArbLeg(f"{position_id}_p", underlying, "put", "long", strike, qty,
                       put_price, put_price, datetime.now()),
            ]
        self._legs[position_id] = legs

    def update_prices(self, position_id: str, call_price: float, put_price: float):
        """Update current prices for a position's legs."""
        if position_id in self._legs:
            for leg in self._legs[position_id]:
                if leg.option_type == "call":
                    leg.current_price = call_price
                else:
                    leg.current_price = put_price

    def get_position_pnl(self, position_id: str, lot_size: int) -> float:
        """Compute unrealized P&L for a position."""
        legs = self._legs.get(position_id, [])
        pnl = 0.0
        for leg in legs:
            diff = leg.current_price - leg.entry_price
            if leg.direction == "short":
                diff = -diff
            pnl += diff * leg.qty * lot_size
        return pnl

    def remove_position(self, position_id: str):
        """Remove a closed position."""
        self._legs.pop(position_id, None)

    def get_all_positions(self) -> Dict[str, List[ArbLeg]]:
        return dict(self._legs)

    def reset(self):
        self._legs.clear()
