"""
Order simulator — simulates order execution with realistic fill delays and partial fills.
"""
from __future__ import annotations
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from data.processors.options_chain import OptionChain, StrikeData
from data.processors.cost_calculator import TransactionCostCalculator
from config.settings import get_settings

@dataclass
class OrderFill:
    order_id: str
    underlying: str
    strike: float
    qty_requested: int
    qty_filled: int
    fill_price_call: float
    fill_price_put: float
    action_type: str
    fill_delay_ms: float
    slippage_pct: float
    timestamp: datetime
    is_partial: bool

@dataclass
class SimulatedPosition:
    position_id: str
    underlying: str
    strike: float
    qty: int
    action_type: str
    entry_fill: OrderFill
    entry_deviation_pct: float
    entry_time: datetime
    lot_size: int

class OrderSimulator:
    """Simulates order execution with realistic delays and partial fills."""

    def __init__(self):
        self.settings = get_settings()
        self.cost_calc = TransactionCostCalculator()
        self._positions: Dict[str, SimulatedPosition] = {}
        self._fills: List[OrderFill] = []
        self._realized_pnl = 0.0

    def execute_entry(self, chain: OptionChain, strike: float, qty: int,
                       action_type: str, deviation_pct: float) -> Optional[OrderFill]:
        """Execute an entry order. Returns fill or None if rejected."""
        sd = chain.get_strike(strike)
        if sd is None:
            return None
        inst = self.settings.instruments.get(chain.underlying)
        lot_size = inst.lot_size if inst else 50
        delay_ms = random.uniform(200, 2000)
        fill_pct = 1.0
        if sd.call_oi < 50 or sd.put_oi < 50:
            fill_pct = min(1.0, sd.call_oi / 100.0)
        qty_filled = max(1, int(qty * fill_pct))
        slippage = random.uniform(0.0001, 0.001)
        if action_type == "enter_long_call_short_put":
            fill_call = sd.call_ask * (1 + slippage)
            fill_put = sd.put_bid * (1 - slippage)
        else:
            fill_call = sd.call_bid * (1 - slippage)
            fill_put = sd.put_ask * (1 + slippage)
        order_id = str(uuid.uuid4())[:8]
        fill = OrderFill(
            order_id=order_id, underlying=chain.underlying, strike=strike,
            qty_requested=qty, qty_filled=qty_filled,
            fill_price_call=round(fill_call, 2), fill_price_put=round(fill_put, 2),
            action_type=action_type, fill_delay_ms=round(delay_ms, 1),
            slippage_pct=round(slippage * 100, 4), timestamp=datetime.now(),
            is_partial=qty_filled < qty)
        self._fills.append(fill)
        pos_id = f"pos_{order_id}"
        self._positions[pos_id] = SimulatedPosition(
            position_id=pos_id, underlying=chain.underlying, strike=strike,
            qty=qty_filled, action_type=action_type, entry_fill=fill,
            entry_deviation_pct=deviation_pct, entry_time=datetime.now(),
            lot_size=lot_size)
        return fill

    def execute_exit(self, position_id: str, chain: OptionChain) -> Optional[Dict]:
        """Exit a position. Returns exit details or None."""
        pos = self._positions.get(position_id)
        if pos is None:
            return None
        sd = chain.get_strike(pos.strike)
        if sd is None:
            return None
        slippage = random.uniform(0.0001, 0.001)
        if "long_call" in pos.action_type:
            exit_call = sd.call_bid * (1 - slippage)
            exit_put = sd.put_ask * (1 + slippage)
            call_pnl = (exit_call - pos.entry_fill.fill_price_call) * pos.qty * pos.lot_size
            put_pnl = (pos.entry_fill.fill_price_put - exit_put) * pos.qty * pos.lot_size
        else:
            exit_call = sd.call_ask * (1 + slippage)
            exit_put = sd.put_bid * (1 - slippage)
            call_pnl = (pos.entry_fill.fill_price_call - exit_call) * pos.qty * pos.lot_size
            put_pnl = (exit_put - pos.entry_fill.fill_price_put) * pos.qty * pos.lot_size
        gross_pnl = call_pnl + put_pnl
        exit_costs = self.cost_calc.calculate_exit_costs(
            pos.underlying, (exit_call + exit_put) / 2, pos.qty, pos.lot_size)
        entry_costs = self.cost_calc.calculate_entry_costs(
            pos.underlying, (pos.entry_fill.fill_price_call + pos.entry_fill.fill_price_put) / 2,
            pos.qty, pos.lot_size)
        net_pnl = gross_pnl - exit_costs.total - entry_costs.total
        self._realized_pnl += net_pnl
        del self._positions[position_id]
        return {
            "position_id": position_id, "gross_pnl": round(gross_pnl, 2),
            "entry_costs": entry_costs.total, "exit_costs": exit_costs.total,
            "net_pnl": round(net_pnl, 2), "holding_seconds": round(pos.entry_fill.timestamp.timestamp() - datetime.now().timestamp(), 1),
            "exit_call_price": round(exit_call, 2), "exit_put_price": round(exit_put, 2)}

    def exit_all(self, chain: OptionChain) -> List[Dict]:
        """Exit all open positions."""
        results = []
        for pid in list(self._positions.keys()):
            result = self.execute_exit(pid, chain)
            if result:
                results.append(result)
        return results

    def update_positions(self, chain: OptionChain):
        """Update current prices for all positions from latest chain."""
        for pos in self._positions.values():
            if pos.underlying == chain.underlying:
                sd = chain.get_strike(pos.strike)
                if sd:
                    pos.entry_fill.fill_price_call  # entry stays same

    @property
    def positions(self) -> Dict[str, SimulatedPosition]:
        return self._positions

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    def reset(self):
        """Reset for new session."""
        self._positions.clear()
        self._fills.clear()
        self._realized_pnl = 0.0
