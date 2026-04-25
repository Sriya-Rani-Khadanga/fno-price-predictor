"""
Transaction cost calculator for NSE equity options and MCX commodity futures.
Models all regulatory charges including the critical STT trap on exercise.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from config.settings import get_settings, CostModelConfig

@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    sebi_charges: float
    exchange_txn_charges: float
    gst: float
    stamp_duty: float
    slippage: float
    ctt: float
    total: float

    def to_dict(self) -> Dict:
        return self.__dict__

@dataclass
class NetArbResult:
    gross_violation_pct: float
    gross_profit_per_lot: float
    costs: CostBreakdown
    net_profit_per_lot: float
    is_profitable: bool
    breakeven_violation_pct: float
    margin_over_breakeven_pct: float

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["costs"] = self.costs.to_dict()
        return d

class TransactionCostCalculator:
    """Full cost model for NSE equity options and MCX commodity futures."""

    def __init__(self):
        self.settings = get_settings()

    def _get_cost_model(self, underlying: str) -> CostModelConfig:
        inst = self.settings.instruments.get(underlying)
        if inst is None:
            return self.settings.cost_models["equity_options"]
        return self.settings.cost_models.get(inst.cost_model, self.settings.cost_models["equity_options"])

    def calculate_entry_costs(self, underlying: str, premium: float, qty: int, lot_size: int) -> CostBreakdown:
        cm = self._get_cost_model(underlying)
        turnover = premium * qty * lot_size
        brokerage = cm.brokerage_per_order * 2  # buy call + sell put (or vice versa)
        stt = 0.0  # no STT on buy side for options
        sebi = turnover * cm.sebi_charges
        exch = turnover * cm.exchange_txn_charges
        stamp = turnover * cm.stamp_duty
        gst_base = brokerage + exch + sebi
        gst = gst_base * cm.gst
        slippage = turnover * (cm.slippage_bps / 10000.0)
        ctt = turnover * cm.ctt if cm.ctt > 0 else 0.0
        total = brokerage + stt + sebi + exch + gst + stamp + slippage + ctt
        return CostBreakdown(brokerage=round(brokerage, 2), stt=round(stt, 2),
                             sebi_charges=round(sebi, 2), exchange_txn_charges=round(exch, 2),
                             gst=round(gst, 2), stamp_duty=round(stamp, 2),
                             slippage=round(slippage, 2), ctt=round(ctt, 2), total=round(total, 2))

    def calculate_exit_costs(self, underlying: str, premium: float, qty: int,
                             lot_size: int, is_exercise: bool = False,
                             intrinsic_value: float = 0.0) -> CostBreakdown:
        cm = self._get_cost_model(underlying)
        turnover = premium * qty * lot_size
        brokerage = cm.brokerage_per_order * 2
        if is_exercise and cm.stt_on_exercise > 0:
            stt = intrinsic_value * qty * lot_size * cm.stt_on_exercise
        else:
            stt = turnover * cm.stt_on_sell
        sebi = turnover * cm.sebi_charges
        exch = turnover * cm.exchange_txn_charges
        stamp = 0.0
        gst_base = brokerage + exch + sebi
        gst = gst_base * cm.gst
        slippage = turnover * (cm.slippage_bps / 10000.0)
        ctt = turnover * cm.ctt if cm.ctt > 0 else 0.0
        total = brokerage + stt + sebi + exch + gst + stamp + slippage + ctt
        return CostBreakdown(brokerage=round(brokerage, 2), stt=round(stt, 2),
                             sebi_charges=round(sebi, 2), exchange_txn_charges=round(exch, 2),
                             gst=round(gst, 2), stamp_duty=round(stamp, 2),
                             slippage=round(slippage, 2), ctt=round(ctt, 2), total=round(total, 2))

    def calculate_full_arb_costs(self, underlying: str, strike: float, spot: float,
                                  expiry_days: int, qty: int,
                                  gross_violation_pct: float) -> NetArbResult:
        inst = self.settings.instruments.get(underlying)
        lot_size = inst.lot_size if inst else 50
        gross_per_lot = spot * (gross_violation_pct / 100.0) * lot_size
        avg_premium = spot * 0.03
        entry = self.calculate_entry_costs(underlying, avg_premium, qty, lot_size)
        exit_c = self.calculate_exit_costs(underlying, avg_premium, qty, lot_size)
        total = CostBreakdown(
            brokerage=entry.brokerage + exit_c.brokerage,
            stt=entry.stt + exit_c.stt,
            sebi_charges=entry.sebi_charges + exit_c.sebi_charges,
            exchange_txn_charges=entry.exchange_txn_charges + exit_c.exchange_txn_charges,
            gst=entry.gst + exit_c.gst,
            stamp_duty=entry.stamp_duty + exit_c.stamp_duty,
            slippage=entry.slippage + exit_c.slippage,
            ctt=entry.ctt + exit_c.ctt,
            total=entry.total + exit_c.total)
        net = gross_per_lot * qty - total.total
        be_pct = (total.total / (spot * lot_size * qty)) * 100.0 if spot > 0 else 999.0
        return NetArbResult(
            gross_violation_pct=gross_violation_pct, gross_profit_per_lot=round(gross_per_lot, 2),
            costs=total, net_profit_per_lot=round(net, 2), is_profitable=net > 0,
            breakeven_violation_pct=round(be_pct, 4),
            margin_over_breakeven_pct=round(gross_violation_pct - be_pct, 4))

    def simulate_stt_trap(self, underlying: str, strike: float, spot: float,
                           expiry_days: int, qty: int,
                           hold_to_expiry: bool) -> Dict:
        inst = self.settings.instruments.get(underlying)
        lot_size = inst.lot_size if inst else 50
        cm = self._get_cost_model(underlying)
        avg_premium = max(abs(spot - strike) * 0.5, spot * 0.01)
        exit_before = self.calculate_exit_costs(underlying, avg_premium, qty, lot_size, False, 0.0)
        intrinsic = max(spot - strike, 0.0)
        exercise_stt = intrinsic * qty * lot_size * cm.stt_on_exercise
        exit_exercise = self.calculate_exit_costs(underlying, avg_premium, qty, lot_size, True, intrinsic)
        trap_magnitude = exit_exercise.total - exit_before.total
        return {
            "exit_before_expiry": exit_before.to_dict(),
            "exit_at_expiry_exercise": exit_exercise.to_dict(),
            "stt_on_exercise": round(exercise_stt, 2),
            "trap_magnitude": round(trap_magnitude, 2),
            "is_trap": trap_magnitude > 0,
            "recommendation": "EXIT BEFORE EXPIRY" if trap_magnitude > 100 else "HOLD OK",
            "intrinsic_value": round(intrinsic, 2),
            "hold_to_expiry": hold_to_expiry}

    def get_breakeven_violation(self, underlying: str, strike: float, spot: float,
                                 expiry_days: int, qty: int,
                                 current_deviation_pct: float) -> Dict:
        result = self.calculate_full_arb_costs(underlying, strike, spot, expiry_days, qty, current_deviation_pct)
        return {
            "breakeven_pct": result.breakeven_violation_pct,
            "breakeven_rupees": round(result.costs.total, 2),
            "current_deviation_pct": current_deviation_pct,
            "margin_over_breakeven_pct": result.margin_over_breakeven_pct,
            "enter_recommended": result.margin_over_breakeven_pct > 0.05}
