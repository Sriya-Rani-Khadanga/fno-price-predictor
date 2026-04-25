"""
MCP Risk Server — FastAPI app on port 8002.
Provides risk management tool endpoints for the LLM agent.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from config.settings import get_settings

app = FastAPI(title="PCP Arb Risk MCP Server", version="1.0.0")

@dataclass
class Position:
    position_id: str
    underlying: str
    strike: float
    qty: int
    action_type: str
    entry_price_call: float
    entry_price_put: float
    entry_deviation_pct: float
    entry_time: datetime
    current_price_call: float = 0.0
    current_price_put: float = 0.0
    current_deviation_pct: float = 0.0
    lot_size: int = 50

    def unrealized_pnl(self) -> float:
        call_pnl = (self.current_price_call - self.entry_price_call) * self.qty * self.lot_size
        put_pnl = (self.entry_price_put - self.current_price_put) * self.qty * self.lot_size
        if "short_call" in self.action_type:
            call_pnl = -call_pnl
            put_pnl = -put_pnl
        return call_pnl + put_pnl

    def time_in_position(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds()

_positions: Dict[str, Position] = {}
_realized_pnl: float = 0.0
_daily_limit: float = 50000.0
_trade_count: int = 0
_settings = None

class EntryCheckRequest(BaseModel):
    underlying: str
    strike: float
    qty: int = 1
    action_type: str = "enter_long_call_short_put"

class ExitEstimateRequest(BaseModel):
    position_id: str

class AddPositionRequest(BaseModel):
    position_id: str
    underlying: str
    strike: float
    qty: int
    action_type: str
    entry_price_call: float
    entry_price_put: float
    entry_deviation_pct: float
    lot_size: int = 50

class UpdatePositionRequest(BaseModel):
    position_id: str
    current_price_call: float
    current_price_put: float
    current_deviation_pct: float

class ClosePositionRequest(BaseModel):
    position_id: str
    exit_pnl: float

class ResetRequest(BaseModel):
    daily_limit: float = 50000.0

@app.on_event("startup")
async def startup():
    global _settings, _daily_limit
    _settings = get_settings()
    _daily_limit = _settings.risk.max_daily_loss

@app.post("/tools/get_position_state")
async def get_position_state():
    positions = []
    for pid, pos in _positions.items():
        from data.processors.cost_calculator import TransactionCostCalculator
        tcc = TransactionCostCalculator()
        exit_cost = tcc.calculate_exit_costs(pos.underlying, pos.current_price_call, pos.qty, pos.lot_size)
        positions.append({
            "position_id": pid, "underlying": pos.underlying,
            "strike": pos.strike, "qty": pos.qty, "action_type": pos.action_type,
            "entry_deviation_pct": round(pos.entry_deviation_pct, 4),
            "current_deviation_pct": round(pos.current_deviation_pct, 4),
            "unrealized_pnl": round(pos.unrealized_pnl(), 2),
            "time_in_position": round(pos.time_in_position(), 1),
            "estimated_exit_cost": exit_cost.total,
            "entry_time": pos.entry_time.isoformat()})
    return {"positions": positions, "count": len(positions),
            "total_unrealized": round(sum(p.unrealized_pnl() for p in _positions.values()), 2)}

@app.post("/tools/check_entry_allowed")
async def check_entry_allowed(req: EntryCheckRequest):
    settings = get_settings()
    reasons = []
    allowed = True
    if len(_positions) >= settings.risk.max_positions:
        allowed = False
        reasons.append(f"Max positions ({settings.risk.max_positions}) reached")
    total_pnl = _realized_pnl + sum(p.unrealized_pnl() for p in _positions.values())
    pct_used = abs(min(total_pnl, 0)) / _daily_limit * 100 if _daily_limit > 0 else 0
    if total_pnl < -_daily_limit:
        allowed = False
        reasons.append(f"Daily loss limit (₹{_daily_limit:,.0f}) breached")
    now = datetime.now().time()
    inst = settings.instruments.get(req.underlying)
    if inst and not inst.can_open_position(now):
        allowed = False
        reasons.append(f"No new positions after {inst.no_new_positions_after}")
    margin = settings.risk.max_capital_per_trade
    if inst:
        margin = inst.margin_pct / 100.0 * req.strike * req.qty * inst.lot_size
    return {
        "allowed": allowed,
        "reason": "; ".join(reasons) if reasons else "Entry allowed",
        "remaining_capacity": settings.risk.max_positions - len(_positions),
        "daily_pnl_used_pct": round(pct_used, 1),
        "estimated_margin_required": round(margin, 2)}

@app.post("/tools/get_daily_pnl")
async def get_daily_pnl():
    unrealized = sum(p.unrealized_pnl() for p in _positions.values())
    total = _realized_pnl + unrealized
    return {
        "realized_pnl": round(_realized_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(total, 2),
        "daily_limit": _daily_limit,
        "pct_used": round(abs(min(total, 0)) / _daily_limit * 100 if _daily_limit > 0 else 0, 1),
        "sessions_remaining": 1, "trade_count": _trade_count}

@app.post("/tools/estimate_exit_pnl")
async def estimate_exit_pnl(req: ExitEstimateRequest):
    pos = _positions.get(req.position_id)
    if pos is None:
        return {"error": f"Position {req.position_id} not found"}
    from data.processors.cost_calculator import TransactionCostCalculator
    tcc = TransactionCostCalculator()
    gross = pos.unrealized_pnl()
    exit_costs = tcc.calculate_exit_costs(pos.underlying, pos.current_price_call, pos.qty, pos.lot_size)
    net = gross - exit_costs.total
    recommend = net > 0 or pos.time_in_position() > 240
    reason = "Profitable exit" if net > 0 else ("Time limit approaching" if pos.time_in_position() > 240 else "Exit would be unprofitable")
    return {
        "position_id": req.position_id, "gross_pnl": round(gross, 2),
        "total_costs_breakdown": exit_costs.to_dict(),
        "net_pnl": round(net, 2), "is_profitable": net > 0,
        "recommend_exit": recommend, "reason": reason}

@app.post("/tools/get_risk_limits")
async def get_risk_limits():
    settings = get_settings()
    return {
        "max_positions": settings.risk.max_positions,
        "max_capital_per_trade": settings.risk.max_capital_per_trade,
        "max_daily_loss": settings.risk.max_daily_loss,
        "max_holding_seconds": settings.risk.max_holding_seconds,
        "no_new_positions_after": "14:45",
        "force_close_by": "15:20",
        "current_positions": len(_positions),
        "current_daily_pnl": round(_realized_pnl + sum(p.unrealized_pnl() for p in _positions.values()), 2)}

@app.post("/internal/add_position")
async def add_position(req: AddPositionRequest):
    global _trade_count
    _positions[req.position_id] = Position(
        position_id=req.position_id, underlying=req.underlying,
        strike=req.strike, qty=req.qty, action_type=req.action_type,
        entry_price_call=req.entry_price_call, entry_price_put=req.entry_price_put,
        entry_deviation_pct=req.entry_deviation_pct, entry_time=datetime.now(),
        lot_size=req.lot_size)
    _trade_count += 1
    return {"status": "ok", "position_id": req.position_id}

@app.post("/internal/update_position")
async def update_position(req: UpdatePositionRequest):
    pos = _positions.get(req.position_id)
    if pos:
        pos.current_price_call = req.current_price_call
        pos.current_price_put = req.current_price_put
        pos.current_deviation_pct = req.current_deviation_pct
    return {"status": "ok"}

@app.post("/internal/close_position")
async def close_position(req: ClosePositionRequest):
    global _realized_pnl
    if req.position_id in _positions:
        del _positions[req.position_id]
    _realized_pnl += req.exit_pnl
    return {"status": "ok", "realized_pnl": round(_realized_pnl, 2)}

@app.post("/internal/reset")
async def reset(req: ResetRequest):
    global _positions, _realized_pnl, _trade_count, _daily_limit
    _positions.clear()
    _realized_pnl = 0.0
    _trade_count = 0
    _daily_limit = req.daily_limit
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "server": "risk", "port": 8002,
            "positions": len(_positions), "realized_pnl": round(_realized_pnl, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
