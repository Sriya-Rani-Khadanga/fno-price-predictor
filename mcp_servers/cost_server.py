"""
MCP Cost Server — FastAPI app on port 8003.
Provides transaction cost calculation tool endpoints.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from data.processors.cost_calculator import TransactionCostCalculator
from config.settings import get_settings

app = FastAPI(title="PCP Arb Cost MCP Server", version="1.0.0")
_tcc: TransactionCostCalculator = None
_cost_history: List[Dict] = []
_settings = None

class ArbCostsRequest(BaseModel):
    underlying: str
    strike: float
    expiry_days: int = 15
    qty: int = 1
    gross_violation_pct: float = 0.5

class BreakevenRequest(BaseModel):
    underlying: str
    strike: float
    expiry_days: int = 15
    qty: int = 1

class STTTrapRequest(BaseModel):
    underlying: str
    strike: float
    expiry_days: int = 15
    qty: int = 1
    hold_to_expiry: bool = False

@app.on_event("startup")
async def startup():
    global _tcc, _settings
    _settings = get_settings()
    _tcc = TransactionCostCalculator()

@app.post("/tools/calculate_arb_costs")
async def calculate_arb_costs(req: ArbCostsRequest):
    inst = _settings.instruments.get(req.underlying)
    spot = 22000.0
    if inst:
        spot = _settings.feed.initial_spots.get(req.underlying, 22000.0)
    result = _tcc.calculate_full_arb_costs(
        req.underlying, req.strike, spot, req.expiry_days, req.qty, req.gross_violation_pct)
    record = {"timestamp": datetime.now().isoformat(), "underlying": req.underlying,
              "strike": req.strike, "gross_pct": req.gross_violation_pct,
              "net_profit": result.net_profit_per_lot, "total_cost": result.costs.total,
              "is_profitable": result.is_profitable}
    _cost_history.append(record)
    if len(_cost_history) > 500:
        _cost_history[:] = _cost_history[-250:]
    return result.to_dict()

@app.post("/tools/get_breakeven_violation")
async def get_breakeven_violation(req: BreakevenRequest):
    inst = _settings.instruments.get(req.underlying)
    spot = _settings.feed.initial_spots.get(req.underlying, 22000.0)
    result = _tcc.get_breakeven_violation(req.underlying, req.strike, spot, req.expiry_days, req.qty, 0.5)
    return result

@app.post("/tools/simulate_stt_trap")
async def simulate_stt_trap(req: STTTrapRequest):
    spot = _settings.feed.initial_spots.get(req.underlying, 22000.0)
    result = _tcc.simulate_stt_trap(req.underlying, req.strike, spot, req.expiry_days, req.qty, req.hold_to_expiry)
    return result

@app.post("/tools/get_cost_history")
async def get_cost_history():
    if not _cost_history:
        return {"trades": 0, "avg_cost": 0, "cost_trend": "stable", "total_cost_drag": 0}
    recent = _cost_history[-50:]
    avg_cost = sum(r["total_cost"] for r in recent) / len(recent)
    total_drag = sum(r["total_cost"] for r in _cost_history)
    if len(recent) >= 10:
        first_half = sum(r["total_cost"] for r in recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(r["total_cost"] for r in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        trend = "increasing" if second_half > first_half * 1.1 else ("decreasing" if second_half < first_half * 0.9 else "stable")
    else:
        trend = "stable"
    return {
        "trades": len(_cost_history), "avg_cost": round(avg_cost, 2),
        "cost_trend": trend, "total_cost_drag": round(total_drag, 2),
        "profitable_pct": round(sum(1 for r in _cost_history if r["is_profitable"]) / max(len(_cost_history), 1) * 100, 1),
        "recent_costs": recent[-5:]}

@app.post("/internal/update_spot")
async def update_spot(data: Dict):
    """Internal endpoint to receive spot price updates for cost calculations."""
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "server": "cost", "port": 8003, "trades_tracked": len(_cost_history)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
