"""
MCP Technical Indicators Server — FastAPI app on port 8004.
Provides RSI, EMA, MACD, and Greeks (Delta, Gamma) for the trading agent.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="PCP Arb Technical MCP Server", version="1.0.0")

# Internal state for price history
_price_history: Dict[str, List[float]] = {}
_MAX_HISTORY = 100

class TechRequest(BaseModel):
    symbol: str
    period: int = 14

class GreeksRequest(BaseModel):
    symbol: str
    strike: float
    expiry_days: float = 30
    iv: float = 0.15
    rate: float = 0.065

class PriceUpdate(BaseModel):
    symbol: str
    price: float

@app.post("/tools/get_rsi")
async def get_rsi(req: TechRequest):
    history = _price_history.get(req.symbol, [])
    if len(history) < req.period + 1:
        return {"error": f"Insufficient history for RSI on {req.symbol}", "needed": req.period + 1, "have": len(history)}
    
    prices = np.array(history[-(req.period+1):])
    deltas = np.diff(prices)
    seed = deltas[:req.period]
    up = seed[seed >= 0].sum() / req.period
    down = -seed[seed < 0].sum() / req.period
    rs = up / down if down != 0 else 100
    rsi = 100. - 100. / (1. + rs)
    
    return {"symbol": req.symbol, "rsi": round(rsi, 2), "period": req.period, "status": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"}

@app.post("/tools/get_ema")
async def get_ema(req: TechRequest):
    history = _price_history.get(req.symbol, [])
    if len(history) < req.period:
        return {"error": f"Insufficient history for EMA on {req.symbol}"}
    
    prices = history[-req.period:]
    ema = pd.Series(prices).ewm(span=req.period, adjust=False).mean().iloc[-1]
    return {"symbol": req.symbol, "ema": round(float(ema), 2), "period": req.period}

@app.post("/tools/get_greeks")
async def get_greeks(req: GreeksRequest):
    from scipy.stats import norm
    
    history = _price_history.get(req.symbol, [])
    S = history[-1] if history else 22000.0 # Default if no history
    K = req.strike
    T = max(0.001, req.expiry_days / 365.0)
    r = req.rate
    sigma = req.iv
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    
    return {
        "symbol": req.symbol, "strike": K,
        "delta_call": round(float(delta_call), 4),
        "delta_put": round(float(delta_put), 4),
        "gamma": round(float(gamma), 6),
        "theta_call_daily": round(float(theta_call) / 365.0, 2),
        "spot_used": round(S, 2)
    }

@app.post("/feed/update")
async def feed_update(data: PriceUpdate):
    if data.symbol not in _price_history:
        _price_history[data.symbol] = []
    _price_history[data.symbol].append(data.price)
    if len(_price_history[data.symbol]) > _MAX_HISTORY:
        _price_history[data.symbol] = _price_history[data.symbol][-_MAX_HISTORY:]
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "server": "technical", "port": 8004}

if __name__ == "__main__":
    import pandas as pd # Needed for EMA
    uvicorn.run(app, host="0.0.0.0", port=8004)
