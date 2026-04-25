"""
MCP Market Data Server — FastAPI app on port 8001.
Provides tool endpoints for the LLM agent to query market intelligence.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from data.processors.options_chain import OptionChain, StrikeData
from data.processors.pcp_calculator import PCPCalculator, PCPViolation
from config.settings import get_settings

app = FastAPI(title="PCP Arb Market Data MCP Server", version="1.0.0")

# Internal state
_chains: Dict[str, OptionChain] = {}
_violation_history: Dict[str, List[Dict]] = {}
_pcp_calc: Optional[PCPCalculator] = None
_settings = None

def _init():
    global _pcp_calc, _settings
    _settings = get_settings()
    lots = {sym: inst.lot_size for sym, inst in _settings.instruments.items()}
    _pcp_calc = PCPCalculator(lots)

class OptionChainRequest(BaseModel):
    underlying: str
    expiry: str = ""

class SpotRequest(BaseModel):
    symbol: str

class DeviationRequest(BaseModel):
    underlying: str
    strike: float
    expiry: str = ""

class IVSurfaceRequest(BaseModel):
    underlying: str
    expiry: str = ""

class HistViolationsRequest(BaseModel):
    underlying: str
    lookback_sessions: int = 10

class RegimeRequest(BaseModel):
    underlying: str

class FeedUpdate(BaseModel):
    underlying: str
    expiry: str
    spot_price: float
    spot_bid: float
    spot_ask: float
    timestamp: str
    data_source: str
    strikes: List[Dict]
    is_stale: Optional[bool] = False
    staleness_seconds: Optional[float] = 0.0
    atm_strike: Optional[float] = 0.0
    atm_iv: Optional[float] = 0.15
    put_call_ratio: Optional[float] = 1.0
    risk_free_rate: Optional[float] = 0.065

@app.on_event("startup")
async def startup():
    _init()

@app.post("/tools/get_option_chain")
async def get_option_chain(req: OptionChainRequest):
    chain = _chains.get(req.underlying)
    if chain is None:
        return {"error": f"No data for {req.underlying}", "available": list(_chains.keys())}
    T = 15.0 / 365.0
    if _pcp_calc:
        _pcp_calc.compute_all_deviations(chain, T)
    result = chain.to_dict()
    result["staleness_warning"] = chain.is_stale
    return result

@app.post("/tools/get_spot_price")
async def get_spot_price(req: SpotRequest):
    chain = _chains.get(req.symbol)
    if chain is None:
        return {"error": f"No data for {req.symbol}"}
    return {"symbol": req.symbol, "ltp": chain.spot_price, "bid": chain.spot_bid,
            "ask": chain.spot_ask, "timestamp": chain.timestamp.isoformat(),
            "source": chain.data_source}

@app.post("/tools/get_pcp_deviation")
async def get_pcp_deviation(req: DeviationRequest):
    chain = _chains.get(req.underlying)
    if chain is None:
        return {"error": f"No data for {req.underlying}"}
    sd = chain.get_strike(req.strike)
    if sd is None:
        nearest = min(chain.strikes, key=lambda s: abs(s.strike - req.strike))
        sd = nearest
    T = 15.0 / 365.0
    v = _pcp_calc.compute_deviation(chain, sd, T)
    result = v.to_dict()
    inst = _settings.instruments.get(req.underlying)
    lot = inst.lot_size if inst else 50
    result["deviation_rupees_per_lot"] = round(v.deviation_rupees * lot, 2)
    return result

@app.post("/tools/get_iv_surface")
async def get_iv_surface(req: IVSurfaceRequest):
    chain = _chains.get(req.underlying)
    if chain is None:
        return {"error": f"No data for {req.underlying}"}
    call_ivs = {s.strike: round(s.call_iv, 4) for s in chain.strikes}
    put_ivs = {s.strike: round(s.put_iv, 4) for s in chain.strikes}
    atm = chain.atm_strike
    atm_iv = chain.atm_iv
    skew_data = []
    for s in chain.strikes:
        moneyness = (s.strike - chain.spot_price) / chain.spot_price
        skew_data.append({"strike": s.strike, "moneyness": round(moneyness, 4),
                          "call_iv": round(s.call_iv, 4), "put_iv": round(s.put_iv, 4),
                          "iv_skew": round(s.put_iv - s.call_iv, 4)})
    return {"underlying": req.underlying, "expiry": chain.expiry,
            "atm_strike": atm, "atm_iv": round(atm_iv, 4),
            "call_ivs": call_ivs, "put_ivs": put_ivs,
            "skew": skew_data, "timestamp": chain.timestamp.isoformat()}

@app.post("/tools/get_historical_violations")
async def get_historical_violations(req: HistViolationsRequest):
    key = req.underlying
    history = _violation_history.get(key, [])
    if not history:
        return {"underlying": key, "sessions_analyzed": 0,
                "avg_frequency": 0, "avg_magnitude": 0, "avg_duration": 0,
                "common_strikes": [], "best_hour": 10, "worst_hour": 14,
                "message": "No historical violation data available. Run alpha analysis first."}
    magnitudes = [v.get("deviation_pct", 0) for v in history]
    durations = [v.get("active_seconds", 0) for v in history]
    strike_counts: Dict[float, int] = {}
    hour_counts: Dict[int, int] = {}
    for v in history:
        sk = v.get("strike", 0)
        strike_counts[sk] = strike_counts.get(sk, 0) + 1
        h = v.get("hour", 12)
        hour_counts[h] = hour_counts.get(h, 0) + 1
    top_strikes = sorted(strike_counts, key=strike_counts.get, reverse=True)[:5]
    return {
        "underlying": key, "sessions_analyzed": req.lookback_sessions,
        "total_violations": len(history),
        "avg_frequency": round(len(history) / max(req.lookback_sessions, 1), 1),
        "avg_magnitude": round(sum(magnitudes) / max(len(magnitudes), 1), 4),
        "avg_duration": round(sum(durations) / max(len(durations), 1), 1),
        "common_strikes": top_strikes,
        "best_hour": max(hour_counts, key=hour_counts.get) if hour_counts else 10,
        "worst_hour": min(hour_counts, key=hour_counts.get) if hour_counts else 14}

@app.post("/tools/get_market_regime")
async def get_market_regime(req: RegimeRequest):
    chain = _chains.get(req.underlying)
    if chain is None:
        return {"error": f"No data for {req.underlying}"}
    atm_iv = chain.atm_iv
    pcr = chain.put_call_ratio
    now = chain.timestamp
    if atm_iv > 0.25:
        regime = "volatile"
    elif atm_iv < 0.12:
        regime = "quiet"
    elif pcr > 1.3:
        regime = "trending"
    else:
        regime = "mean_reverting"
    return {
        "underlying": req.underlying,
        "session_time": now.strftime("%H:%M:%S"),
        "vix_proxy": round(atm_iv * 100, 2),
        "trend_direction": "neutral",
        "realized_vol_60s": round(atm_iv * 0.8, 4),
        "realized_vol_300s": round(atm_iv * 0.9, 4),
        "put_call_ratio": round(pcr, 3),
        "regime": regime}

@app.post("/feed/update")
async def feed_update(data: FeedUpdate):
    """Receive market data updates from the active feed."""
    strikes = []
    for sd in data.strikes:
        strikes.append(StrikeData(
            strike=sd["strike"], call_bid=sd.get("call_bid", 0), call_ask=sd.get("call_ask", 0),
            call_ltp=sd.get("call_ltp", 0), call_oi=sd.get("call_oi", 0),
            call_volume=sd.get("call_volume", 0), call_iv=sd.get("call_iv", 0.15),
            put_bid=sd.get("put_bid", 0), put_ask=sd.get("put_ask", 0),
            put_ltp=sd.get("put_ltp", 0), put_oi=sd.get("put_oi", 0),
            put_volume=sd.get("put_volume", 0), put_iv=sd.get("put_iv", 0.15),
            theoretical_call=sd.get("theoretical_call", 0),
            theoretical_put=sd.get("theoretical_put", 0),
            pcp_deviation_pct=sd.get("pcp_deviation_pct", 0),
            pcp_deviation_rupees=sd.get("pcp_deviation_rupees", 0)))
    chain = OptionChain(
        underlying=data.underlying, expiry=data.expiry,
        spot_price=data.spot_price, spot_bid=data.spot_bid, spot_ask=data.spot_ask,
        timestamp=datetime.fromisoformat(data.timestamp),
        strikes=strikes, data_source=data.data_source)
    _chains[data.underlying] = chain
    T = 15.0 / 365.0
    if _pcp_calc:
        violations = _pcp_calc.get_active_violations(chain, T, min_pct=0.1)
        key = data.underlying
        if key not in _violation_history:
            _violation_history[key] = []
        for v in violations:
            _violation_history[key].append({**v.to_dict(), "hour": chain.timestamp.hour})
            if len(_violation_history[key]) > 1000:
                _violation_history[key] = _violation_history[key][-500:]
    return {"status": "ok", "underlying": data.underlying, "strikes_count": len(strikes)}

@app.get("/health")
async def health():
    return {"status": "ok", "server": "market_data", "port": 8001,
            "instruments": list(_chains.keys()),
            "last_updates": {k: v.timestamp.isoformat() for k, v in _chains.items()}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
