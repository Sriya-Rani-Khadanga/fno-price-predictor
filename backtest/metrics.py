"""
Backtest metrics — computes performance statistics.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

def compute_metrics(session_pnls: List[float], equity_curve: List[float]) -> Dict:
    """Compute comprehensive backtest metrics."""
    pnls = np.array(session_pnls)
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    std_pnl = pnls.std()
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    win_rate = wins / max(len(pnls), 1)
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls <= 0].mean() if losses > 0 else 0
    profit_factor = abs(pnls[pnls > 0].sum() / pnls[pnls <= 0].sum()) if losses > 0 and pnls[pnls <= 0].sum() != 0 else float('inf')
    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = 0
    peak = equity_curve[0]
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)
    calmar = (total_pnl / equity_curve[0]) / max_dd if max_dd > 0 else 0
    return {
        "total_pnl": round(total_pnl, 2), "avg_session_pnl": round(avg_pnl, 2),
        "std_session_pnl": round(std_pnl, 2), "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3), "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2), "calmar_ratio": round(calmar, 3),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "sessions": len(pnls), "winning": int(wins), "losing": int(losses)}
