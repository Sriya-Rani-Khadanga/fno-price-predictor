"""
Feature engineering for PCP arbitrage signals.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from data.processors.options_chain import OptionChain

def extract_features(chain: OptionChain, spot: float, T: float) -> Dict[str, float]:
    """Extract numerical features from an option chain for signal generation."""
    atm_iv = chain.atm_iv
    pcr = chain.put_call_ratio
    n_strikes = len(chain.strikes)
    deviations = [s.pcp_deviation_pct for s in chain.strikes]
    max_dev = max(deviations) if deviations else 0.0
    mean_dev = np.mean(deviations) if deviations else 0.0
    std_dev = np.std(deviations) if len(deviations) > 1 else 0.0
    total_call_oi = sum(s.call_oi for s in chain.strikes)
    total_put_oi = sum(s.put_oi for s in chain.strikes)
    avg_call_spread = np.mean([s.call_spread for s in chain.strikes if s.call_ask > 0]) if chain.strikes else 0.0
    avg_put_spread = np.mean([s.put_spread for s in chain.strikes if s.put_ask > 0]) if chain.strikes else 0.0
    near_money = chain.near_money_strikes(3)
    nm_avg_dev = np.mean([s.pcp_deviation_pct for s in near_money]) if near_money else 0.0
    return {
        "spot": spot, "atm_iv": atm_iv, "pcr": pcr, "T": T,
        "max_deviation": max_dev, "mean_deviation": mean_dev, "std_deviation": std_dev,
        "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
        "avg_call_spread": avg_call_spread, "avg_put_spread": avg_put_spread,
        "near_money_avg_dev": nm_avg_dev, "n_strikes": n_strikes,
        "staleness": chain.last_update_seconds_ago}
