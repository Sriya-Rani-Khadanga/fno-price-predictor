"""
Feature engineering for the Gymnasium RL environment.

Extracts 45+ numerical features from an option chain snapshot
into a flat numpy array suitable for Stable-Baselines3 policies.

Feature groups:
  1. Spot / Underlying (5)
  2. ATM Greeks (5)
  3. PCP Deviation stats (6)
  4. Microstructure (6)
  5. IV Surface (5)
  6. Technical indicators (6)
  7. Temporal (4)
  8. Portfolio state (5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try importing from the project's own modules when available
try:
    from data.processors.options_chain import (
        OptionChain, StrikeData, compute_greeks,
        black_scholes_call, black_scholes_put,
    )
except ImportError:
    OptionChain = None


# ── Feature dimension constant ────────────────────────────────────────────────
FEATURE_DIM = 42


@dataclass
class PortfolioState:
    """Current portfolio state passed to feature engineer."""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    n_positions: int = 0
    max_positions: int = 5
    capital_used_pct: float = 0.0
    avg_entry_deviation: float = 0.0
    holding_seconds: float = 0.0


class FeatureEngineer:
    """
    Converts raw OptionChain snapshots into a fixed-size feature vector
    for the Gymnasium observation space.
    """

    def __init__(self, history_len: int = 20):
        """
        Args:
            history_len: number of past spot values to keep for technical indicators.
        """
        self.history_len = history_len
        self.spot_history: List[float] = []
        self.iv_history: List[float] = []
        self.volume_history: List[int] = []
        self.oi_history: List[int] = []

    def reset(self):
        """Clear all history buffers."""
        self.spot_history.clear()
        self.iv_history.clear()
        self.volume_history.clear()
        self.oi_history.clear()

    def extract(
        self,
        chain: OptionChain,
        T: float,
        minutes_to_close: int,
        days_to_expiry: int,
        portfolio: PortfolioState,
    ) -> np.ndarray:
        """
        Extract full feature vector from current market state.

        Returns:
            np.ndarray of shape (FEATURE_DIM,) with float32 values.
        """
        spot = chain.spot_price
        r = chain.risk_free_rate
        atm_strike = chain.atm_strike

        # Update history
        self.spot_history.append(spot)
        if len(self.spot_history) > self.history_len:
            self.spot_history = self.spot_history[-self.history_len:]

        atm_iv = chain.atm_iv
        self.iv_history.append(atm_iv)
        if len(self.iv_history) > self.history_len:
            self.iv_history = self.iv_history[-self.history_len:]

        total_vol = sum(s.call_volume + s.put_volume for s in chain.strikes)
        self.volume_history.append(total_vol)
        if len(self.volume_history) > self.history_len:
            self.volume_history = self.volume_history[-self.history_len:]

        total_oi = sum(s.call_oi + s.put_oi for s in chain.strikes)
        self.oi_history.append(total_oi)
        if len(self.oi_history) > self.history_len:
            self.oi_history = self.oi_history[-self.history_len:]

        features: List[float] = []

        # ── 1. Spot / Underlying (5 features) ─────────────────────────────
        features.append(spot / 1000.0)  # normalized spot
        spot_ret_1 = ((spot / self.spot_history[-2]) - 1) * 100 if len(self.spot_history) >= 2 else 0.0
        features.append(spot_ret_1)
        spot_ret_5 = ((spot / self.spot_history[-6]) - 1) * 100 if len(self.spot_history) >= 6 else 0.0
        features.append(spot_ret_5)
        spread_pct = ((chain.spot_ask - chain.spot_bid) / spot * 100) if spot > 0 else 0.0
        features.append(spread_pct)
        moneyness_atm = (spot - atm_strike) / spot * 100 if spot > 0 else 0.0
        features.append(moneyness_atm)

        # ── 2. ATM Greeks (5 features) ────────────────────────────────────
        atm_data = chain.get_strike(atm_strike)
        if atm_data is not None:
            greeks = compute_greeks(spot, atm_strike, T, r, atm_data.call_iv, "call")
            features.append(greeks["delta"])
            features.append(greeks["gamma"] * 1000)  # scale up
            features.append(greeks["theta"] * 100)    # scale up
            features.append(greeks["vega"])
            # IV skew: put IV - call IV at ATM
            features.append(atm_data.put_iv - atm_data.call_iv)
        else:
            features.extend([0.5, 0.0, 0.0, 0.0, 0.0])

        # ── 3. PCP Deviation stats (6 features) ──────────────────────────
        deviations = [s.pcp_deviation_pct for s in chain.strikes]
        if deviations:
            features.append(max(deviations))
            features.append(min(deviations))
            features.append(float(np.mean(deviations)))
            features.append(float(np.std(deviations)))
            # Count of strikes with |deviation| > 0.3%
            sig_count = sum(1 for d in deviations if abs(d) > 0.3)
            features.append(sig_count / max(len(deviations), 1))
            # Best deviation (most positive for long arb)
            best_idx = int(np.argmax(np.abs(deviations)))
            features.append(chain.strikes[best_idx].strike / 1000.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, spot / 1000.0])

        # ── 4. Microstructure (6 features) ────────────────────────────────
        near_strikes = chain.near_money_strikes(3)
        if near_strikes:
            avg_call_spread = float(np.mean([s.call_spread for s in near_strikes]))
            avg_put_spread = float(np.mean([s.put_spread for s in near_strikes]))
            avg_oi = float(np.mean([s.total_oi for s in near_strikes]))
            features.append(avg_call_spread)
            features.append(avg_put_spread)
            features.append(avg_oi / 10000.0)  # normalize
            # Bid-ask tightness (inverse spread as liquidity signal)
            tightness = 1.0 / (1.0 + avg_call_spread + avg_put_spread)
            features.append(tightness)
            # Call/Put volume ratio near money
            cv = sum(s.call_volume for s in near_strikes)
            pv = sum(s.put_volume for s in near_strikes)
            features.append(cv / max(pv, 1))
            # OI change direction
            oi_change = sum(s.call_oi + s.put_oi for s in near_strikes)
            features.append(oi_change / 100000.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.5, 1.0, 0.0])

        # ── 5. IV Surface (5 features) ────────────────────────────────────
        features.append(atm_iv)
        # IV change
        iv_change = atm_iv - self.iv_history[-2] if len(self.iv_history) >= 2 else 0.0
        features.append(iv_change)
        # IV skew: OTM put IV vs OTM call IV
        otm_calls = [s for s in chain.strikes if s.strike > spot]
        otm_puts = [s for s in chain.strikes if s.strike < spot]
        avg_otm_call_iv = float(np.mean([s.call_iv for s in otm_calls])) if otm_calls else atm_iv
        avg_otm_put_iv = float(np.mean([s.put_iv for s in otm_puts])) if otm_puts else atm_iv
        features.append(avg_otm_put_iv - avg_otm_call_iv)  # skew
        # Term structure proxy (near vs far OTM)
        if len(otm_calls) >= 2:
            features.append(otm_calls[-1].call_iv - otm_calls[0].call_iv)
        else:
            features.append(0.0)
        # Put-call ratio (OI based)
        features.append(chain.put_call_ratio)

        # ── 6. Technical Indicators (6 features) ──────────────────────────
        # SMA5, SMA20
        sma5 = float(np.mean(self.spot_history[-5:])) if len(self.spot_history) >= 5 else spot
        sma20 = float(np.mean(self.spot_history[-20:])) if len(self.spot_history) >= 20 else spot
        features.append((spot - sma5) / spot * 100)  # distance from SMA5
        features.append((spot - sma20) / spot * 100)  # distance from SMA20

        # RSI(14)
        features.append(self._compute_rsi(self.spot_history, 14))

        # Spot volatility (std of recent returns)
        if len(self.spot_history) >= 5:
            returns = np.diff(self.spot_history[-6:]) / np.array(self.spot_history[-6:-1])
            features.append(float(np.std(returns)) * 100)
        else:
            features.append(0.0)

        # Volume trend
        if len(self.volume_history) >= 5:
            vol_sma = float(np.mean(self.volume_history[-5:]))
            features.append(total_vol / max(vol_sma, 1))
        else:
            features.append(1.0)

        # OI trend
        if len(self.oi_history) >= 5:
            oi_sma = float(np.mean(self.oi_history[-5:]))
            features.append(total_oi / max(oi_sma, 1))
        else:
            features.append(1.0)

        # ── 7. Temporal (4 features) ──────────────────────────────────────
        features.append(minutes_to_close / 375.0)  # session progress
        features.append(days_to_expiry / 30.0)      # time to expiry
        features.append(T)                            # T in years
        # Session phase: 0=open, 0.5=mid, 1=close
        features.append(1.0 - minutes_to_close / 375.0)

        # ── 8. Portfolio State (5 features) ───────────────────────────────
        features.append(portfolio.unrealized_pnl / 10000.0)
        features.append(portfolio.realized_pnl / 10000.0)
        features.append(portfolio.n_positions / max(portfolio.max_positions, 1))
        features.append(portfolio.capital_used_pct)
        features.append(portfolio.holding_seconds / 300.0)  # normalize by 5min

        assert len(features) == FEATURE_DIM, (
            f"Expected {FEATURE_DIM} features, got {len(features)}"
        )

        arr = np.array(features, dtype=np.float32)
        # Replace any NaN/inf with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        return arr

    @staticmethod
    def _compute_rsi(prices: List[float], period: int = 14) -> float:
        """Compute RSI from price history."""
        if len(prices) < period + 1:
            return 50.0  # neutral
        changes = np.diff(prices[-(period + 1):])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def feature_names() -> List[str]:
        """Return human-readable feature names for debugging."""
        return [
            # Spot (5)
            "spot_norm", "spot_ret_1", "spot_ret_5", "spot_spread_pct", "moneyness_atm",
            # Greeks (5)
            "atm_delta", "atm_gamma_1k", "atm_theta_100", "atm_vega", "atm_iv_skew",
            # PCP (6)
            "pcp_max", "pcp_min", "pcp_mean", "pcp_std", "pcp_sig_ratio", "pcp_best_strike",
            # Microstructure (6)
            "avg_call_spread", "avg_put_spread", "avg_oi_norm", "liquidity_tightness",
            "cv_pv_ratio", "oi_change_norm",
            # IV (5)
            "atm_iv", "iv_change", "iv_skew", "iv_term_structure", "pcr_oi",
            # Technical (6)
            "dist_sma5", "dist_sma20", "rsi_14", "spot_vol", "vol_trend", "oi_trend",
            # Temporal (4)
            "session_progress_inv", "dte_norm", "T_years", "session_phase",
            # Portfolio (5)
            "unreal_pnl_norm", "real_pnl_norm", "pos_utilization",
            "capital_used_pct", "holding_time_norm",
        ]
