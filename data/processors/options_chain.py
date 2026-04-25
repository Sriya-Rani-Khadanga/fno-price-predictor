"""
Options chain data structures and processing.
Defines the core dataclasses used throughout the system for representing
option chains, individual strikes, and PCP deviation data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class StrikeData:
    """Data for a single option strike in the chain."""
    strike: float
    call_bid: float
    call_ask: float
    call_ltp: float
    call_oi: int
    call_volume: int
    call_iv: float
    put_bid: float
    put_ask: float
    put_ltp: float
    put_oi: int
    put_volume: int
    put_iv: float
    theoretical_call: float = 0.0
    theoretical_put: float = 0.0
    pcp_deviation_pct: float = 0.0
    pcp_deviation_rupees: float = 0.0

    @property
    def call_mid(self) -> float:
        """Mid-price for call."""
        return (self.call_bid + self.call_ask) / 2.0 if self.call_ask > 0 else self.call_ltp

    @property
    def put_mid(self) -> float:
        """Mid-price for put."""
        return (self.put_bid + self.put_ask) / 2.0 if self.put_ask > 0 else self.put_ltp

    @property
    def call_spread(self) -> float:
        """Bid-ask spread for call."""
        return self.call_ask - self.call_bid if self.call_ask > 0 else 0.0

    @property
    def put_spread(self) -> float:
        """Bid-ask spread for put."""
        return self.put_ask - self.put_bid if self.put_ask > 0 else 0.0

    @property
    def total_oi(self) -> int:
        """Combined open interest for call + put at this strike."""
        return self.call_oi + self.put_oi

    def is_liquid(self, min_oi: int = 100) -> bool:
        """Check if strike has enough liquidity for trading."""
        return self.call_oi >= min_oi and self.put_oi >= min_oi


@dataclass
class OptionChain:
    """Full option chain for an underlying at a specific expiry."""
    underlying: str
    expiry: str
    spot_price: float
    spot_bid: float
    spot_ask: float
    timestamp: datetime
    strikes: List[StrikeData]
    data_source: str = "mock"  # "mock", "historical", "live"
    risk_free_rate: float = 0.065  # 6.5% RBI repo rate

    @property
    def last_update_seconds_ago(self) -> float:
        """Seconds since last update."""
        return (datetime.now() - self.timestamp).total_seconds()

    @property
    def is_stale(self) -> bool:
        """Check if data is stale (>5s old)."""
        return self.last_update_seconds_ago > 5.0

    @property
    def atm_strike(self) -> float:
        """Get the at-the-money strike."""
        if not self.strikes:
            return self.spot_price
        return min(self.strikes, key=lambda s: abs(s.strike - self.spot_price)).strike

    @property
    def atm_iv(self) -> float:
        """Get ATM implied volatility (average of call and put ATM IV)."""
        atm = self.atm_strike
        for s in self.strikes:
            if s.strike == atm:
                return (s.call_iv + s.put_iv) / 2.0
        return 0.2  # fallback

    @property
    def put_call_ratio(self) -> float:
        """Aggregate put/call OI ratio across all strikes."""
        total_put_oi = sum(s.put_oi for s in self.strikes)
        total_call_oi = sum(s.call_oi for s in self.strikes)
        if total_call_oi == 0:
            return 1.0
        return total_put_oi / total_call_oi

    def get_strike(self, strike_price: float) -> Optional[StrikeData]:
        """Get StrikeData for a specific strike price."""
        for s in self.strikes:
            if abs(s.strike - strike_price) < 0.01:
                return s
        return None

    def near_money_strikes(self, n: int = 5) -> List[StrikeData]:
        """Get n strikes closest to spot on each side."""
        sorted_strikes = sorted(self.strikes, key=lambda s: abs(s.strike - self.spot_price))
        return sorted_strikes[:n * 2]

    def to_dict(self) -> Dict:
        """Serialize to dictionary for MCP server responses."""
        return {
            "underlying": self.underlying,
            "expiry": self.expiry,
            "spot_price": self.spot_price,
            "spot_bid": self.spot_bid,
            "spot_ask": self.spot_ask,
            "timestamp": self.timestamp.isoformat(),
            "data_source": self.data_source,
            "is_stale": self.is_stale,
            "staleness_seconds": round(self.last_update_seconds_ago, 1),
            "atm_strike": self.atm_strike,
            "atm_iv": round(self.atm_iv, 4),
            "put_call_ratio": round(self.put_call_ratio, 3),
            "risk_free_rate": self.risk_free_rate,
            "strikes": [
                {
                    "strike": s.strike,
                    "call_bid": s.call_bid,
                    "call_ask": s.call_ask,
                    "call_ltp": s.call_ltp,
                    "call_oi": s.call_oi,
                    "call_volume": s.call_volume,
                    "call_iv": round(s.call_iv, 4),
                    "put_bid": s.put_bid,
                    "put_ask": s.put_ask,
                    "put_ltp": s.put_ltp,
                    "put_oi": s.put_oi,
                    "put_volume": s.put_volume,
                    "put_iv": round(s.put_iv, 4),
                    "theoretical_call": round(s.theoretical_call, 2),
                    "theoretical_put": round(s.theoretical_put, 2),
                    "pcp_deviation_pct": round(s.pcp_deviation_pct, 4),
                    "pcp_deviation_rupees": round(s.pcp_deviation_rupees, 2),
                }
                for s in self.strikes
            ],
        }


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Theoretical call option price.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Theoretical put option price.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_call(S: float, K: float, T: float, r: float, market_price: float,
                     tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Newton-Raphson implied volatility for a call option.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        market_price: Observed market price
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        Implied volatility or 0.01 if convergence fails.
    """
    if T <= 0 or market_price <= 0:
        return 0.2
    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        price = black_scholes_call(S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break
        sigma = sigma - (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.01
        if abs(price - market_price) < tol:
            break
    return max(sigma, 0.01)


def implied_vol_put(S: float, K: float, T: float, r: float, market_price: float,
                    tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Newton-Raphson implied volatility for a put option.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        market_price: Observed market price
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        Implied volatility or 0.01 if convergence fails.
    """
    if T <= 0 or market_price <= 0:
        return 0.2
    sigma = 0.25
    for _ in range(max_iter):
        price = black_scholes_put(S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break
        sigma = sigma - (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.01
        if abs(price - market_price) < tol:
            break
    return max(sigma, 0.01)


def compute_greeks(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call") -> Dict[str, float]:
    """
    Compute Black-Scholes Greeks for an option.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility
        option_type: "call" or "put"
    
    Returns:
        Dictionary with delta, gamma, theta, vega, rho.
    """
    if T <= 0 or sigma <= 0:
        delta = 1.0 if option_type == "call" and S > K else (
            -1.0 if option_type == "put" and S < K else 0.0)
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta / 365.0, 4),  # daily theta
        "vega": round(vega, 4),
        "rho": round(rho, 4),
    }
