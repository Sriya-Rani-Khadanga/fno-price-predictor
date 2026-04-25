"""
Microstructure noise and realistic market simulation utilities.
Generates GBM paths, bid-ask spreads, volume profiles, and OI distributions.
"""
from __future__ import annotations
import math
import random
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple
import numpy as np

class MicrostructureGenerator:
    """Generates realistic market microstructure for synthetic data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._volume_profile = self._build_intraday_volume_profile()

    def _build_intraday_volume_profile(self) -> Dict[int, float]:
        """U-shaped intraday volume profile (high at open/close, low midday)."""
        profile = {}
        for minute in range(375):
            t = minute / 375.0
            volume = 1.0 + 2.0 * math.exp(-((t - 0.0) ** 2) / 0.01) + 1.5 * math.exp(-((t - 1.0) ** 2) / 0.01) + 0.3 * math.exp(-((t - 0.5) ** 2) / 0.05)
            profile[minute] = volume
        total = sum(profile.values())
        return {k: v / total for k, v in profile.items()}

    def generate_gbm_path(self, s0: float, mu: float, sigma: float,
                           dt: float, n_steps: int) -> np.ndarray:
        """Generate Geometric Brownian Motion price path."""
        prices = np.zeros(n_steps + 1)
        prices[0] = s0
        z = self.rng.standard_normal(n_steps)
        for i in range(n_steps):
            prices[i + 1] = prices[i] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z[i])
        return prices

    def generate_bid_ask(self, mid: float, base_spread_pct: float = 0.05,
                          oi: int = 1000) -> Tuple[float, float]:
        """Generate bid-ask prices around a mid price with OI-dependent spread."""
        liq_factor = min(1.0, oi / 5000.0)
        spread = mid * (base_spread_pct / 100.0) * (2.0 - liq_factor)
        noise = self.rng.uniform(-0.1, 0.1) * spread
        half = (spread + noise) / 2.0
        return round(max(0.05, mid - half), 2), round(mid + half, 2)

    def generate_oi_distribution(self, spot: float, n_strikes: int,
                                  strike_interval: float = 100.0) -> Dict[float, int]:
        """Generate realistic OI distribution centered on ATM."""
        oi_dist = {}
        atm = round(spot / strike_interval) * strike_interval
        for i in range(-n_strikes, n_strikes + 1):
            strike = atm + i * strike_interval
            distance = abs(i)
            base_oi = int(10000 * math.exp(-0.3 * distance))
            noise = self.rng.randint(-base_oi // 5, base_oi // 5 + 1)
            oi_dist[strike] = max(10, base_oi + noise)
        return oi_dist

    def generate_iv_surface(self, spot: float, strikes: List[float],
                             atm_iv: float = 0.15) -> Dict[float, Tuple[float, float]]:
        """Generate IV smile/skew surface. Returns {strike: (call_iv, put_iv)}."""
        surface = {}
        for k in strikes:
            moneyness = math.log(k / spot) if spot > 0 else 0
            skew = 0.02 * moneyness ** 2 + 0.01 * moneyness
            call_iv = atm_iv + skew + self.rng.uniform(-0.005, 0.005)
            put_iv = atm_iv + skew + 0.005 + self.rng.uniform(-0.005, 0.005)
            surface[k] = (max(0.05, call_iv), max(0.05, put_iv))
        return surface

    def get_volume_weight(self, session_minute: int) -> float:
        """Get volume weight for a given minute in the session."""
        return self._volume_profile.get(min(session_minute, 374), 0.002)

    def add_microstructure_noise(self, price: float, volatility: float = 0.001) -> float:
        """Add tick-level noise to a price."""
        noise = self.rng.normal(0, price * volatility)
        return round(max(0.05, price + noise), 2)

    def generate_intraday_from_ohlc(self, open_p: float, high: float, low: float,
                                      close: float, n_ticks: int) -> np.ndarray:
        """Reconstruct intraday path from OHLC data using bridge process."""
        prices = np.zeros(n_ticks)
        prices[0] = open_p
        prices[-1] = close
        mid_point = n_ticks // 2
        range_size = high - low
        if range_size < 0.01:
            return np.linspace(open_p, close, n_ticks)
        bridge = np.zeros(n_ticks)
        bridge[0] = open_p
        bridge[-1] = close
        for i in range(1, n_ticks - 1):
            t = i / (n_ticks - 1)
            target = open_p + (close - open_p) * t
            noise = self.rng.normal(0, range_size * 0.1)
            bridge[i] = target + noise
        scale = range_size / (max(bridge) - min(bridge)) if max(bridge) != min(bridge) else 1.0
        mean_b = np.mean(bridge)
        prices = (bridge - mean_b) * scale + (open_p + close) / 2
        prices[0] = open_p
        prices[-1] = close
        prices = np.clip(prices, low, high)
        return prices
