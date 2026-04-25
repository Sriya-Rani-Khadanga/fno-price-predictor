"""
Synthetic data generator for training when historical data is insufficient.
Generates full sessions with realistic characteristics.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from data.processors.options_chain import OptionChain, StrikeData, black_scholes_call, black_scholes_put
from data.processors.microstructure import MicrostructureGenerator
from data.historical.store import HistoricalStore
from config.settings import get_settings

class SyntheticGenerator:
    """Generates synthetic historical sessions with configurable characteristics."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.micro = MicrostructureGenerator(seed)
        self.settings = get_settings()
        self.store = HistoricalStore()

    def generate_session(self, underlying: str, dt: date,
                          n_snapshots: int = 125, violation_rate: float = 0.1,
                          base_spot: float = None) -> List[OptionChain]:
        """Generate a complete synthetic session with n_snapshots chain snapshots."""
        inst = self.settings.instruments.get(underlying)
        if base_spot is None:
            base_spot = self.settings.feed.initial_spots.get(underlying, 22000.0)
        sigma = 0.15
        dt_per_snap = 3.0 / (252 * 6.25 * 3600)
        path = self.micro.generate_gbm_path(base_spot, 0.0, sigma, dt_per_snap, n_snapshots)
        session_start = datetime.combine(dt, datetime.min.time().replace(hour=9, minute=15))
        interval = timedelta(minutes=3)
        T = 15.0 / 365.0
        r = 0.065
        strike_interval = 50.0 if base_spot < 30000 else 100.0
        violation_snaps = set()
        n_violations = int(n_snapshots * violation_rate)
        if n_violations > 0:
            violation_snaps = set(self.rng.choice(n_snapshots, size=min(n_violations, n_snapshots), replace=False))
        chains = []
        for i in range(n_snapshots):
            spot = path[i]
            ts = session_start + interval * i
            atm = round(spot / strike_interval) * strike_interval
            strikes = []
            for j in range(-10, 11):
                k = atm + j * strike_interval
                iv = 0.15 + 0.02 * ((k - spot) / spot) ** 2 + self.rng.uniform(-0.005, 0.005)
                iv = max(0.05, iv)
                tc = black_scholes_call(spot, k, T, r, iv)
                tp = black_scholes_put(spot, k, T, r, iv)
                cp, pp = tc, tp
                if i in violation_snaps and abs(j) <= 2:
                    dev = spot * self.rng.uniform(0.002, 0.01)
                    if self.rng.random() > 0.5:
                        cp += dev
                    else:
                        pp += dev
                oi = max(10, int(5000 * np.exp(-0.2 * abs(j))) + self.rng.randint(-300, 300))
                cb, ca = self.micro.generate_bid_ask(max(0.05, cp), 0.08, oi)
                pb, pa = self.micro.generate_bid_ask(max(0.05, pp), 0.08, oi)
                strikes.append(StrikeData(
                    strike=k, call_bid=cb, call_ask=ca, call_ltp=round(cp, 2),
                    call_oi=oi, call_volume=int(oi * 0.1), call_iv=round(iv, 4),
                    put_bid=pb, put_ask=pa, put_ltp=round(pp, 2),
                    put_oi=oi, put_volume=int(oi * 0.1), put_iv=round(iv + 0.005, 4),
                    theoretical_call=round(tc, 2), theoretical_put=round(tp, 2)))
            chains.append(OptionChain(
                underlying=underlying, expiry=(dt + timedelta(days=15)).strftime("%d%b").upper(),
                spot_price=round(spot, 2), spot_bid=round(spot * 0.9999, 2),
                spot_ask=round(spot * 1.0001, 2), timestamp=ts,
                strikes=sorted(strikes, key=lambda s: s.strike), data_source="synthetic"))
        return chains

    def generate_and_store(self, underlying: str, start: date, end: date,
                            callback=None) -> int:
        """Generate and store synthetic sessions for a date range."""
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5:
                chains = self.generate_session(underlying, current)
                self.store.save_session(current, underlying, chains)
                count += 1
                if callback:
                    callback(current, True)
            current += timedelta(days=1)
        return count
