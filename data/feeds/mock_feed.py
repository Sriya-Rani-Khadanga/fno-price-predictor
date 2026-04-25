"""
Mock data feed for training. Generates synthetic market data with GBM price paths,
realistic microstructure noise, and tagged PCP violations at configurable rates.
"""
from __future__ import annotations
import math
import random
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from data.feeds.base import BaseFeed, TickData
from data.processors.options_chain import (OptionChain, StrikeData, black_scholes_call,
                                            black_scholes_put, implied_vol_call, implied_vol_put)
from data.processors.microstructure import MicrostructureGenerator
from config.settings import get_settings

class MockFeed(BaseFeed):
    """Synthetic feed with GBM prices and injected PCP violations for training."""

    def __init__(self, underlyings: List[str] = None, violation_pct_range: Tuple[float, float] = (0.5, 2.0),
                 violation_duration_range: Tuple[int, int] = (60, 180), violations_per_session: int = 8,
                 num_strikes: int = 10, seed: int = 42):
        settings = get_settings()
        if underlyings is None:
            underlyings = settings.feed.mock_underlyings
        super().__init__(underlyings)
        self.violation_pct_range = violation_pct_range
        self.violation_duration_range = violation_duration_range
        self.violations_per_session = violations_per_session
        self.num_strikes = num_strikes
        self.micro = MicrostructureGenerator(seed)
        self.rng = np.random.RandomState(seed)
        self._settings = settings
        self._tick_interval = settings.feed.tick_interval_seconds
        self._session_minutes = settings.feed.session_duration_minutes
        self._total_ticks = (self._session_minutes * 60) // self._tick_interval
        self._spot_paths: Dict[str, np.ndarray] = {}
        self._violation_schedule: List[Dict] = []
        self._current_minute = 0
        self._done = False

    def reset(self) -> TickData:
        self._tick_count = 0
        self._current_minute = 0
        self._done = False
        self._session_start = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        self._spot_paths = {}
        for sym in self.underlyings:
            s0 = self._settings.feed.initial_spots.get(sym, 22000.0)
            sigma = 0.15 if sym != "CRUDEOIL" else 0.25
            dt = self._tick_interval / (252 * 6.25 * 3600)
            path = self.micro.generate_gbm_path(s0, 0.0, sigma, dt, self._total_ticks)
            self._spot_paths[sym] = path
        self._violation_schedule = self._schedule_violations()
        self._current_tick = self._build_tick()
        return self._current_tick

    def next_tick(self) -> TickData:
        self._tick_count += 1
        self._current_minute = (self._tick_count * self._tick_interval) / 60.0
        if self._tick_count >= self._total_ticks:
            self._done = True
        self._current_tick = self._build_tick()
        return self._current_tick

    def is_done(self) -> bool:
        return self._done

    def _schedule_violations(self) -> List[Dict]:
        schedule = []
        for _ in range(self.violations_per_session):
            start_tick = self.rng.randint(10, self._total_ticks - 100)
            duration_ticks = self.rng.randint(self.violation_duration_range[0] // self._tick_interval,
                                               self.violation_duration_range[1] // self._tick_interval + 1)
            magnitude = self.rng.uniform(self.violation_pct_range[0], self.violation_pct_range[1])
            underlying = self.rng.choice(self.underlyings)
            strike_offset = self.rng.choice([-2, -1, 0, 1, 2])
            direction = self.rng.choice(["call_rich", "put_rich"])
            schedule.append({
                "start_tick": start_tick, "end_tick": start_tick + duration_ticks,
                "magnitude_pct": magnitude, "underlying": underlying,
                "strike_offset": strike_offset, "direction": direction,
                "peak_tick": start_tick + duration_ticks // 2})
        return sorted(schedule, key=lambda x: x["start_tick"])

    def _get_active_violations(self) -> List[Dict]:
        return [v for v in self._violation_schedule if v["start_tick"] <= self._tick_count <= v["end_tick"]]

    def _build_tick(self) -> TickData:
        ts = self._session_start + timedelta(seconds=self._tick_count * self._tick_interval)
        chains: Dict[str, OptionChain] = {}
        spots: Dict[str, float] = {}
        violations_count = 0
        active_violations = self._get_active_violations()
        for sym in self.underlyings:
            path = self._spot_paths[sym]
            idx = min(self._tick_count, len(path) - 1)
            spot = path[idx]
            spots[sym] = round(spot, 2)
            inst = self._settings.instruments.get(sym)
            lot_size = inst.lot_size if inst else 50
            strike_interval = 50.0 if spot < 30000 else 100.0
            if sym == "CRUDEOIL":
                strike_interval = 50.0
            atm = round(spot / strike_interval) * strike_interval
            strikes_list = []
            T = 15.0 / 365.0
            r = 0.065
            atm_iv = 0.15 + self.rng.uniform(-0.02, 0.02)
            sym_violations = [v for v in active_violations if v["underlying"] == sym]
            for i in range(-self.num_strikes, self.num_strikes + 1):
                k = atm + i * strike_interval
                moneyness = math.log(k / spot) if spot > 0 else 0
                call_iv = atm_iv + 0.02 * moneyness ** 2 + 0.01 * moneyness + self.rng.uniform(-0.003, 0.003)
                put_iv = call_iv + 0.005 + self.rng.uniform(-0.002, 0.002)
                call_iv = max(0.05, call_iv)
                put_iv = max(0.05, put_iv)
                tc = black_scholes_call(spot, k, T, r, call_iv)
                tp = black_scholes_put(spot, k, T, r, put_iv)
                call_price = tc
                put_price = tp
                for v in sym_violations:
                    if v["strike_offset"] == i or (v["strike_offset"] == 0 and abs(i) <= 1):
                        progress = (self._tick_count - v["start_tick"]) / max(1, v["end_tick"] - v["start_tick"])
                        envelope = math.sin(math.pi * progress)
                        dev_amount = spot * (v["magnitude_pct"] / 100.0) * envelope
                        if v["direction"] == "call_rich":
                            call_price += dev_amount
                        else:
                            put_price += dev_amount
                        violations_count += 1
                call_price = max(0.05, call_price)
                put_price = max(0.05, put_price)
                oi_base = int(5000 * math.exp(-0.2 * abs(i)))
                call_oi = max(10, oi_base + self.rng.randint(-500, 500))
                put_oi = max(10, oi_base + self.rng.randint(-500, 500))
                cb, ca = self.micro.generate_bid_ask(call_price, 0.08, call_oi)
                pb, pa = self.micro.generate_bid_ask(put_price, 0.08, put_oi)
                vol_weight = self.micro.get_volume_weight(int(self._current_minute))
                call_vol = max(0, int(call_oi * vol_weight * 100 * self.rng.uniform(0.5, 2.0)))
                put_vol = max(0, int(put_oi * vol_weight * 100 * self.rng.uniform(0.5, 2.0)))
                strikes_list.append(StrikeData(
                    strike=k, call_bid=cb, call_ask=ca, call_ltp=round(call_price, 2),
                    call_oi=call_oi, call_volume=call_vol, call_iv=round(call_iv, 4),
                    put_bid=pb, put_ask=pa, put_ltp=round(put_price, 2),
                    put_oi=put_oi, put_volume=put_vol, put_iv=round(put_iv, 4),
                    theoretical_call=round(tc, 2), theoretical_put=round(tp, 2)))
            sb, sa = self.micro.generate_bid_ask(spot, 0.01, 100000)
            exp_date = self._session_start + timedelta(days=15)
            chains[sym] = OptionChain(
                underlying=sym, expiry=exp_date.strftime("%d%b").upper(),
                spot_price=round(spot, 2), spot_bid=sb, spot_ask=sa,
                timestamp=ts, strikes=strikes_list, data_source="mock")
        return TickData(timestamp=ts, session_minute=int(self._current_minute),
                        chains=chains, spots=spots, is_session_end=self._done,
                        violations_injected=violations_count)
