"""
Historical feed — replays stored session data at configurable speed.
Reconstructs intraday movement from EOD data using GBM interpolation.
"""
from __future__ import annotations
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from data.feeds.base import BaseFeed, TickData
from data.processors.options_chain import OptionChain, StrikeData, black_scholes_call, black_scholes_put
from data.processors.microstructure import MicrostructureGenerator
from data.historical.store import HistoricalStore
from data.historical.nse_downloader import NSEDownloader
from data.historical.bhavcopy_parser import BhavcopyParser
from config.settings import get_settings

class HistoricalFeed(BaseFeed):
    """Replays historical sessions from stored data or downloaded bhavcopies."""

    def __init__(self, underlying: str, replay_date: date = None,
                 speed_multiplier: float = 60.0, seed: int = 42):
        super().__init__([underlying])
        self.underlying = underlying
        self.replay_date = replay_date
        self.speed_multiplier = speed_multiplier
        self.store = HistoricalStore()
        self.downloader = NSEDownloader()
        self.parser = BhavcopyParser()
        self.micro = MicrostructureGenerator(seed)
        self.rng = np.random.RandomState(seed)
        self.settings = get_settings()
        self._chains: List[OptionChain] = []
        self._chain_idx = 0
        self._interpolated_ticks: List[TickData] = []
        self._done = False
        self._tick_interval = self.settings.feed.tick_interval_seconds

    def reset(self, start_time: Optional[datetime] = None) -> TickData:
        self._tick_count = 0
        self._chain_idx = 0
        self._done = False
        if self.replay_date is None:
            available = self.store.list_available_dates(self.underlying)
            if available:
                self.replay_date = available[-1]
            else:
                bhavcopy_dates = self.store.list_bhavcopy_dates()
                if bhavcopy_dates:
                    self.replay_date = bhavcopy_dates[-1]
                else:
                    self.replay_date = date.today() - timedelta(days=1)
        self._chains = self.store.load_session(self.replay_date, self.underlying)
        if not self._chains:
            self._chains = self._load_from_bhavcopy()
        if not self._chains:
            from data.historical.generator import SyntheticGenerator
            gen = SyntheticGenerator()
            self._chains = gen.generate_session(self.underlying, self.replay_date)
        self._interpolated_ticks = self._interpolate_to_ticks()
        
        if start_time:
            session_start = datetime.combine(self.replay_date, datetime.min.time().replace(hour=9, minute=15))
            elapsed = (start_time - session_start).total_seconds()
            self._tick_count = max(0, int(elapsed / self._tick_interval))

        if self._interpolated_ticks and self._tick_count < len(self._interpolated_ticks):
            self._current_tick = self._interpolated_ticks[self._tick_count]
        else:
            self._current_tick = self._make_empty_tick()
        self._session_start = self._current_tick.timestamp
        return self._current_tick

    def next_tick(self) -> TickData:
        self._tick_count += 1
        if self._tick_count >= len(self._interpolated_ticks):
            self._done = True
            if self._interpolated_ticks:
                self._current_tick = self._interpolated_ticks[-1]
                self._current_tick.is_session_end = True
            return self._current_tick
        self._current_tick = self._interpolated_ticks[self._tick_count]
        return self._current_tick

    def is_done(self) -> bool:
        return self._done

    def _load_from_bhavcopy(self) -> List[OptionChain]:
        df = self.downloader.download_bhavcopy(self.replay_date)
        if df is None:
            return []
        return self.parser.parse(df, self.underlying, self.replay_date)

    def _interpolate_to_ticks(self) -> List[TickData]:
        """Interpolate stored snapshots into per-tick data using GBM bridges."""
        if not self._chains:
            return []
        total_session_seconds = self.settings.feed.session_duration_minutes * 60
        n_ticks = total_session_seconds // self._tick_interval
        session_start = datetime.combine(
            self.replay_date, datetime.min.time().replace(hour=9, minute=15))
        
        # Use first chain if available, else make dummy
        if self._chains:
            base_chain = self._chains[0]
            spot = base_chain.spot_price
        else:
            # Fallback spot if no chains at all
            spot = 22000.0
            from data.processors.options_chain import OptionChain
            base_chain = OptionChain(underlying=self.underlying, expiry="2024-04-25",
                                     spot_price=spot, spot_bid=spot*0.999, spot_ask=spot*1.001,
                                     timestamp=session_start, strikes=[], data_source="synthetic")

        bhavcopy_df = self.downloader.download_bhavcopy(self.replay_date)
        if bhavcopy_df is None or bhavcopy_df.empty:
            ohlc = {}
        else:
            ohlc = self.parser.extract_ohlc(bhavcopy_df, self.underlying)
        
        if ohlc and ohlc.get("high", 0) > 0:
            spot_path = self.micro.generate_intraday_from_ohlc(
                ohlc.get("open", spot), ohlc.get("high", spot * 1.005),
                ohlc.get("low", spot * 0.995), ohlc.get("close", spot), n_ticks)
        else:
            dt_per = self._tick_interval / (252 * 6.25 * 3600)
            spot_path = self.micro.generate_gbm_path(spot, 0.0, 0.15, dt_per, n_ticks)
        violation_ticks = set()
        n_violations = self.rng.randint(3, 10)
        for _ in range(n_violations):
            start = self.rng.randint(10, max(11, n_ticks - 50))
            duration = self.rng.randint(10, 40)
            for t in range(start, min(start + duration, n_ticks)):
                violation_ticks.add(t)
        ticks = []
        T = 15.0 / 365.0
        r = 0.065
        for i in range(n_ticks):
            ts = session_start + timedelta(seconds=i * self._tick_interval)
            s = spot_path[min(i, len(spot_path) - 1)]
            chain = self._rebuild_chain(base_chain, s, ts, T, r, i in violation_ticks)
            minute = (i * self._tick_interval) // 60
            ticks.append(TickData(
                timestamp=ts, session_minute=minute,
                chains={self.underlying: chain}, spots={self.underlying: round(s, 2)},
                is_session_end=(i == n_ticks - 1),
                violations_injected=1 if i in violation_ticks else 0))
        return ticks

    def _rebuild_chain(self, base: OptionChain, spot: float, ts: datetime,
                        T: float, r: float, inject_violation: bool) -> OptionChain:
        """Rebuild chain at a new spot level with noise and optional violation."""
        strikes = []
        for s in base.strikes:
            iv = s.call_iv + self.rng.uniform(-0.005, 0.005)
            iv = max(0.05, iv)
            tc = black_scholes_call(spot, s.strike, T, r, iv)
            tp = black_scholes_put(spot, s.strike, T, r, iv)
            cp, pp = tc, tp
            if inject_violation and abs(s.strike - spot) < spot * 0.02:
                dev = spot * self.rng.uniform(0.003, 0.008)
                if self.rng.random() > 0.5:
                    cp += dev
                else:
                    pp += dev
            cp = max(0.05, cp)
            pp = max(0.05, pp)
            cb, ca = self.micro.generate_bid_ask(cp, 0.08, s.call_oi)
            pb, pa = self.micro.generate_bid_ask(pp, 0.08, s.put_oi)
            strikes.append(StrikeData(
                strike=s.strike, call_bid=cb, call_ask=ca, call_ltp=round(cp, 2),
                call_oi=s.call_oi, call_volume=s.call_volume, call_iv=round(iv, 4),
                put_bid=pb, put_ask=pa, put_ltp=round(pp, 2),
                put_oi=s.put_oi, put_volume=s.put_volume, put_iv=round(iv + 0.005, 4),
                theoretical_call=round(tc, 2), theoretical_put=round(tp, 2)))
        return OptionChain(
            underlying=base.underlying, expiry=base.expiry,
            spot_price=round(spot, 2), spot_bid=round(spot * 0.9999, 2),
            spot_ask=round(spot * 1.0001, 2), timestamp=ts,
            strikes=strikes, data_source="historical")

    def _make_empty_tick(self) -> TickData:
        return TickData(timestamp=datetime.now(), session_minute=0,
                        chains={}, spots={}, is_session_end=True)
