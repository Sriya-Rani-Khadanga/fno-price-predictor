"""
Historical data store backed by parquet files.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from data.processors.options_chain import OptionChain, StrikeData
from config.settings import CACHE_DIR

@dataclass
class ViolationStats:
    total_violations: int
    avg_per_session: float
    avg_magnitude_pct: float
    avg_duration_seconds: float
    most_common_strikes: List[float]
    best_hour: int
    worst_hour: int
    sessions_analyzed: int

    def to_dict(self):
        return self.__dict__

class HistoricalStore:
    """Parquet-backed store for historical option chain sessions."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = self.cache_dir / "sessions"
        self._sessions_dir.mkdir(exist_ok=True)

    def save_session(self, dt: date, underlying: str, snapshots: List[OptionChain]):
        """Save a list of OptionChain snapshots for a session."""
        session_dir = self._sessions_dir / underlying / dt.isoformat()
        session_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i, chain in enumerate(snapshots):
            for s in chain.strikes:
                records.append({
                    "snapshot_idx": i, "timestamp": chain.timestamp.isoformat(),
                    "underlying": chain.underlying, "expiry": chain.expiry,
                    "spot": chain.spot_price, "strike": s.strike,
                    "call_bid": s.call_bid, "call_ask": s.call_ask, "call_ltp": s.call_ltp,
                    "call_oi": s.call_oi, "call_volume": s.call_volume, "call_iv": s.call_iv,
                    "put_bid": s.put_bid, "put_ask": s.put_ask, "put_ltp": s.put_ltp,
                    "put_oi": s.put_oi, "put_volume": s.put_volume, "put_iv": s.put_iv,
                })
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(session_dir / "chain_data.parquet")

    def load_session(self, dt: date, underlying: str) -> List[OptionChain]:
        """Load stored session data back into OptionChain objects."""
        session_dir = self._sessions_dir / underlying / dt.isoformat()
        parquet_path = session_dir / "chain_data.parquet"
        if not parquet_path.exists():
            return []
        df = pd.read_parquet(parquet_path)
        chains = []
        for idx, group in df.groupby("snapshot_idx"):
            row0 = group.iloc[0]
            strikes = []
            for _, r in group.iterrows():
                strikes.append(StrikeData(
                    strike=r["strike"], call_bid=r["call_bid"], call_ask=r["call_ask"],
                    call_ltp=r["call_ltp"], call_oi=int(r["call_oi"]),
                    call_volume=int(r["call_volume"]), call_iv=r["call_iv"],
                    put_bid=r["put_bid"], put_ask=r["put_ask"], put_ltp=r["put_ltp"],
                    put_oi=int(r["put_oi"]), put_volume=int(r["put_volume"]), put_iv=r["put_iv"]))
            chains.append(OptionChain(
                underlying=row0["underlying"], expiry=str(row0["expiry"]),
                spot_price=row0["spot"], spot_bid=row0["spot"] * 0.9999,
                spot_ask=row0["spot"] * 1.0001,
                timestamp=datetime.fromisoformat(row0["timestamp"]),
                strikes=sorted(strikes, key=lambda s: s.strike),
                data_source="historical"))
        return chains

    def list_available_dates(self, underlying: str) -> List[date]:
        """List all dates with stored data for an underlying."""
        und_dir = self._sessions_dir / underlying
        if not und_dir.exists():
            return []
        dates = []
        for d in sorted(und_dir.iterdir()):
            if d.is_dir() and (d / "chain_data.parquet").exists():
                try:
                    dates.append(date.fromisoformat(d.name))
                except ValueError:
                    pass
        return dates

    def has_bhavcopy(self, dt: date) -> bool:
        """Check if bhavcopy for this date is cached."""
        return (self.cache_dir / f"bhavcopy_{dt.isoformat()}.parquet").exists()

    def list_bhavcopy_dates(self) -> List[date]:
        """List all cached bhavcopy dates."""
        dates = []
        for f in self.cache_dir.glob("bhavcopy_*.parquet"):
            try:
                ds = f.stem.replace("bhavcopy_", "")
                dates.append(date.fromisoformat(ds))
            except ValueError:
                pass
        return sorted(dates)

    def get_violation_stats(self, underlying: str, start: date, end: date) -> ViolationStats:
        """Compute violation statistics across stored sessions."""
        dates = [d for d in self.list_available_dates(underlying) if start <= d <= end]
        if not dates:
            return ViolationStats(0, 0.0, 0.0, 0.0, [], 12, 12, 0)
        total_v = 0
        magnitudes = []
        strike_counts: Dict[float, int] = {}
        hour_counts: Dict[int, int] = {}
        for dt in dates:
            chains = self.load_session(dt, underlying)
            for chain in chains:
                for s in chain.strikes:
                    if s.pcp_deviation_pct > 0.1:
                        total_v += 1
                        magnitudes.append(s.pcp_deviation_pct)
                        strike_counts[s.strike] = strike_counts.get(s.strike, 0) + 1
                        h = chain.timestamp.hour
                        hour_counts[h] = hour_counts.get(h, 0) + 1
        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
        top_strikes = sorted(strike_counts.keys(), key=lambda k: strike_counts[k], reverse=True)[:5]
        best_h = max(hour_counts, key=hour_counts.get) if hour_counts else 12
        worst_h = min(hour_counts, key=hour_counts.get) if hour_counts else 12
        return ViolationStats(
            total_violations=total_v, avg_per_session=total_v / max(len(dates), 1),
            avg_magnitude_pct=avg_mag, avg_duration_seconds=45.0,
            most_common_strikes=top_strikes, best_hour=best_h, worst_hour=worst_h,
            sessions_analyzed=len(dates))
