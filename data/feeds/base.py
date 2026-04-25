"""
Base feed interface for all data feed implementations.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from data.processors.options_chain import OptionChain

@dataclass
class TickData:
    """A single tick of market data."""
    timestamp: datetime
    session_minute: int
    chains: Dict[str, OptionChain]  # underlying -> chain
    spots: Dict[str, float]
    is_session_end: bool = False
    violations_injected: int = 0

class BaseFeed(ABC):
    """Abstract base class for all market data feeds."""

    def __init__(self, underlyings: List[str]):
        self.underlyings = underlyings
        self._current_tick: Optional[TickData] = None
        self._tick_count = 0
        self._session_start: Optional[datetime] = None

    @abstractmethod
    def reset(self) -> TickData:
        """Reset feed to session start. Returns initial tick."""
        ...

    @abstractmethod
    def next_tick(self) -> TickData:
        """Advance feed by one tick and return new data."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Check if feed has reached session end."""
        ...

    @property
    def current_tick(self) -> Optional[TickData]:
        return self._current_tick

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def fast_forward_to_violation(self, min_deviation_pct: float = 0.5) -> Optional[TickData]:
        """Fast-forward feed until a violation is detected. Used in curriculum stage 1."""
        for _ in range(10000):
            tick = self.next_tick()
            if tick.is_session_end:
                return tick
            for chain in tick.chains.values():
                for s in chain.strikes:
                    if s.pcp_deviation_pct >= min_deviation_pct:
                        return tick
        return self.next_tick()
