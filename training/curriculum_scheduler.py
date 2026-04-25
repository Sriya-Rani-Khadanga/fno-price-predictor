"""
Curriculum scheduler for training — manages feed and difficulty transitions.
"""
from __future__ import annotations
from typing import Optional
from data.feeds.base import BaseFeed
from data.feeds.mock_feed import MockFeed
from pcp_arb_env.curriculum import CurriculumManager
from config.settings import get_settings

class CurriculumScheduler:
    """Creates and manages feeds based on current curriculum stage."""

    def __init__(self, curriculum: CurriculumManager):
        self.curriculum = curriculum
        self.settings = get_settings()
        self._current_feed: Optional[BaseFeed] = None

    def get_feed(self) -> BaseFeed:
        """Get the appropriate feed for the current curriculum stage."""
        stage = self.curriculum.current_stage
        config = self.curriculum.get_feed_config()
        if stage.feed_type == "mock" or stage.feed_type == "mixed":
            vr = config["violation_pct_range"]
            dr = config["violation_duration_range"]
            feed = MockFeed(
                underlyings=config["underlyings"],
                violation_pct_range=vr,
                violation_duration_range=dr,
                violations_per_session=8,
                num_strikes=config["num_strikes"],
                seed=self.curriculum.step)
            self._current_feed = feed
            return feed
        elif stage.feed_type == "live_or_historical":
            try:
                from data.feeds.historical_feed import HistoricalFeed
                feed = HistoricalFeed(config["underlyings"][0])
                self._current_feed = feed
                return feed
            except Exception:
                feed = MockFeed(underlyings=config["underlyings"],
                                violation_pct_range=config["violation_pct_range"],
                                violation_duration_range=config["violation_duration_range"])
                self._current_feed = feed
                return feed
        feed = MockFeed(underlyings=config["underlyings"])
        self._current_feed = feed
        return feed

    def should_advance(self, metrics: dict) -> bool:
        """Check if we should advance to next curriculum stage."""
        return False  # Advancement is step-based in CurriculumManager
