"""
Global settings loader for PCP Arb RL system.
Loads instruments.yaml and provides typed access to all configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "historical" / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
for _d in [CACHE_DIR, LOGS_DIR, REPORTS_DIR, RECORDINGS_DIR, CHECKPOINTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class InstrumentConfig:
    """Configuration for a single tradeable instrument."""
    symbol: str
    lot_size: int
    tick_size: float
    exchange: str
    segment: str
    cost_model: str
    trading_hours_start: time
    trading_hours_end: time
    circuit_breaker_upper: float
    circuit_breaker_lower: float
    no_new_positions_after: time
    force_close_by: time
    margin_pct: float
    expiry_calendar_2024: List[date]
    expiry_calendar_2025: List[date]
    option_chain_url: Optional[str] = None
    data_url: Optional[str] = None
    ctt_applicable: bool = False

    def get_next_expiry(self, from_date: date) -> Optional[date]:
        """Return the next expiry date on or after from_date."""
        all_expiries = sorted(self.expiry_calendar_2024 + self.expiry_calendar_2025)
        for exp in all_expiries:
            if exp >= from_date:
                return exp
        return None

    def days_to_expiry(self, from_date: date) -> int:
        """Return calendar days to next expiry from given date."""
        nxt = self.get_next_expiry(from_date)
        if nxt is None:
            return 30  # fallback
        return (nxt - from_date).days

    def is_trading_hour(self, t: time) -> bool:
        """Check if time is within trading hours."""
        return self.trading_hours_start <= t <= self.trading_hours_end

    def can_open_position(self, t: time) -> bool:
        """Check if new positions can be opened at this time."""
        return t <= self.no_new_positions_after

    def must_close_all(self, t: time) -> bool:
        """Check if all positions must be closed by this time."""
        return t >= self.force_close_by


@dataclass
class CostModelConfig:
    """Cost model parameters for a segment."""
    brokerage_per_order: float
    sebi_charges: float
    exchange_txn_charges: float
    gst: float
    stamp_duty: float
    slippage_bps: float
    stt_on_sell: float = 0.0
    stt_on_exercise: float = 0.0
    ctt: float = 0.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_positions: int
    max_capital_per_trade: float
    max_daily_loss: float
    max_holding_seconds: int
    initial_capital: float


@dataclass
class FeedConfig:
    """Feed configuration."""
    mock_underlyings: List[str]
    session_duration_minutes: int
    tick_interval_seconds: int
    violation_rate_per_session: int
    initial_spots: Dict[str, float]
    poll_interval_min: float
    poll_interval_max: float
    staleness_threshold_seconds: int
    session_cookie_refresh_minutes: int
    speed_multiplier: float
    cache_dir: str
    lookback_days: int


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str
    quantization: str
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    learning_rate: float
    gradient_clip: float
    schedule: str
    lora_r: int
    lora_alpha: int
    wandb_project: str
    checkpoint_interval: int


@dataclass
class CurriculumStage:
    """A single curriculum stage configuration."""
    name: str
    step_range: tuple
    violation_pct_range: tuple
    violation_duration_range: tuple
    underlyings: List[str]
    num_strikes: int
    fast_forward_to_violation: bool
    feed_type: str


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""
    stages: List[CurriculumStage]

    def get_stage(self, step: int) -> CurriculumStage:
        """Return the curriculum stage for the given training step."""
        for stage in self.stages:
            lo, hi = stage.step_range
            if lo <= step < hi:
                return stage
        return self.stages[-1]


@dataclass
class Settings:
    """Master settings object containing all configuration."""
    instruments: Dict[str, InstrumentConfig]
    cost_models: Dict[str, CostModelConfig]
    risk: RiskConfig
    feed: FeedConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def _parse_time(s: str) -> time:
    """Parse a time string like '09:15' into a datetime.time."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


def _parse_date(s: str) -> date:
    """Parse a date string like '2024-01-25' into a datetime.date."""
    if isinstance(s, date):
        return s
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from instruments.yaml.
    
    Args:
        config_path: Optional path to config file. Defaults to config/instruments.yaml.
    
    Returns:
        Settings object with all configuration loaded and typed.
    """
    if config_path is None:
        config_path = str(CONFIG_DIR / "instruments.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse instruments
    instruments: Dict[str, InstrumentConfig] = {}
    for sym, cfg in raw.get("instruments", {}).items():
        instruments[sym] = InstrumentConfig(
            symbol=sym,
            lot_size=cfg["lot_size"],
            tick_size=cfg["tick_size"],
            exchange=cfg["exchange"],
            segment=cfg.get("segment", "indices"),
            cost_model=cfg["cost_model"],
            trading_hours_start=_parse_time(cfg["trading_hours"]["start"]),
            trading_hours_end=_parse_time(cfg["trading_hours"]["end"]),
            circuit_breaker_upper=cfg["circuit_breaker"]["upper_pct"],
            circuit_breaker_lower=cfg["circuit_breaker"]["lower_pct"],
            no_new_positions_after=_parse_time(cfg["no_new_positions_after"]),
            force_close_by=_parse_time(cfg["force_close_by"]),
            margin_pct=cfg["margin_pct"],
            option_chain_url=cfg.get("option_chain_url"),
            data_url=cfg.get("data_url"),
            ctt_applicable=cfg.get("ctt_applicable", False),
            expiry_calendar_2024=[_parse_date(d) for d in cfg.get("expiry_calendar_2024", [])],
            expiry_calendar_2025=[_parse_date(d) for d in cfg.get("expiry_calendar_2025", [])],
        )

    # Parse cost models
    cost_models: Dict[str, CostModelConfig] = {}
    for name, cfg in raw.get("cost_models", {}).items():
        cost_models[name] = CostModelConfig(
            brokerage_per_order=cfg["brokerage_per_order"],
            sebi_charges=cfg["sebi_charges"],
            exchange_txn_charges=cfg["exchange_txn_charges"],
            gst=cfg["gst"],
            stamp_duty=cfg["stamp_duty"],
            slippage_bps=cfg["slippage_bps"],
            stt_on_sell=cfg.get("stt_on_sell", 0.0),
            stt_on_exercise=cfg.get("stt_on_exercise", 0.0),
            ctt=cfg.get("ctt", 0.0),
        )

    # Parse risk limits
    rl = raw.get("risk_limits", {})
    risk = RiskConfig(
        max_positions=rl.get("max_positions", 5),
        max_capital_per_trade=rl.get("max_capital_per_trade", 200000.0),
        max_daily_loss=rl.get("max_daily_loss", 50000.0),
        max_holding_seconds=rl.get("max_holding_seconds", 300),
        initial_capital=rl.get("initial_capital", 1000000.0),
    )

    # Parse feed config
    fc = raw.get("feed", {})
    mc = fc.get("mock", {})
    lc = fc.get("live", {})
    hc = fc.get("historical", {})
    feed = FeedConfig(
        mock_underlyings=mc.get("underlyings", ["NIFTY"]),
        session_duration_minutes=mc.get("session_duration_minutes", 375),
        tick_interval_seconds=mc.get("tick_interval_seconds", 3),
        violation_rate_per_session=mc.get("violation_rate_per_session", 8),
        initial_spots=mc.get("initial_spots", {"NIFTY": 22000.0}),
        poll_interval_min=lc.get("poll_interval_min", 2.5),
        poll_interval_max=lc.get("poll_interval_max", 3.5),
        staleness_threshold_seconds=lc.get("staleness_threshold_seconds", 10),
        session_cookie_refresh_minutes=lc.get("session_cookie_refresh_minutes", 5),
        speed_multiplier=hc.get("speed_multiplier", 60.0),
        cache_dir=hc.get("cache_dir", "data/historical/cache"),
        lookback_days=hc.get("lookback_days", 90),
    )

    # Parse training config
    tc = raw.get("training", {})
    training = TrainingConfig(
        model_name=tc.get("model_name", "unsloth/Qwen2.5-1.5B-Instruct"),
        quantization=tc.get("quantization", "4bit"),
        num_generations=tc.get("num_generations", 6),
        max_prompt_length=tc.get("max_prompt_length", 1024),
        max_completion_length=tc.get("max_completion_length", 120),
        learning_rate=tc.get("learning_rate", 2e-6),
        gradient_clip=tc.get("gradient_clip", 1.0),
        schedule=tc.get("schedule", "cosine"),
        lora_r=tc.get("lora_r", 16),
        lora_alpha=tc.get("lora_alpha", 16),
        wandb_project=tc.get("wandb_project", "pcp-arb-rl"),
        checkpoint_interval=tc.get("checkpoint_interval", 100),
    )

    # Parse curriculum
    stages = []
    for sc in raw.get("curriculum", {}).get("stages", []):
        stages.append(CurriculumStage(
            name=sc["name"],
            step_range=tuple(sc["step_range"]),
            violation_pct_range=tuple(sc["violation_pct_range"]),
            violation_duration_range=tuple(sc["violation_duration_range"]),
            underlyings=sc["underlyings"],
            num_strikes=sc["num_strikes"],
            fast_forward_to_violation=sc["fast_forward_to_violation"],
            feed_type=sc["feed_type"],
        ))
    curriculum = CurriculumConfig(stages=stages)

    return Settings(
        instruments=instruments,
        cost_models=cost_models,
        risk=risk,
        feed=feed,
        training=training,
        curriculum=curriculum,
        raw=raw,
    )


# Global singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
