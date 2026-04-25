"""
Curriculum configuration and stage management for progressive training.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from config.settings import CurriculumConfig, CurriculumStage, get_settings

class CurriculumManager:
    """Manages curriculum progression during training."""

    def __init__(self, config: CurriculumConfig = None):
        settings = get_settings()
        self.config = config or settings.curriculum
        self._current_step = 0
        self._stage_history: List[str] = []

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.get_stage(self._current_step)

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    def advance(self, steps: int = 1):
        old_stage = self.stage_name
        self._current_step += steps
        new_stage = self.stage_name
        if new_stage != old_stage:
            self._stage_history.append(f"Step {self._current_step}: {old_stage} -> {new_stage}")

    def get_feed_config(self) -> Dict:
        stage = self.current_stage
        return {
            "feed_type": stage.feed_type,
            "underlyings": stage.underlyings,
            "violation_pct_range": stage.violation_pct_range,
            "violation_duration_range": stage.violation_duration_range,
            "num_strikes": stage.num_strikes,
            "fast_forward": stage.fast_forward_to_violation,
        }

    def should_fast_forward(self) -> bool:
        return self.current_stage.fast_forward_to_violation

    def get_violation_range(self) -> Tuple[float, float]:
        return self.current_stage.violation_pct_range

    def get_duration_range(self) -> Tuple[int, int]:
        return self.current_stage.violation_duration_range

    @property
    def step(self) -> int:
        return self._current_step

    @step.setter
    def step(self, value: int):
        self._current_step = value

    def reset(self):
        self._current_step = 0
        self._stage_history.clear()
