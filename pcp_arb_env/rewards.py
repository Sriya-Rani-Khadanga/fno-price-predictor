"""
Reward functions for PCP Arb RL environment.
Four independent reward components: profitability, timing, cost awareness, format compliance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RewardBreakdown:
    profitability: float
    timing: float
    cost_awareness: float
    format_compliance: float
    total: float
    details: Dict[str, str]

    def to_dict(self) -> Dict:
        return {"profitability": round(self.profitability, 4),
                "timing": round(self.timing, 4),
                "cost_awareness": round(self.cost_awareness, 4),
                "format_compliance": round(self.format_compliance, 4),
                "total": round(self.total, 4),
                "details": self.details}

def compute_profitability_reward(realized_pnl_delta: float, unrealized_pnl: float,
                                  daily_pnl: float, max_daily_loss: float) -> tuple:
    """Reward for profitable trading. Penalize losses, reward gains."""
    reward = 0.0
    detail = ""
    if realized_pnl_delta > 0:
        reward = min(1.0, realized_pnl_delta / 1000.0)
        detail = f"Realized +₹{realized_pnl_delta:.0f}"
    elif realized_pnl_delta < 0:
        reward = max(-1.0, realized_pnl_delta / 1000.0)
        detail = f"Realized -₹{abs(realized_pnl_delta):.0f}"
    else:
        if unrealized_pnl > 0:
            reward = min(0.3, unrealized_pnl / 3000.0)
            detail = f"Unrealized +₹{unrealized_pnl:.0f}"
        elif unrealized_pnl < 0:
            reward = max(-0.3, unrealized_pnl / 3000.0)
            detail = f"Unrealized -₹{abs(unrealized_pnl):.0f}"
    if daily_pnl < -max_daily_loss * 0.8:
        reward -= 0.5
        detail += " [NEAR DAILY LIMIT]"
    return reward, detail

def compute_timing_reward(action_type: str, deviation_pct: float,
                           active_seconds: float, trend: str,
                           breakeven_pct: float) -> tuple:
    """Reward for good timing — enter early in violations, exit before they close."""
    reward = 0.0
    detail = ""
    if "enter" in action_type:
        if deviation_pct > breakeven_pct:
            margin = deviation_pct - breakeven_pct
            reward = min(0.8, margin / 0.5)
            if active_seconds < 30:
                reward += 0.2
                detail = f"Early entry at {deviation_pct:.2f}% (BE:{breakeven_pct:.2f}%), {active_seconds:.0f}s active"
            elif active_seconds > 120:
                reward -= 0.2
                detail = f"Late entry at {deviation_pct:.2f}%, {active_seconds:.0f}s active (late)"
            else:
                detail = f"Entry at {deviation_pct:.2f}% after {active_seconds:.0f}s"
            if trend == "narrowing":
                reward -= 0.3
                detail += " [NARROWING - risky]"
        else:
            reward = -0.5
            detail = f"Entry BELOW breakeven ({deviation_pct:.2f}% < {breakeven_pct:.2f}%)"
    elif "exit" in action_type:
        if deviation_pct < breakeven_pct * 0.5:
            reward = 0.5
            detail = f"Good exit as deviation collapsed to {deviation_pct:.2f}%"
        elif trend == "narrowing":
            reward = 0.3
            detail = f"Exit on narrowing trend at {deviation_pct:.2f}%"
    elif action_type == "hold":
        if deviation_pct > breakeven_pct and trend != "narrowing":
            reward = 0.1
            detail = "Holding profitable position, trend OK"
        elif deviation_pct < breakeven_pct:
            reward = -0.1
            detail = "Holding below breakeven — consider exit"
    return reward, detail

def compute_cost_awareness_reward(action_type: str, used_cost_tools: bool,
                                    margin_over_breakeven: float,
                                    called_stt_trap: bool,
                                    is_near_expiry: bool) -> tuple:
    """Reward for cost-aware decision making — using cost tools before entry."""
    reward = 0.0
    detail = ""
    if "enter" in action_type:
        if used_cost_tools:
            reward += 0.3
            detail = "Checked costs before entry"
        else:
            reward -= 0.3
            detail = "Entered WITHOUT checking costs"
        if margin_over_breakeven > 0.1:
            reward += 0.2
            detail += f", margin +{margin_over_breakeven:.2f}% over BE"
        elif margin_over_breakeven < 0:
            reward -= 0.5
            detail += f", BELOW breakeven by {abs(margin_over_breakeven):.2f}%"
        if is_near_expiry and not called_stt_trap:
            reward -= 0.3
            detail += " [STT TRAP NOT CHECKED near expiry!]"
        elif called_stt_trap:
            reward += 0.2
            detail += ", STT trap checked"
    return reward, detail

def compute_format_reward(parsed_ok: bool, has_action_type: bool,
                           valid_action: bool) -> tuple:
    """Reward for correct output format — valid JSON with required fields."""
    if parsed_ok and has_action_type and valid_action:
        return 0.5, "Valid JSON output"
    elif parsed_ok and has_action_type:
        return 0.2, "Parseable but invalid action"
    elif parsed_ok:
        return 0.0, "JSON parsed but missing action_type"
    else:
        return -1.0, "PARSE FAILURE — invalid output format"

def compute_total_reward(profitability: float, timing: float,
                          cost_awareness: float, format_compliance: float,
                          weights: Dict[str, float] = None) -> float:
    """Weighted sum of all reward components."""
    if weights is None:
        weights = {"profitability": 0.35, "timing": 0.25,
                   "cost_awareness": 0.25, "format_compliance": 0.15}
    return (profitability * weights["profitability"] +
            timing * weights["timing"] +
            cost_awareness * weights["cost_awareness"] +
            format_compliance * weights["format_compliance"])

def compute_reward(action_type: str, realized_pnl_delta: float, unrealized_pnl: float,
                   daily_pnl: float, max_daily_loss: float, deviation_pct: float,
                   active_seconds: float, trend: str, breakeven_pct: float,
                   used_cost_tools: bool, margin_over_breakeven: float,
                   called_stt_trap: bool, is_near_expiry: bool,
                   parsed_ok: bool, has_action_type: bool, valid_action: bool,
                   weights: Dict[str, float] = None) -> RewardBreakdown:
    """Compute full reward breakdown for a step."""
    p, pd = compute_profitability_reward(realized_pnl_delta, unrealized_pnl, daily_pnl, max_daily_loss)
    t, td = compute_timing_reward(action_type, deviation_pct, active_seconds, trend, breakeven_pct)
    c, cd = compute_cost_awareness_reward(action_type, used_cost_tools, margin_over_breakeven, called_stt_trap, is_near_expiry)
    f, fd = compute_format_reward(parsed_ok, has_action_type, valid_action)
    total = compute_total_reward(p, t, c, f, weights)
    return RewardBreakdown(
        profitability=p, timing=t, cost_awareness=c, format_compliance=f,
        total=total, details={"profitability": pd, "timing": td, "cost_awareness": cd, "format": fd})
