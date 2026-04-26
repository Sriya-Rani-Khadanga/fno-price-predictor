"""
PCPArbEnv — OpenEnv environment for put-call parity arbitrage RL training.
The agent actively calls MCP tools to gather market intelligence, then decides.
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from data.feeds.base import BaseFeed, TickData
from data.processors.pcp_calculator import PCPCalculator
from execution.order_simulator import OrderSimulator
from execution.risk import RiskManager
from mcp_servers.mcp_client import MCPClient
from pcp_arb_env.rewards import compute_reward, RewardBreakdown
from pcp_arb_env.observations import build_text_observation
from pcp_arb_env.curriculum import CurriculumManager
from config.settings import get_settings, CurriculumConfig, RiskConfig

VALID_ACTIONS = {"hold", "enter_long_call_short_put", "enter_short_call_long_put",
                 "exit_all", "exit_strike"}

@dataclass
class StepResult:
    observation: str
    reward: RewardBreakdown
    done: bool
    info: Dict[str, Any]

class PCPArbEnv:
    """OpenEnv-compatible environment for PCP arbitrage RL training."""

    def __init__(self, feed: BaseFeed, mcp_client: MCPClient,
                 curriculum_config: CurriculumConfig = None,
                 risk_config: RiskConfig = None):
        self.feed = feed
        self.mcp = mcp_client
        settings = get_settings()
        self.curriculum = CurriculumManager(curriculum_config or settings.curriculum)
        self.risk_mgr = RiskManager(risk_config or settings.risk)
        self.order_sim = OrderSimulator()
        self.pcp_calc = PCPCalculator({s: i.lot_size for s, i in settings.instruments.items()})
        self.settings = settings
        self._tick: Optional[TickData] = None
        self._daily_pnl = 0.0
        self._step_count = 0
        self._session_start: Optional[datetime] = None
        self._last_tool_results: Dict[str, Any] = {}
        self._tool_call_log: List[Dict] = []
        self._action_log: List[Dict] = []
        self._violations_detected: List[Dict] = []
        self._done = False

    def reset(self, start_time: Optional[datetime] = None) -> str:
        """Start new session, clear positions, reset P&L."""
        self.order_sim.reset()
        self.risk_mgr.reset()
        self.pcp_calc.reset()
        self._daily_pnl = 0.0
        self._step_count = 0
        self._last_tool_results = {}
        self._tool_call_log = []
        self._action_log = []
        self._violations_detected = []
        self._done = False
        try:
            self._tick = self.feed.reset(start_time=start_time)
        except TypeError:
            self._tick = self.feed.reset()
        self._session_start = self._tick.timestamp
        if self.curriculum.should_fast_forward():
            min_pct = self.curriculum.get_violation_range()[0]
            self._tick = self.feed.fast_forward_to_violation(min_pct)
        self._push_tick_to_mcp()
        self.mcp.call_internal("risk", "reset", {"daily_limit": self.settings.risk.max_daily_loss})
        return self.get_text_observation()

    def step(self, action: Dict) -> StepResult:
        """Execute one step: tool calls + trade action + advance feed."""
        self._step_count += 1
        tool_results = {}
        tool_calls = action.get("tool_calls", [])
        for tc in tool_calls[:5]:
            server = tc.get("server", "market_data")
            tool = tc.get("tool", "")
            params = tc.get("params", {})
            result = self.mcp.call_tool(server, tool, params)
            tool_results[f"{tool}({', '.join(f'{k}={v}' for k, v in params.items())})"] = result
            self._tool_call_log.append({
                "step": self._step_count, "server": server, "tool": tool,
                "params": params, "timestamp": datetime.now().isoformat()})
        self._last_tool_results = tool_results
        action_type = action.get("action_type", "hold")
        strike = action.get("strike")
        qty = action.get("qty", 1)
        parsed_ok = True
        has_action_type = "action_type" in action
        valid_action = action_type in VALID_ACTIONS
        realized_delta = 0.0
        deviation_pct = 0.0
        active_seconds = 0.0
        trend = "stable"
        breakeven_pct = 0.3
        margin_over_be = 0.0
        used_cost = any("cost" in tc.get("tool", "") or "breakeven" in tc.get("tool", "") or "stt" in tc.get("tool", "")
                        for tc in tool_calls)
        called_stt = any("stt_trap" in tc.get("tool", "") for tc in tool_calls)
        is_near_expiry = False
        if valid_action and action_type.startswith("enter") and strike is not None:
            for sym, chain in self._tick.chains.items():
                sd = chain.get_strike(strike)
                if sd is not None:
                    fill = self.order_sim.execute_entry(chain, strike, qty, action_type, sd.pcp_deviation_pct)
                    if fill:
                        inst = self.settings.instruments.get(sym)
                        lot = inst.lot_size if inst else 50
                        self.mcp.call_internal("risk", "add_position", {
                            "position_id": f"pos_{fill.order_id}",
                            "underlying": sym, "strike": strike, "qty": fill.qty_filled,
                            "action_type": action_type,
                            "entry_price_call": fill.fill_price_call,
                            "entry_price_put": fill.fill_price_put,
                            "entry_deviation_pct": sd.pcp_deviation_pct,
                            "lot_size": lot})
                        deviation_pct = sd.pcp_deviation_pct
                    break
        elif action_type == "exit_all":
            for chain in self._tick.chains.values():
                results = self.order_sim.exit_all(chain)
                for r in results:
                    realized_delta += r["net_pnl"]
                    self.mcp.call_internal("risk", "close_position",
                                           {"position_id": r["position_id"], "exit_pnl": r["net_pnl"]})
        elif action_type == "exit_strike" and strike is not None:
            for pid, pos in list(self.order_sim.positions.items()):
                if abs(pos.strike - strike) < 0.01:
                    for chain in self._tick.chains.values():
                        result = self.order_sim.execute_exit(pid, chain)
                        if result:
                            realized_delta += result["net_pnl"]
                            self.mcp.call_internal("risk", "close_position",
                                                   {"position_id": pid, "exit_pnl": result["net_pnl"]})
                        break
        self._daily_pnl += realized_delta
        self._tick = self.feed.next_tick()
        self._push_tick_to_mcp()
        self._update_violations()
        unrealized = sum(pos.entry_fill.fill_price_call for pos in self.order_sim.positions.values()) * 0
        for pos in self.order_sim.positions.values():
            for chain in self._tick.chains.values():
                sd = chain.get_strike(pos.strike)
                if sd:
                    if "long_call" in pos.action_type:
                        unrealized += (sd.call_mid - pos.entry_fill.fill_price_call) * pos.qty * pos.lot_size
                        unrealized += (pos.entry_fill.fill_price_put - sd.put_mid) * pos.qty * pos.lot_size
        done = self._check_done()
        reward = compute_reward(
            action_type=action_type, realized_pnl_delta=realized_delta,
            unrealized_pnl=unrealized, daily_pnl=self._daily_pnl,
            max_daily_loss=self.settings.risk.max_daily_loss,
            deviation_pct=deviation_pct, active_seconds=active_seconds,
            trend=trend, breakeven_pct=breakeven_pct,
            used_cost_tools=used_cost, margin_over_breakeven=margin_over_be,
            called_stt_trap=called_stt, is_near_expiry=is_near_expiry,
            parsed_ok=parsed_ok, has_action_type=has_action_type, valid_action=valid_action)
        self._action_log.append({
            "step": self._step_count, "action": action, "reward": reward.total,
            "daily_pnl": self._daily_pnl, "positions": self.order_sim.open_position_count,
            "timestamp": datetime.now().isoformat()})
        obs = self.get_text_observation()
        return StepResult(observation=obs, reward=reward, done=done,
                          info={"step": self._step_count, "daily_pnl": self._daily_pnl,
                                "positions": self.order_sim.open_position_count,
                                "tool_calls_made": len(tool_calls),
                                "reward_breakdown": reward.to_dict()})

    def state(self) -> Dict:
        """Return current environment state."""
        if self._tick is None:
            return {}
        
        # Calculate minutes until 15:30 close
        market_close = self._tick.timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
        minutes_to_close = max(0, int((market_close - self._tick.timestamp).total_seconds() / 60))
        
        positions_info = []
        for pid, pos in self.order_sim.positions.items():
            positions_info.append({
                "position_id": pid, "underlying": pos.underlying,
                "strike": pos.strike, "qty": pos.qty,
                "action_type": pos.action_type,
                "entry_deviation_pct": pos.entry_deviation_pct,
                "current_deviation_pct": 0.0,
                "unrealized_pnl": 0.0,
                "time_in_position": (datetime.now() - pos.entry_time).total_seconds()})
        return {
            "session_date": self._tick.timestamp.date().isoformat(),
            "session_time": self._tick.timestamp.strftime("%H:%M:%S"),
            "minutes_to_close": minutes_to_close,
            "daily_pnl": self._daily_pnl,
            "positions": positions_info,
            "positions_count": self.order_sim.open_position_count,
            "max_positions": self.settings.risk.max_positions,
            "last_tool_results": self._last_tool_results,
            "available_tools": self.mcp.get_tool_names(),
            "violations": self._violations_detected[-5:],
            "step": self._step_count,
            "curriculum_stage": self.curriculum.stage_name,
            "tick_count": self.feed.tick_count}

    def get_text_observation(self) -> str:
        """Build text observation from current state."""
        s = self.state()
        return build_text_observation(
            session_date=s.get("session_date", "2024-01-01"),
            session_time=s.get("session_time", "00:00:00"),
            minutes_to_close=s.get("minutes_to_close", 0),
            daily_pnl=s.get("daily_pnl", 0),
            positions_count=s.get("positions_count", 0),
            max_positions=s.get("max_positions", 5),
            available_tools=s.get("available_tools", []),
            last_tool_results=s.get("last_tool_results", {}),
            positions_info=s.get("positions", []),
            violations=s.get("violations", []),
            risk_utilization={})

    def _push_tick_to_mcp(self):
        """Push current tick data to MCP market data server."""
        if self._tick is None:
            return
        for sym, chain in self._tick.chains.items():
            try:
                self.mcp.push_feed_update(chain.to_dict())
            except Exception:
                pass

    def _update_violations(self):
        """Detect and track violations in current tick."""
        if self._tick is None:
            return
        T = 15.0 / 365.0
        for sym, chain in self._tick.chains.items():
            violations = self.pcp_calc.get_active_violations(chain, T, min_pct=0.1)
            for v in violations:
                self._violations_detected.append(v.to_dict())
        if len(self._violations_detected) > 100:
            self._violations_detected = self._violations_detected[-50:]

    def _check_done(self) -> bool:
        """Check termination conditions."""
        if self.feed.is_done():
            self._done = True
            return True
        if self._daily_pnl < -self.settings.risk.max_daily_loss:
            self._done = True
            return True
        if self._tick and self._tick.timestamp.hour >= 15 and self._tick.timestamp.minute >= 20:
            if self.order_sim.open_position_count == 0:
                self._done = True
                return True
        return False

    @property
    def done(self) -> bool:
        return self._done
