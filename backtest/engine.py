"""
Backtest engine — replays historical sessions through the agent and records results.
"""
from __future__ import annotations
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from config.settings import get_settings, LOGS_DIR, REPORTS_DIR
from data.feeds.mock_feed import MockFeed
from data.feeds.historical_feed import HistoricalFeed
from data.historical.store import HistoricalStore
from execution.order_simulator import OrderSimulator
from mcp_servers.mcp_client import MCPClient
from pcp_arb_env.environment import PCPArbEnv
from training.rollout import SYSTEM_PROMPT, parse_action

class BacktestEngine:
    """Replays sessions through an agent and records all decisions."""

    def __init__(self, initial_capital: float = 1_000_000):
        self.settings = get_settings()
        self.initial_capital = initial_capital
        self.store = HistoricalStore()
        self.session_results: List[Dict] = []

    def run(self, model=None, tokenizer=None, start_date: date = None,
            end_date: date = None, underlying: str = "NIFTY",
            mode: str = "historical", max_steps_per_session: int = 200) -> Dict:
        """
        Run backtest over a date range.
        
        Args:
            model: Trained LLM model (or None for baseline agent)
            tokenizer: Tokenizer for the model
            start_date: Start date for backtest
            end_date: End date for backtest
            underlying: Underlying to backtest
            mode: "historical" or "mock"
            max_steps_per_session: Max steps per trading session
        
        Returns:
            Summary results dictionary.
        """
        if start_date is None:
            start_date = date(2024, 1, 1)
        if end_date is None:
            end_date = date(2024, 6, 30)

        if mode == "historical":
            available = self.store.list_available_dates(underlying)
            dates = [d for d in available if start_date <= d <= end_date]
            if not dates:
                # Generate synthetic data
                from data.historical.generator import SyntheticGenerator
                gen = SyntheticGenerator()
                current = start_date
                while current <= end_date:
                    if current.weekday() < 5:
                        chains = gen.generate_session(underlying, current)
                        self.store.save_session(current, underlying, chains)
                    current += timedelta(days=1)
                dates = self.store.list_available_dates(underlying)
                dates = [d for d in dates if start_date <= d <= end_date]
        else:
            n_sessions = min(20, (end_date - start_date).days // 7)
            dates = [start_date + timedelta(days=i * 7) for i in range(n_sessions)]

        print(f"[Backtest] Running {len(dates)} sessions for {underlying} ({mode})")
        capital = self.initial_capital
        equity_curve = [capital]

        for i, dt in enumerate(dates):
            print(f"  Session {i+1}/{len(dates)}: {dt}")
            session_result = self._run_session(
                model, tokenizer, underlying, dt, mode, max_steps_per_session)
            capital += session_result["session_pnl"]
            equity_curve.append(capital)
            session_result["cumulative_capital"] = capital
            self.session_results.append(session_result)

        summary = self._compute_summary(equity_curve)
        self._save_results(underlying, start_date, end_date, summary)
        return summary

    def _run_session(self, model, tokenizer, underlying: str, dt: date,
                     mode: str, max_steps: int) -> Dict:
        """Run a single trading session."""
        if mode == "historical":
            feed = HistoricalFeed(underlying, replay_date=dt)
        else:
            feed = MockFeed(underlyings=[underlying])

        mcp_client = MCPClient(timeout=5.0)
        env = PCPArbEnv(feed=feed, mcp_client=mcp_client)
        obs = env.reset()

        session_log = []
        tool_call_count = 0
        tool_sequence = []

        for step in range(max_steps):
            if env.done:
                break
            if model is not None and tokenizer is not None:
                prompt = f"{SYSTEM_PROMPT}\n\nCurrent state:\n{obs}"
                try:
                    import torch
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=1024)
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=120,
                                                  do_sample=True, temperature=0.5,
                                                  pad_token_id=tokenizer.eos_token_id)
                    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
                    action, parsed_ok = parse_action(response)
                except Exception as e:
                    action = {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}
                    parsed_ok = False
                    response = str(e)
            else:
                action, response = self._baseline_action(env, underlying)
                parsed_ok = True

            result = env.step(action)
            tc_count = len(action.get("tool_calls", []))
            tool_call_count += tc_count
            for tc in action.get("tool_calls", []):
                tool_sequence.append(tc.get("tool", "unknown"))

            session_log.append({
                "step": step, "action": action, "reward": result.reward.total,
                "reward_breakdown": result.reward.to_dict(),
                "parsed_ok": parsed_ok, "pnl": result.info.get("daily_pnl", 0)})
            obs = result.observation

        session_pnl = env._daily_pnl
        log_path = LOGS_DIR / f"session_{dt.isoformat()}_{underlying}.jsonl"
        with open(log_path, "w") as f:
            for entry in session_log:
                f.write(json.dumps(entry, default=str) + "\n")

        return {
            "date": dt.isoformat(), "underlying": underlying,
            "session_pnl": round(session_pnl, 2), "steps": len(session_log),
            "tool_calls": tool_call_count, "tool_sequence": tool_sequence[:20],
            "trades": env.order_sim._trade_count if hasattr(env.order_sim, '_trade_count') else 0,
            "log_path": str(log_path)}

    def _baseline_action(self, env: PCPArbEnv, underlying: str) -> tuple:
        """Baseline agent: enter every violation above breakeven, exit when it closes."""
        state = env.state()
        violations = state.get("violations", [])
        positions = state.get("positions", [])

        if positions:
            for pos in positions:
                if pos.get("current_deviation_pct", 0) < 0.1:
                    action = {"action_type": "exit_all", "tool_calls": [], "strike": None, "qty": 1}
                    return action, "baseline_exit"
            return {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}, "baseline_hold"

        for v in violations:
            if v.get("deviation_pct", 0) > 0.3:
                action = {
                    "action_type": "enter_long_call_short_put",
                    "tool_calls": [],
                    "strike": v.get("strike", 0),
                    "qty": 1}
                return action, "baseline_enter"

        return {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}, "baseline_hold"

    def _compute_summary(self, equity_curve: List[float]) -> Dict:
        """Compute backtest summary statistics."""
        import numpy as np
        returns = []
        for i in range(1, len(equity_curve)):
            r = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(r)
        returns = np.array(returns) if returns else np.array([0.0])
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        avg_return = np.mean(returns) * 100
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_dd = 0
        peak = equity_curve[0]
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        win_sessions = sum(1 for r in self.session_results if r["session_pnl"] > 0)
        total_sessions = len(self.session_results)
        return {
            "total_return_pct": round(total_return, 2),
            "avg_session_return_pct": round(avg_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_sessions / max(total_sessions, 1) * 100, 1),
            "total_sessions": total_sessions,
            "winning_sessions": win_sessions,
            "losing_sessions": total_sessions - win_sessions,
            "final_capital": round(equity_curve[-1], 2),
            "total_pnl": round(equity_curve[-1] - equity_curve[0], 2),
            "equity_curve": [round(e, 2) for e in equity_curve]}

    def _save_results(self, underlying: str, start: date, end: date, summary: Dict):
        """Save backtest results to JSON."""
        results_path = REPORTS_DIR / f"backtest_{underlying}_{start}_{end}.json"
        with open(results_path, "w") as f:
            json.dump({"summary": summary, "sessions": self.session_results}, f, indent=2, default=str)
        print(f"[Backtest] Results saved to {results_path}")
