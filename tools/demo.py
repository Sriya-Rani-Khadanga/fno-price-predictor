"""
Interactive CLI demo tool for exploring the PCP arb system.
"""
from __future__ import annotations
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from data.feeds.mock_feed import MockFeed
from mcp_servers.mcp_client import MCPClient
from pcp_arb_env.environment import PCPArbEnv
from training.rollout import parse_action
from config.settings import get_settings

console = Console()

from data.feeds.historical_feed import HistoricalFeed
from datetime import date, datetime

def run_demo():
    """Run interactive CLI demo."""
    console.print(Panel("[bold cyan]PCP Arbitrage RL System — Interactive Demo[/bold cyan]",
                        subtitle="Type 'help' for commands"))
    settings = get_settings()
    feed = HistoricalFeed(underlying="NIFTY", replay_date=date(2024, 4, 24), speed_multiplier=1.0)
    mcp = MCPClient(timeout=5.0)
    env = PCPArbEnv(feed=feed, mcp_client=mcp)
    start_dt = datetime.combine(date(2024, 4, 24), datetime.min.time().replace(hour=10, minute=0))
    obs = env.reset(start_time=start_dt)
    console.print(Panel(obs, title="Initial Observation", border_style="green"))

    while True:
        user_input = Prompt.ask("\n[bold yellow]Action[/bold yellow]", default="hold")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        elif user_input.lower() == "help":
            console.print("[cyan]Commands:[/cyan]")
            console.print("  hold — do nothing")
            console.print("  enter <strike> — enter long call/short put at strike")
            console.print("  exit — exit all positions")
            console.print("  scan — scan for violations")
            console.print("  news — get news summary for today")
            console.print("  rsi — get RSI for NIFTY")
            console.print("  state — show full state")
            console.print("  costs <strike> — calculate arb costs")
            console.print("  stt <strike> — simulate STT trap")
            console.print("  quit — exit demo")
            continue
        elif user_input.lower() == "news":
            state = env.state()
            action = {"action_type": "hold", "strike": None, "qty": 1,
                      "tool_calls": [{"server": "news", "tool": "get_news_summary",
                                      "params": {"date_iso": state.get("session_date", "2024-04-24"), "symbol": "NIFTY"}}]}
        elif user_input.lower() == "rsi":
            action = {"action_type": "hold", "strike": None, "qty": 1,
                      "tool_calls": [{"server": "technical", "tool": "get_rsi",
                                      "params": {"symbol": "NIFTY", "period": 14}}]}
        elif user_input.lower() == "state":
            state = env.state()
            console.print_json(json.dumps(state, default=str, indent=2))
            continue
        elif user_input.lower() == "scan":
            action = {"action_type": "hold", "strike": None, "qty": 1,
                      "tool_calls": [{"server": "market_data", "tool": "get_option_chain",
                                      "params": {"underlying": "NIFTY", "expiry": ""}}]}
        elif user_input.lower().startswith("enter"):
            parts = user_input.split()
            strike = float(parts[1]) if len(parts) > 1 else 22000
            action = {"action_type": "enter_long_call_short_put", "strike": strike, "qty": 1,
                      "tool_calls": [{"server": "cost", "tool": "calculate_arb_costs",
                                      "params": {"underlying": "NIFTY", "strike": strike,
                                                 "expiry_days": 15, "qty": 1, "gross_violation_pct": 0.5}}]}
        elif user_input.lower() == "exit":
            action = {"action_type": "exit_all", "strike": None, "qty": 1, "tool_calls": []}
        elif user_input.lower().startswith("costs"):
            parts = user_input.split()
            strike = float(parts[1]) if len(parts) > 1 else 22000
            action = {"action_type": "hold", "strike": None, "qty": 1,
                      "tool_calls": [{"server": "cost", "tool": "calculate_arb_costs",
                                      "params": {"underlying": "NIFTY", "strike": strike,
                                                 "expiry_days": 15, "qty": 1, "gross_violation_pct": 0.5}}]}
        elif user_input.lower().startswith("stt"):
            parts = user_input.split()
            strike = float(parts[1]) if len(parts) > 1 else 22000
            action = {"action_type": "hold", "strike": None, "qty": 1,
                      "tool_calls": [{"server": "cost", "tool": "simulate_stt_trap",
                                      "params": {"underlying": "NIFTY", "strike": strike,
                                                 "expiry_days": 3, "qty": 1, "hold_to_expiry": True}}]}
        else:
            try:
                action, _ = parse_action(user_input)
            except Exception:
                action = {"action_type": "hold", "strike": None, "qty": 1, "tool_calls": []}

        result = env.step(action)
        reward_table = Table(title="Reward Breakdown")
        reward_table.add_column("Component", style="cyan")
        reward_table.add_column("Value", style="green")
        reward_table.add_column("Detail", style="dim")
        rb = result.reward
        for comp, val, detail in [
            ("Profitability", rb.profitability, rb.details.get("profitability", "")),
            ("Timing", rb.timing, rb.details.get("timing", "")),
            ("Cost Awareness", rb.cost_awareness, rb.details.get("cost_awareness", "")),
            ("Format", rb.format_compliance, rb.details.get("format", "")),
            ("TOTAL", rb.total, "")]:
            color = "green" if val > 0 else "red" if val < 0 else "white"
            reward_table.add_row(comp, f"[{color}]{val:+.3f}[/{color}]", detail)
        console.print(reward_table)
        console.print(Panel(result.observation, title=f"Step {result.info['step']}", border_style="blue"))
        if result.done:
            console.print("[bold red]Session ended![/bold red]")
            final_pnl = result.info.get("daily_pnl", 0)
            console.print(f"Final P&L: ₹{final_pnl:,.0f}")
            break
    console.print("[bold cyan]Demo complete.[/bold cyan]")

if __name__ == "__main__":
    run_demo()
