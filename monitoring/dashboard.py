"""
Rich-based live terminal dashboard for monitoring the PCP arb system.
"""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from config.settings import get_settings, LOGS_DIR

class Dashboard:
    """Rich-based live terminal dashboard with 6 panels."""

    def __init__(self):
        self.console = Console()
        self.settings = get_settings()
        self._market_state: Dict = {}
        self._agent_state: Dict = {}
        self._positions: List[Dict] = []
        self._session_pnl: float = 0.0
        self._equity_points: List[float] = []
        self._training_metrics: Dict = {}
        self._feed_health: Dict = {}
        self._step_count = 0
        self._log_file: Optional[Path] = None
        self._log_entries: List[Dict] = []

    def start_logging(self, session_id: str = None):
        """Start tick-by-tick logging to JSONL file."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = LOGS_DIR / f"session_{session_id}.jsonl"

    def update(self, market: Dict = None, agent: Dict = None, positions: List[Dict] = None,
               pnl: float = None, training: Dict = None, feed_health: Dict = None,
               step: int = None):
        """Update dashboard state."""
        if market: self._market_state = market
        if agent: self._agent_state = agent
        if positions is not None: self._positions = positions
        if pnl is not None:
            self._session_pnl = pnl
            self._equity_points.append(pnl)
        if training: self._training_metrics = training
        if feed_health: self._feed_health = feed_health
        if step is not None: self._step_count = step
        entry = {
            "timestamp": datetime.now().isoformat(), "step": self._step_count,
            "pnl": self._session_pnl, "positions": len(self._positions),
            "market": self._market_state, "agent": self._agent_state}
        self._log_entries.append(entry)
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def _build_market_panel(self) -> Panel:
        """Panel 1 — Market state with PCP deviation table."""
        table = Table(title="PCP Deviations", show_header=True, header_style="bold cyan")
        table.add_column("Underlying", style="white")
        table.add_column("Strike", justify="right")
        table.add_column("Dev %", justify="right")
        table.add_column("Active", justify="right")
        table.add_column("Trend")
        violations = self._market_state.get("violations", [])
        for v in violations[:8]:
            dev = v.get("deviation_pct", 0)
            color = "green" if dev < 0.2 else "yellow" if dev < 0.5 else "red"
            trend_arrow = "↑" if v.get("trend") == "widening" else "↓" if v.get("trend") == "narrowing" else "→"
            table.add_row(
                v.get("underlying", "?"), f"{v.get('strike', 0):.0f}",
                f"[{color}]{dev:.2f}%[/{color}]",
                f"{v.get('active_seconds', 0):.0f}s", trend_arrow)
        if not violations:
            table.add_row("—", "—", "[dim]No violations[/dim]", "—", "—")
        return Panel(table, title="📊 Market State", border_style="blue")

    def _build_agent_panel(self) -> Panel:
        """Panel 2 — Agent reasoning."""
        lines = []
        tool_calls = self._agent_state.get("tool_calls", [])
        if tool_calls:
            lines.append("[cyan]Active Tools:[/cyan]")
            for tc in tool_calls[-2:]:
                lines.append(f"  {tc.get('tool', '?')}")
        
        # New Context Section
        action = self._agent_state.get("action", {})
        if action:
            lines.append(f"\n[bold yellow]DECISION: {action.get('action_type', 'hold').upper()}[/bold yellow]")
            if action.get("strike"):
                lines.append(f"Strike: {action.get('strike')}")

        # Show Technicals if available
        market = self._market_state
        if market.get("rsi"):
            lines.append(f"[blue]Technicals: RSI {market['rsi']} | EMA {market.get('ema')}[/blue]")
        
        reward = self._agent_state.get("reward_breakdown", {})
        if reward:
            lines.append("\n[cyan]Reward Signal:[/cyan]")
            val = reward.get("total", 0)
            color = "green" if val > 0 else "red" if val < 0 else "white"
            lines.append(f"  Total Score: [{color}]{val:+.3f}[/{color}]")

        return Panel("\n".join(lines) if lines else "[dim]Waiting for agent...[/dim]",
                     title="🧠 Multi-Factor Logic", border_style="yellow")

    def _build_positions_panel(self) -> Panel:
        """Panel 3 — Open positions."""
        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", style="dim")
        table.add_column("Strike", justify="right")
        table.add_column("Entry Dev", justify="right")
        table.add_column("Curr Dev", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Time", justify="right")
        for pos in self._positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_color = "green" if pnl > 0 else "red"
            table.add_row(
                pos.get("position_id", "?")[:8],
                f"{pos.get('strike', 0):.0f}",
                f"{pos.get('entry_deviation_pct', 0):.2f}%",
                f"{pos.get('current_deviation_pct', 0):.2f}%",
                f"[{pnl_color}]₹{pnl:+,.0f}[/{pnl_color}]",
                f"{pos.get('time_in_position', 0):.0f}s")
        if not self._positions:
            table.add_row("—", "—", "—", "—", "[dim]No positions[/dim]", "—")
        return Panel(table, title="📈 Positions", border_style="green")

    def _build_pnl_panel(self) -> Panel:
        """Panel 4 — Session P&L."""
        pnl_color = "green" if self._session_pnl >= 0 else "red"
        sparkline = ""
        if self._equity_points:
            mini = min(self._equity_points)
            maxi = max(self._equity_points) if max(self._equity_points) != mini else mini + 1
            chars = "▁▂▃▄▅▆▇█"
            for p in self._equity_points[-40:]:
                idx = int((p - mini) / (maxi - mini) * 7)
                sparkline += chars[max(0, min(7, idx))]
        lines = [
            f"[{pnl_color}]₹{self._session_pnl:+,.0f}[/{pnl_color}]",
            f"Equity: {sparkline}" if sparkline else "",
            f"Step: {self._step_count}"]
        return Panel("\n".join(lines), title="💰 Session P&L", border_style=pnl_color)

    def _build_training_panel(self) -> Panel:
        """Panel 5 — Training metrics."""
        lines = []
        if self._training_metrics:
            lines.append(f"Stage: [cyan]{self._training_metrics.get('stage', '?')}[/cyan]")
            lines.append(f"Episode: {self._training_metrics.get('episode', 0)}")
            lines.append(f"Avg Reward: {self._training_metrics.get('avg_reward', 0):.3f}")
            lines.append(f"Parse Fail: {self._training_metrics.get('parse_fail_rate', 0):.1%}")
            lines.append(f"Best P&L: ₹{self._training_metrics.get('best_pnl', 0):,.0f}")
        else:
            lines.append("[dim]Not in training mode[/dim]")
        return Panel("\n".join(lines), title="🎓 Training", border_style="magenta")

    def _build_health_panel(self) -> Panel:
        """Panel 6 — Data feed health."""
        lines = []
        for server, healthy in self._feed_health.items():
            icon = "🟢" if healthy else "🔴"
            lines.append(f"{icon} {server}")
        if not lines:
            lines = ["[dim]No health data[/dim]"]
        return Panel("\n".join(lines), title="🏥 Feed Health", border_style="cyan")

    def render(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="middle", size=10),
            Layout(name="bottom", size=8))
        layout["top"].split_row(
            Layout(self._build_market_panel(), name="market"),
            Layout(self._build_agent_panel(), name="agent"))
        layout["middle"].split_row(
            Layout(self._build_positions_panel(), name="positions"),
            Layout(self._build_pnl_panel(), name="pnl"))
        layout["bottom"].split_row(
            Layout(self._build_training_panel(), name="training"),
            Layout(self._build_health_panel(), name="health"))
        return layout

    def run_live(self, env=None, refresh_rate: float = 1.0):
        """Run the dashboard with live refresh."""
        with Live(self.render(), console=self.console, refresh_per_second=1.0/refresh_rate) as live:
            try:
                while True:
                    live.update(self.render())
                    time.sleep(refresh_rate)
            except KeyboardInterrupt:
                pass
        if self._log_file:
            print(f"[Dashboard] Session log saved to {self._log_file}")
