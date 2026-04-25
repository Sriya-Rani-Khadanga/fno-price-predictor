"""
Alert system for monitoring the PCP arb trading system.
"""
from __future__ import annotations
from datetime import datetime
from typing import Callable, Dict, List, Optional
from rich.console import Console

console = Console()

class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class Alert:
    def __init__(self, level: str, message: str, source: str, timestamp: datetime = None):
        self.level = level
        self.message = message
        self.source = source
        self.timestamp = timestamp or datetime.now()

    def __str__(self):
        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        return f"{icons.get(self.level, '?')} [{self.level.upper()}] {self.source}: {self.message}"

class AlertManager:
    """Manages alerts for the trading system."""

    def __init__(self):
        self._alerts: List[Alert] = []
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, level: str, message: str, source: str):
        alert = Alert(level, message, source)
        self._alerts.append(alert)
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                pass
        color = {"info": "blue", "warning": "yellow", "critical": "red"}.get(level, "white")
        console.print(f"[{color}]{alert}[/{color}]")

    def check_staleness(self, staleness: Dict[str, float], threshold: float = 10.0):
        for sym, age in staleness.items():
            if age > threshold:
                self.emit(AlertLevel.WARNING, f"Data stale by {age:.0f}s", f"feed.{sym}")

    def check_daily_pnl(self, pnl: float, limit: float):
        if pnl < -limit * 0.8:
            self.emit(AlertLevel.CRITICAL, f"Daily P&L ₹{pnl:,.0f} near limit ₹{-limit:,.0f}", "risk")
        elif pnl < -limit * 0.5:
            self.emit(AlertLevel.WARNING, f"Daily P&L ₹{pnl:,.0f} past 50% of limit", "risk")

    def check_server_health(self, health: Dict[str, bool]):
        for server, ok in health.items():
            if not ok:
                self.emit(AlertLevel.CRITICAL, f"MCP server {server} is DOWN", "mcp")

    def recent(self, n: int = 10) -> List[Alert]:
        return self._alerts[-n:]
