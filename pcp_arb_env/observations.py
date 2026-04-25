"""
Observation builder — creates text observations for the LLM agent.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional

def build_text_observation(session_date: str, session_time: str, minutes_to_close: int, daily_pnl: float,
                            positions_count: int, max_positions: int,
                            available_tools: List[str], last_tool_results: Dict,
                            positions_info: List[Dict], violations: List[Dict],
                            risk_utilization: Dict) -> str:
    """Build natural language observation for the LLM agent."""
    lines = []
    pnl_str = f"+₹{daily_pnl:,.0f}" if daily_pnl >= 0 else f"-₹{abs(daily_pnl):,.0f}"
    lines.append(f"Date: {session_date} | Session: {session_time} | {minutes_to_close} min to close | Daily P&L: ₹{pnl_str} | Positions: {positions_count}/{max_positions}")
    lines.append("")
    tool_names = ", ".join(available_tools[:8])
    if len(available_tools) > 8:
        tool_names += f" (+{len(available_tools) - 8} more)"
    lines.append(f"Available tools: {tool_names}")
    lines.append("")
    if last_tool_results:
        lines.append("Last tool results:")
        for tool_name, result in last_tool_results.items():
            if isinstance(result, dict):
                if "error" in result:
                    lines.append(f"  {tool_name}: ERROR - {result['error']}")
                elif "deviation_pct" in result:
                    dev = result.get("deviation_pct", 0)
                    trend = result.get("trend", "stable")
                    conf = result.get("confidence", 0)
                    lines.append(f"  {tool_name}: deviation {dev:.2f}%, {result.get('active_seconds', 0):.0f}s active, {trend}, confidence {conf:.2f}")
                elif "allowed" in result:
                    lines.append(f"  {tool_name}: allowed={result['allowed']}, {result.get('reason', '')}")
                elif "net_pnl" in result:
                    lines.append(f"  {tool_name}: net P&L ₹{result['net_pnl']:,.0f}, profitable={result.get('is_profitable', False)}")
                elif "breakeven_pct" in result:
                    lines.append(f"  {tool_name}: breakeven {result['breakeven_pct']:.2f}%, margin {result.get('margin_over_breakeven_pct', 0):.2f}%")
                elif "is_trap" in result:
                    trap = "YES" if result["is_trap"] else "NO"
                    lines.append(f"  {tool_name}: STT trap={trap}, magnitude ₹{result.get('trap_magnitude', 0):,.0f}, {result.get('recommendation', '')}")
                elif "regime" in result:
                    lines.append(f"  {tool_name}: regime={result['regime']}, VIX proxy={result.get('vix_proxy', 0):.1f}")
                else:
                    summary = str(result)[:150]
                    lines.append(f"  {tool_name}: {summary}")
            else:
                lines.append(f"  {tool_name}: {str(result)[:150]}")
        lines.append("")
    if positions_info:
        for pos in positions_info:
            action = pos.get("action_type", "unknown")
            strike = pos.get("strike", 0)
            entry_dev = pos.get("entry_deviation_pct", 0)
            curr_dev = pos.get("current_deviation_pct", 0)
            upnl = pos.get("unrealized_pnl", 0)
            time_s = pos.get("time_in_position", 0)
            pnl_sign = "+" if upnl >= 0 else ""
            lines.append(f"Open: {action} @ {strike:.0f}, entry {entry_dev:.2f}% → now {curr_dev:.2f}%. "
                         f"Unrealized: ₹{pnl_sign}{upnl:,.0f}. {time_s:.0f}s held.")
    elif positions_count == 0:
        lines.append("No open positions.")
    lines.append("")
    if violations:
        lines.append("Active violations:")
        for v in violations[:5]:
            lines.append(f"  {v.get('underlying', '?')} {v.get('strike', 0):.0f}: "
                         f"{v.get('deviation_pct', 0):.2f}%, {v.get('trend', 'stable')}, "
                         f"{v.get('active_seconds', 0):.0f}s active")
        lines.append("")
    lines.append('Action: output JSON with optional tool_calls and trade action.')
    lines.append('Format: {"tool_calls": [{"server": "...", "tool": "...", "params": {...}}], '
                 '"action_type": "hold|enter_long_call_short_put|enter_short_call_long_put|exit_all|exit_strike", '
                 '"strike": null, "qty": 1}')
    return "\n".join(lines)
