"""
Step recorder and analyzer — detailed breakdown of every step in a session.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template
from config.settings import REPORTS_DIR, RECORDINGS_DIR

class StepAnalyzer:
    """Produces detailed breakdown for every step in a session."""

    def __init__(self):
        self._steps: List[Dict] = []

    def record_step(self, step: int, observation: str, raw_output: str,
                     parsed_action: Dict, tool_calls: List[Dict], tool_results: Dict,
                     reward_breakdown: Dict, position_delta: int,
                     cumulative_pnl: float, timestamp: datetime):
        """Record a single step with full context."""
        self._steps.append({
            "step": step, "timestamp": timestamp.isoformat(),
            "observation": observation[:500], "raw_output": raw_output[:300],
            "parsed_action": parsed_action,
            "tool_calls": tool_calls, "tool_results": tool_results,
            "reward_breakdown": reward_breakdown,
            "position_delta": position_delta,
            "cumulative_pnl": round(cumulative_pnl, 2)})

    def generate_step_report(self, session_id: str = None) -> str:
        """Generate scrollable HTML timeline of every step."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        steps_html = ""
        for s in self._steps:
            action_type = s["parsed_action"].get("action_type", "hold")
            reward = s["reward_breakdown"]
            total_r = reward.get("total", 0)
            color = "#3fb950" if total_r > 0 else "#f85149" if total_r < 0 else "#8b949e"
            
            # Extract technical/news context from tool results
            tech_info = ""
            news_info = ""
            for tool_call, res in s.get("tool_results", {}).items():
                if "get_rsi" in tool_call:
                    tech_info += f" RSI: {res.get('rsi', 'N/A')} ({res.get('status', '')})"
                if "get_news" in tool_call:
                    news_info += f" Sentiment: {res.get('avg_sentiment', 'N/A')} Impact: {res.get('market_impact', '')}"

            details = json.dumps(s["reward_breakdown"].get("details", {}), indent=2)
            steps_html += f"""
            <div class="step-card" style="border-left: 3px solid {color}">
              <div class="step-header">
                <span class="step-num">Step {s['step']}</span>
                <span class="step-time">{s['timestamp']}</span>
                <span class="step-action" style="color:{color}">{action_type}</span>
                <span class="step-reward" style="color:{color}">R: {total_r:.3f}</span>
                <span class="step-pnl">P&L: ₹{s['cumulative_pnl']:,.0f}</span>
              </div>
              <div class="analysis-box">
                <span class="tech-tag">{tech_info or "No Tech Data"}</span>
                <span class="news-tag">{news_info or "No News Data"}</span>
              </div>
              <details>
                <summary>Reasoning & Tool Results</summary>
                <div class="step-details">
                  <p><b>Tool Results:</b></p><pre>{json.dumps(s['tool_results'], indent=2)}</pre>
                  <p><b>Reward Logic:</b></p><pre>{details}</pre>
                </div>
              </details>
            </div>"""

        html = f"""<!DOCTYPE html>
<html><head><title>Step Analysis — {session_id}</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{color:#58a6ff}} .step-card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;margin:8px 0}}
.step-header{{display:flex;gap:20px;align-items:center;flex-wrap:wrap}}
.analysis-box{{margin-top:8px; display:flex; gap:10px;}}
.tech-tag{{background:#21262d; color:#79c0ff; padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid #30363d}}
.news-tag{{background:#21262d; color:#aff5b4; padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid #30363d}}
.step-num{{color:#79c0ff;font-weight:bold}} .step-time{{color:#8b949e;font-size:12px}}
.step-action{{font-weight:bold; font-size:18px}} .step-reward{{font-weight:bold}} .step-pnl{{color:#c9d1d9}}
details{{margin-top:8px}} summary{{cursor:pointer;color:#58a6ff; font-size:13px}}
.step-details{{padding:10px;background:#0d1117;border-radius:4px;margin-top:5px}}
pre{{color:#c9d1d9;font-size:11px;overflow-x:auto}} code{{color:#79c0ff;font-size:11px}}
</style></head><body>
<h1>📋 Multi-Factor Step Analysis — {session_id}</h1>
<p>{len(self._steps)} steps recorded</p>
{steps_html}
</body></html>"""
        path = REPORTS_DIR / f"step_analysis_{session_id}.html"
        path.write_text(html, encoding="utf-8")
        return str(path)

    def record_before_after(self, session_log_untrained: List[Dict],
                             session_log_trained: List[Dict],
                             underlying: str, dt: str) -> str:
        """Generate side-by-side comparison of untrained vs trained agent."""
        rows = ""
        max_steps = max(len(session_log_untrained), len(session_log_trained))
        for i in range(min(max_steps, 50)):
            u = session_log_untrained[i] if i < len(session_log_untrained) else {}
            t = session_log_trained[i] if i < len(session_log_trained) else {}
            u_action = u.get("action", {}).get("action_type", "-")
            t_action = t.get("action", {}).get("action_type", "-")
            u_reward = u.get("reward", 0)
            t_reward = t.get("reward", 0)
            highlight = ""
            if t_reward > u_reward + 0.1:
                highlight = "background:#0d2818;"
            elif u_reward > t_reward + 0.1:
                highlight = "background:#2d1117;"
            rows += f"""<tr style="{highlight}">
                <td>{i}</td><td>{u_action}</td><td>{u_reward:.3f}</td>
                <td>{t_action}</td><td>{t_reward:.3f}</td>
                <td>{t_reward - u_reward:+.3f}</td></tr>"""
        html = f"""<!DOCTYPE html>
<html><head><title>Before/After — {underlying} {dt}</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{color:#58a6ff}} table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #21262d}} th{{color:#8b949e}}
.positive{{color:#3fb950}} .negative{{color:#f85149}}
</style></head><body>
<h1>🔄 Before/After Comparison — {underlying} {dt}</h1>
<table><tr><th>Step</th><th>Untrained Action</th><th>Untrained R</th>
<th>Trained Action</th><th>Trained R</th><th>Δ</th></tr>{rows}</table>
</body></html>"""
        path = RECORDINGS_DIR / "comparisons" / f"{underlying}_{dt}_comparison.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return str(path)

    def clear(self):
        self._steps.clear()
