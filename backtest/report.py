"""
Backtest report generator — creates HTML reports from backtest results.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Dict
from jinja2 import Template
from config.settings import REPORTS_DIR

REPORT_TEMPLATE = """<!DOCTYPE html>
<html><head><title>PCP Arb Backtest Report — {{ underlying }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}
  .container{max-width:1200px;margin:0 auto}
  h1{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:10px}
  h2{color:#79c0ff}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:15px 0}
  .metric{display:inline-block;margin:10px 20px;text-align:center}
  .metric .value{font-size:28px;font-weight:bold;color:#58a6ff}
  .metric .label{font-size:12px;color:#8b949e}
  .positive{color:#3fb950} .negative{color:#f85149}
  table{width:100%;border-collapse:collapse;margin:10px 0}
  th,td{padding:8px 12px;text-align:right;border-bottom:1px solid #21262d}
  th{color:#8b949e;font-weight:normal}
  .chart-container{position:relative;height:300px;margin:20px 0}
</style></head><body><div class="container">
<h1>🔄 PCP Arbitrage Backtest Report</h1>
<p>{{ underlying }} | {{ start_date }} to {{ end_date }}</p>
<div class="card">
  <h2>Summary</h2>
  <div class="metric"><div class="value {% if total_pnl >= 0 %}positive{% else %}negative{% endif %}">₹{{ "{:,.0f}".format(total_pnl) }}</div><div class="label">Total P&L</div></div>
  <div class="metric"><div class="value">{{ sharpe_ratio }}</div><div class="label">Sharpe Ratio</div></div>
  <div class="metric"><div class="value">{{ win_rate_pct }}%</div><div class="label">Win Rate</div></div>
  <div class="metric"><div class="value negative">{{ max_drawdown_pct }}%</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{{ total_sessions }}</div><div class="label">Sessions</div></div>
</div>
<div class="card"><h2>Equity Curve</h2><div class="chart-container"><canvas id="equityChart"></canvas></div></div>
<div class="card"><h2>Session Results</h2>
<table><tr><th>Date</th><th>P&L</th><th>Steps</th><th>Tool Calls</th><th>Trades</th></tr>
{% for s in sessions %}
<tr><td>{{ s.date }}</td><td class="{% if s.session_pnl >= 0 %}positive{% else %}negative{% endif %}">₹{{ "{:,.0f}".format(s.session_pnl) }}</td>
<td>{{ s.steps }}</td><td>{{ s.tool_calls }}</td><td>{{ s.trades }}</td></tr>
{% endfor %}
</table></div>
</div>
<script>
new Chart(document.getElementById('equityChart'),{type:'line',data:{
  labels:{{ equity_labels | tojson }},
  datasets:[{label:'Equity',data:{{ equity_data | tojson }},
  borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.1)',fill:true,tension:0.3}]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
  scales:{y:{grid:{color:'#21262d'},ticks:{color:'#8b949e'}},x:{grid:{color:'#21262d'},ticks:{color:'#8b949e'}}}}});
</script></body></html>"""

def generate_report(results: Dict, underlying: str, start: date, end: date) -> str:
    """Generate HTML backtest report. Returns path to generated file."""
    summary = results.get("summary", results)
    sessions = results.get("sessions", [])
    equity = summary.get("equity_curve", [1000000])
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        underlying=underlying, start_date=start.isoformat(), end_date=end.isoformat(),
        total_pnl=summary.get("total_pnl", 0), sharpe_ratio=summary.get("sharpe_ratio", 0),
        win_rate_pct=summary.get("win_rate_pct", 0), max_drawdown_pct=summary.get("max_drawdown_pct", 0),
        total_sessions=summary.get("total_sessions", 0), sessions=sessions,
        equity_labels=list(range(len(equity))), equity_data=equity,
        total_return_pct=summary.get("total_return_pct", 0))
    path = REPORTS_DIR / f"backtest_{underlying}_{start}_{end}.html"
    path.write_text(html, encoding="utf-8")
    print(f"[Report] HTML report saved to {path}")
    return str(path)
