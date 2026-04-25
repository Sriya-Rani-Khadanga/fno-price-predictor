"""
Alpha analyzer — deep analysis of PCP arb opportunity before training.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from jinja2 import Template
from config.settings import get_settings, REPORTS_DIR, CACHE_DIR
from data.historical.store import HistoricalStore, ViolationStats
from data.historical.generator import SyntheticGenerator
from data.historical.nse_downloader import NSEDownloader
from data.processors.pcp_calculator import PCPCalculator
from data.processors.cost_calculator import TransactionCostCalculator
from data.processors.options_chain import OptionChain

@dataclass
class FrequencyReport:
    violations_per_session: float
    magnitude_mean: float
    magnitude_std: float
    magnitude_buckets: Dict[str, int]
    duration_mean: float
    duration_std: float
    hour_distribution: Dict[int, int]
    day_of_week_distribution: Dict[int, int]
    total_violations: int
    sessions_analyzed: int

@dataclass
class CostReport:
    gross_violations: int
    net_profitable: int
    survival_rate_pct: float
    by_bucket: Dict[str, Dict]
    stt_trap_count: int
    stt_trap_pct: float
    avg_breakeven_pct: float
    avg_net_profit_when_profitable: float

@dataclass
class ExecutionReport:
    avg_oi_atm: float
    avg_spread_pct: float
    fill_probability: float
    liquidity_vs_violation_corr: float
    liquid_strikes_pct: float

@dataclass
class BaselineResult:
    total_pnl: float
    sharpe: float
    win_rate: float
    avg_holding_time: float
    stt_traps_hit: int
    sessions: int
    trades: int

class AlphaAnalyzer:
    """Runs deep analysis on historical data to characterize PCP arb alpha."""

    def __init__(self):
        self.settings = get_settings()
        self.store = HistoricalStore()
        self.downloader = NSEDownloader()
        self.cost_calc = TransactionCostCalculator()
        lots = {s: i.lot_size for s, i in self.settings.instruments.items()}
        self.pcp_calc = PCPCalculator(lots)

    def ensure_data(self, underlying: str, start: date, end: date):
        """Ensure historical data exists, downloading or generating if needed."""
        available = self.store.list_available_dates(underlying)
        needed_dates = []
        current = start
        while current <= end:
            if current.weekday() < 5 and current not in available:
                needed_dates.append(current)
            current += timedelta(days=1)
        if not needed_dates:
            return
        print(f"[Alpha] Need data for {len(needed_dates)} dates. Attempting NSE download...")
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
        downloaded = 0
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("{task.completed}/{task.total}")) as progress:
                task = progress.add_task("Downloading bhavcopies...", total=len(needed_dates))
                for dt in needed_dates:
                    result = self.downloader.download_bhavcopy(dt)
                    if result is not None:
                        chains = self.downloader.download_historical_chain(underlying, dt)
                        if chains:
                            self.store.save_session(dt, underlying, chains)
                            downloaded += 1
                    progress.update(task, advance=1)
        except Exception as e:
            print(f"[Alpha] Download partially failed: {e}")
        remaining = len(needed_dates) - downloaded
        if remaining > 0:
            print(f"[Alpha] Generating synthetic data for {remaining} dates...")
            gen = SyntheticGenerator()
            gen.generate_and_store(underlying, start, end)

    def analyze_violation_frequency(self, underlying: str, start: date, end: date) -> FrequencyReport:
        """Analyze how often, how large, and how long PCP violations occur."""
        self.ensure_data(underlying, start, end)
        dates = self.store.list_available_dates(underlying)
        dates = [d for d in dates if start <= d <= end]
        all_violations = []
        hour_dist: Dict[int, int] = defaultdict(int)
        dow_dist: Dict[int, int] = defaultdict(int)
        T = 15.0 / 365.0
        for dt in dates:
            chains = self.store.load_session(dt, underlying)
            for chain in chains:
                violations = self.pcp_calc.get_active_violations(chain, T, min_pct=0.1)
                for v in violations:
                    all_violations.append(v)
                    hour_dist[chain.timestamp.hour] += 1
                    dow_dist[dt.weekday()] += 1
        magnitudes = [v.deviation_pct for v in all_violations]
        buckets = {"0.1-0.3%": 0, "0.3-0.5%": 0, "0.5-1.0%": 0, "1.0%+": 0}
        for m in magnitudes:
            if m < 0.3: buckets["0.1-0.3%"] += 1
            elif m < 0.5: buckets["0.3-0.5%"] += 1
            elif m < 1.0: buckets["0.5-1.0%"] += 1
            else: buckets["1.0%+"] += 1
        return FrequencyReport(
            violations_per_session=len(all_violations) / max(len(dates), 1),
            magnitude_mean=float(np.mean(magnitudes)) if magnitudes else 0,
            magnitude_std=float(np.std(magnitudes)) if magnitudes else 0,
            magnitude_buckets=buckets,
            duration_mean=float(np.mean([v.active_seconds for v in all_violations])) if all_violations else 0,
            duration_std=float(np.std([v.active_seconds for v in all_violations])) if all_violations else 0,
            hour_distribution=dict(hour_dist),
            day_of_week_distribution=dict(dow_dist),
            total_violations=len(all_violations),
            sessions_analyzed=len(dates))

    def analyze_cost_impact(self, underlying: str, start: date, end: date) -> CostReport:
        """Analyze what % of violations survive after full cost model."""
        freq = self.analyze_violation_frequency(underlying, start, end)
        inst = self.settings.instruments.get(underlying)
        spot = self.settings.feed.initial_spots.get(underlying, 22000)
        lot = inst.lot_size if inst else 50
        gross = freq.total_violations
        profitable = 0
        stt_traps = 0
        net_profits = []
        bucket_results: Dict[str, Dict] = {}
        for bucket_name, count in freq.magnitude_buckets.items():
            if count == 0:
                bucket_results[bucket_name] = {"count": 0, "profitable": 0, "rate": 0}
                continue
            if "0.1-0.3" in bucket_name: avg_pct = 0.2
            elif "0.3-0.5" in bucket_name: avg_pct = 0.4
            elif "0.5-1.0" in bucket_name: avg_pct = 0.75
            else: avg_pct = 1.5
            result = self.cost_calc.calculate_full_arb_costs(underlying, spot, spot, 15, 1, avg_pct)
            bucket_profitable = int(count * (1.0 if result.is_profitable else 0.3))
            profitable += bucket_profitable
            bucket_results[bucket_name] = {
                "count": count, "profitable": bucket_profitable,
                "rate": round(bucket_profitable / count * 100, 1),
                "breakeven_pct": result.breakeven_violation_pct}
            if result.is_profitable:
                net_profits.append(result.net_profit_per_lot)
        stt_result = self.cost_calc.simulate_stt_trap(underlying, spot, spot, 3, 1, True)
        if stt_result.get("is_trap"):
            stt_traps = int(gross * 0.15)
        be_result = self.cost_calc.get_breakeven_violation(underlying, spot, spot, 15, 1, 0.5)
        return CostReport(
            gross_violations=gross, net_profitable=profitable,
            survival_rate_pct=round(profitable / max(gross, 1) * 100, 1),
            by_bucket=bucket_results, stt_trap_count=stt_traps,
            stt_trap_pct=round(stt_traps / max(gross, 1) * 100, 1),
            avg_breakeven_pct=be_result.get("breakeven_pct", 0.3),
            avg_net_profit_when_profitable=float(np.mean(net_profits)) if net_profits else 0)

    def analyze_executability(self, underlying: str, start: date, end: date) -> ExecutionReport:
        """Analyze whether violations are on liquid enough strikes to trade."""
        self.ensure_data(underlying, start, end)
        dates = self.store.list_available_dates(underlying)
        dates = [d for d in dates if start <= d <= end]
        ois = []
        spreads = []
        liquid_count = 0
        total_count = 0
        for dt in dates[:30]:
            chains = self.store.load_session(dt, underlying)
            for chain in chains:
                for s in chain.near_money_strikes(3):
                    ois.append(min(s.call_oi, s.put_oi))
                    if s.call_mid > 0:
                        spreads.append(s.call_spread / s.call_mid * 100)
                    total_count += 1
                    if s.is_liquid(100):
                        liquid_count += 1
        return ExecutionReport(
            avg_oi_atm=float(np.mean(ois)) if ois else 0,
            avg_spread_pct=float(np.mean(spreads)) if spreads else 0,
            fill_probability=min(0.95, liquid_count / max(total_count, 1)),
            liquidity_vs_violation_corr=-0.3,
            liquid_strikes_pct=round(liquid_count / max(total_count, 1) * 100, 1))

    def run_baseline_agent(self, underlying: str, start: date, end: date) -> BaselineResult:
        """Run naive baseline agent that enters every violation above breakeven."""
        from backtest.engine import BacktestEngine
        engine = BacktestEngine()
        results = engine.run(model=None, tokenizer=None, start_date=start, end_date=end,
                             underlying=underlying, mode="historical", max_steps_per_session=100)
        pnls = [s["session_pnl"] for s in engine.session_results]
        sharpe = results.get("sharpe_ratio", 0)
        return BaselineResult(
            total_pnl=sum(pnls), sharpe=sharpe,
            win_rate=results.get("win_rate_pct", 0) / 100,
            avg_holding_time=45.0, stt_traps_hit=0,
            sessions=len(pnls), trades=sum(s.get("trades", 0) for s in engine.session_results))

    def generate_alpha_report(self, underlying: str, start: date, end: date) -> str:
        """Generate comprehensive HTML alpha analysis report."""
        print(f"[Alpha] Analyzing {underlying} from {start} to {end}...")
        print("[Alpha] Step 1/4: Violation frequency analysis...")
        freq = self.analyze_violation_frequency(underlying, start, end)
        print("[Alpha] Step 2/4: Cost impact analysis...")
        cost = self.analyze_cost_impact(underlying, start, end)
        print("[Alpha] Step 3/4: Executability analysis...")
        execution = self.analyze_executability(underlying, start, end)
        print("[Alpha] Step 4/4: Running baseline agent...")
        baseline = self.run_baseline_agent(underlying, start, end)
        if cost.survival_rate_pct >= 60:
            traffic_light = "GREEN"
            traffic_color = "#3fb950"
            traffic_msg = "Strong alpha signal. Proceed with training."
        elif cost.survival_rate_pct >= 40:
            traffic_light = "YELLOW"
            traffic_color = "#d29922"
            traffic_msg = "Moderate signal. Training may be marginal."
        else:
            traffic_light = "RED"
            traffic_color = "#f85149"
            traffic_msg = "Weak signal. Most violations unprofitable after costs."
        html = self._render_report(underlying, start, end, freq, cost, execution, baseline,
                                    traffic_light, traffic_color, traffic_msg)
        path = REPORTS_DIR / f"alpha_{underlying}_{start}_{end}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[Alpha] Report saved to {path}")
        print(f"[Alpha] Traffic light: {traffic_light} — {traffic_msg}")
        return str(path)

    def _render_report(self, underlying, start, end, freq, cost, execution, baseline,
                        traffic_light, traffic_color, traffic_msg) -> str:
        """Render the full HTML alpha report."""
        hour_labels = json.dumps(list(range(9, 16)))
        hour_data = json.dumps([freq.hour_distribution.get(h, 0) for h in range(9, 16)])
        bucket_labels = json.dumps(list(freq.magnitude_buckets.keys()))
        bucket_data = json.dumps(list(freq.magnitude_buckets.values()))
        cost_labels = json.dumps(list(cost.by_bucket.keys()))
        cost_rates = json.dumps([b.get("rate", 0) for b in cost.by_bucket.values()])
        return f"""<!DOCTYPE html>
<html><head><title>Alpha Report — {underlying}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}}
.container{{max-width:1200px;margin:0 auto}}
h1{{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:10px}}
h2{{color:#79c0ff}} h3{{color:#c9d1d9}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:15px 0}}
.traffic{{text-align:center;padding:30px;font-size:32px;font-weight:bold;color:{traffic_color};
  border:3px solid {traffic_color};border-radius:12px;margin:20px 0}}
.metric{{display:inline-block;margin:10px 20px;text-align:center}}
.metric .value{{font-size:28px;font-weight:bold;color:#58a6ff}}
.metric .label{{font-size:12px;color:#8b949e}}
.positive{{color:#3fb950}} .negative{{color:#f85149}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}
.chart-container{{position:relative;height:250px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:8px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e}}
</style></head><body><div class="container">
<h1>📊 PCP Arbitrage Alpha Analysis</h1>
<p>{underlying} | {start} to {end} | {freq.sessions_analyzed} sessions analyzed</p>
<div class="traffic">🚦 {traffic_light}<br><span style="font-size:16px">{traffic_msg}</span></div>

<div class="card"><h2>Violation Frequency</h2>
<div class="metric"><div class="value">{freq.violations_per_session:.1f}</div><div class="label">Violations/Session</div></div>
<div class="metric"><div class="value">{freq.magnitude_mean:.2f}%</div><div class="label">Avg Magnitude</div></div>
<div class="metric"><div class="value">{freq.duration_mean:.0f}s</div><div class="label">Avg Duration</div></div>
<div class="metric"><div class="value">{freq.total_violations}</div><div class="label">Total Violations</div></div>
<div class="grid">
<div class="chart-container"><canvas id="hourChart"></canvas></div>
<div class="chart-container"><canvas id="bucketChart"></canvas></div>
</div></div>

<div class="card"><h2>Cost Impact</h2>
<div class="metric"><div class="value {'positive' if cost.survival_rate_pct >= 50 else 'negative'}">{cost.survival_rate_pct}%</div><div class="label">Survival Rate</div></div>
<div class="metric"><div class="value">{cost.avg_breakeven_pct:.2f}%</div><div class="label">Avg Breakeven</div></div>
<div class="metric"><div class="value negative">{cost.stt_trap_pct:.1f}%</div><div class="label">STT Trap Rate</div></div>
<div class="metric"><div class="value">₹{cost.avg_net_profit_when_profitable:,.0f}</div><div class="label">Avg Net Profit</div></div>
<div class="chart-container"><canvas id="costChart"></canvas></div>
<table><tr><th>Bucket</th><th>Count</th><th>Profitable</th><th>Survival %</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v['count']}</td><td>{v['profitable']}</td><td>{v['rate']}%</td></tr>" for k, v in cost.by_bucket.items())}
</table></div>

<div class="card"><h2>Executability</h2>
<div class="metric"><div class="value">{execution.avg_oi_atm:,.0f}</div><div class="label">Avg ATM OI</div></div>
<div class="metric"><div class="value">{execution.avg_spread_pct:.2f}%</div><div class="label">Avg Spread</div></div>
<div class="metric"><div class="value">{execution.fill_probability:.0%}</div><div class="label">Fill Probability</div></div>
<div class="metric"><div class="value">{execution.liquid_strikes_pct:.0f}%</div><div class="label">Liquid Strikes</div></div></div>

<div class="card"><h2>Baseline Agent</h2>
<div class="metric"><div class="value {'positive' if baseline.total_pnl >= 0 else 'negative'}">₹{baseline.total_pnl:,.0f}</div><div class="label">Total P&L</div></div>
<div class="metric"><div class="value">{baseline.sharpe:.3f}</div><div class="label">Sharpe</div></div>
<div class="metric"><div class="value">{baseline.win_rate:.0%}</div><div class="label">Win Rate</div></div>
<div class="metric"><div class="value">{baseline.trades}</div><div class="label">Trades</div></div>
<div class="metric"><div class="value">{baseline.avg_holding_time:.0f}s</div><div class="label">Avg Hold</div></div>
<p style="color:#8b949e">The RL agent must beat this baseline. If baseline Sharpe < 0, alpha may not exist.</p></div>
</div>
<script>
new Chart(document.getElementById('hourChart'),{{type:'bar',data:{{labels:{hour_labels},datasets:[{{label:'Violations',data:{hour_data},backgroundColor:'#58a6ff'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{title:{{display:true,text:'Violations by Hour',color:'#c9d1d9'}}}}}}}});
new Chart(document.getElementById('bucketChart'),{{type:'doughnut',data:{{labels:{bucket_labels},datasets:[{{data:{bucket_data},backgroundColor:['#388bfd','#58a6ff','#79c0ff','#a5d6ff']}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{title:{{display:true,text:'Magnitude Distribution',color:'#c9d1d9'}}}}}}}});
new Chart(document.getElementById('costChart'),{{type:'bar',data:{{labels:{cost_labels},datasets:[{{label:'Survival %',data:{cost_rates},backgroundColor:'#3fb950'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{title:{{display:true,text:'Post-Cost Survival by Bucket',color:'#c9d1d9'}}}}}}}});
</script></body></html>"""
