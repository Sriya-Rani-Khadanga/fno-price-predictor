"""
PCP Arb RL System — Main entry point.
Supports multiple modes: alpha, backtest, train, paper, demo, analyze.
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_alpha(args):
    """Run alpha analysis to determine if PCP arb opportunity exists."""
    from tools.alpha_analyzer import AlphaAnalyzer
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    analyzer = AlphaAnalyzer()
    report_path = analyzer.generate_alpha_report(args.underlying, start, end)
    print(f"\n✅ Alpha report generated: {report_path}")
    print("Open this HTML file in a browser to see interactive charts.")


def cmd_backtest(args):
    """Run backtest over a date range."""
    from backtest.engine import BacktestEngine
    from backtest.report import generate_report
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    engine = BacktestEngine(initial_capital=args.capital)

    model, tokenizer = None, None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Main] Loading model from {args.checkpoint}...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint, max_seq_length=1144,
                dtype=None, load_in_4bit=True)
        except Exception as e:
            print(f"[Main] Could not load model: {e}. Running baseline agent.")

    results = engine.run(model=model, tokenizer=tokenizer,
                         start_date=start, end_date=end,
                         underlying=args.underlying, mode=args.feed_mode)
    report_path = generate_report(
        {"summary": results, "sessions": engine.session_results},
        args.underlying, start, end)
    print(f"\n✅ Backtest complete!")
    print(f"   Total P&L: ₹{results['total_pnl']:,.0f}")
    print(f"   Sharpe: {results['sharpe_ratio']:.3f}")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Report: {report_path}")

    # Generate step analysis for 3 sample sessions
    if engine.session_results:
        from tools.recorder import StepAnalyzer
        analyzer = StepAnalyzer()
        for session in engine.session_results[:3]:
            log_path = session.get("log_path")
            if log_path and os.path.exists(log_path):
                with open(log_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        analyzer.record_step(
                            step=entry.get("step", 0),
                            observation="",
                            raw_output=json.dumps(entry.get("action", {})),
                            parsed_action=entry.get("action", {}),
                            tool_calls=entry.get("action", {}).get("tool_calls", []),
                            tool_results={},
                            reward_breakdown=entry.get("reward_breakdown", {}),
                            position_delta=0,
                            cumulative_pnl=entry.get("pnl", 0),
                            timestamp=datetime.now())
                report = analyzer.generate_step_report(session.get("date", "unknown"))
                print(f"   Step analysis: {report}")
                analyzer.clear()


def cmd_train(args):
    """Start GRPO training with curriculum progression."""
    from training.train import train
    print(f"[Main] Starting training for {args.steps} steps...")
    checkpoint_path = train(
        total_steps=args.steps,
        checkpoint_path=args.checkpoint,
        wandb_enabled=not args.no_wandb)
    print(f"\n✅ Training complete! Checkpoint: {checkpoint_path}")


def cmd_paper(args):
    """Run paper trading with live feed and dashboard."""
    from data.feeds.live_feed import LiveFeed
    from data.feeds.mock_feed import MockFeed
    from mcp_servers.mcp_client import MCPClient
    from pcp_arb_env.environment import PCPArbEnv
    from monitoring.dashboard import Dashboard
    from monitoring.alerts import AlertManager
    from training.rollout import SYSTEM_PROMPT, parse_action

    if args.feed == "live":
        feed = LiveFeed()
        print("[Main] Starting live feed (polling NSE endpoints)...")
    else:
        feed = MockFeed(underlyings=["NIFTY", "BANKNIFTY"],
                        violations_per_session=10,
                        violation_pct_range=(0.3, 1.5))
        print("[Main] Starting mock feed for paper trading...")

    mcp = MCPClient(timeout=5.0)
    env = PCPArbEnv(feed=feed, mcp_client=mcp)
    dashboard = Dashboard()
    alerts = AlertManager()
    dashboard.start_logging()

    model, tokenizer = None, None
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint, max_seq_length=1144,
                dtype=None, load_in_4bit=True)
            print(f"[Main] Loaded model from {args.checkpoint}")
        except Exception as e:
            print(f"[Main] Model load failed: {e}")

    obs = env.reset()
    dashboard.update(feed_health=mcp.check_health())

    print("[Main] Paper trading started. Press Ctrl+C to stop.")
    try:
        import time
        step = 0
        while not env.done:
            step += 1
            if model and tokenizer:
                import torch
                prompt = f"{SYSTEM_PROMPT}\n\nCurrent state:\n{obs}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=120,
                                              do_sample=True, temperature=0.5,
                                              pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                action, _ = parse_action(response)
            else:
                action = {"action_type": "hold", "tool_calls": [
                    {"server": "market_data", "tool": "get_option_chain",
                     "params": {"underlying": "NIFTY", "expiry": ""}}
                ], "strike": None, "qty": 1}

            result = env.step(action)
            state = env.state()
            dashboard.update(
                market={"violations": state.get("violations", [])},
                agent={"action": action, "reward_breakdown": result.reward.to_dict(),
                       "tool_calls": action.get("tool_calls", [])},
                positions=state.get("positions", []),
                pnl=state.get("daily_pnl", 0),
                feed_health=mcp.check_health(),
                step=step)
            alerts.check_daily_pnl(state.get("daily_pnl", 0), 50000)
            obs = result.observation
            from rich.live import Live
            # In a real run we'd use Live, but for this loop we'll just print
            # However, the current code just calls render() and discards it.
            # Let's fix it to use console.print
            from rich.console import Console
            console = Console()
            console.print(dashboard.render())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Paper trading stopped.")
        print(f"[Main] Final P&L: ₹{env._daily_pnl:,.0f}")


def cmd_demo(args):
    """Run interactive CLI demo."""
    from tools.demo import run_demo
    run_demo()


def cmd_analyze(args):
    """Analyze a recorded session log."""
    from tools.recorder import StepAnalyzer
    if not os.path.exists(args.session):
        print(f"[Main] Session file not found: {args.session}")
        return
    analyzer = StepAnalyzer()
    with open(args.session) as f:
        for line in f:
            entry = json.loads(line)
            analyzer.record_step(
                step=entry.get("step", 0), observation="",
                raw_output=json.dumps(entry.get("action", {})),
                parsed_action=entry.get("action", {}),
                tool_calls=entry.get("action", {}).get("tool_calls", []),
                tool_results={}, reward_breakdown=entry.get("reward_breakdown", {}),
                position_delta=0, cumulative_pnl=entry.get("pnl", 0),
                timestamp=datetime.now())
    report = analyzer.generate_step_report(Path(args.session).stem)
    print(f"✅ Step analysis report: {report}")


def main():
    """Main entry point for the PCP Arb RL system."""
    parser = argparse.ArgumentParser(
        description="PCP Arbitrage RL System — Put-Call Parity Arb with MCP + GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode alpha --underlying NIFTY --start 2024-01-01 --end 2024-06-30
  python main.py --mode backtest --underlying NIFTY --start 2024-01-01 --end 2024-06-30
  python main.py --mode train --steps 3000
  python main.py --mode paper --feed mock
  python main.py --mode demo
  python main.py --mode analyze --session logs/session_20240415.jsonl
        """)
    parser.add_argument("--mode", required=True,
                        choices=["alpha", "backtest", "train", "paper", "demo", "analyze"],
                        help="Operating mode")
    parser.add_argument("--underlying", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--feed", default="mock", choices=["mock", "live", "historical"],
                        help="Feed type for paper/backtest")
    parser.add_argument("--feed-mode", default="historical",
                        choices=["historical", "mock"], help="Backtest feed mode")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital for backtest")
    parser.add_argument("--session", default=None, help="Session log path for analyze mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    print(f"🔄 PCP Arbitrage RL System — Mode: {args.mode}")
    print(f"{'='*60}")

    mode_map = {
        "alpha": cmd_alpha,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "paper": cmd_paper,
        "demo": cmd_demo,
        "analyze": cmd_analyze,
    }
    mode_map[args.mode](args)


if __name__ == "__main__":
    main()
