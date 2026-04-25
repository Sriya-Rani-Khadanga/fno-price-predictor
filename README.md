# PCP Arb RL — Put-Call Parity Arbitrage with Reinforcement Learning

A production-grade RL system that trains an LLM agent to exploit put-call parity violations in NSE equity options and MCX commodity futures, using MCP (Model Context Protocol) servers as the agent's tooling interface.

## Why RL for PCP Arbitrage?

### The STT Trap Problem
Put-call parity violations in Indian markets appear frequently, but **most are unprofitable after transaction costs**. The critical cost is **STT on exercise** — if you hold an in-the-money option to expiry, NSE charges **0.125% of intrinsic value** as Securities Transaction Tax. This single cost makes many apparently profitable arbitrage opportunities deeply unprofitable.

A static ML model that simply detects violations fails because:
1. **Cost-aware sequencing matters**: The agent must learn to check costs *before* entering, not after
2. **Timing is everything**: Enter early in a violation, exit before it closes — but also exit before expiry to avoid the STT trap
3. **Tool use is the differentiator**: The agent must actively query cost servers and risk servers to make informed decisions

### Why MCP Servers?
Instead of passively receiving a state dict, our agent **actively calls MCP tools** to gather market intelligence. This mirrors how a real trader would operate — checking the option chain, computing costs, evaluating risk limits — before making a decision.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  LLM Agent (Qwen2.5-1.5B)        │
│        Trained via GRPO (TRL + Unsloth)          │
└────────────┬─────────────────────┬───────────────┘
             │ tool_calls          │ trade actions
     ┌───────▼───────┐     ┌──────▼──────┐
     │  MCP Servers   │     │  OpenEnv    │
     │  (FastAPI)     │     │ Environment │
     ├───────────────┤     ├─────────────┤
     │ Market Data   │     │ Feed → Tick │
     │ :8001         │     │ Rewards (4) │
     ├───────────────┤     │ Curriculum  │
     │ Risk          │     │ Order Sim   │
     │ :8002         │     └─────────────┘
     ├───────────────┤
     │ Cost          │
     │ :8003         │
     └───────────────┘
```

## Three Runtime Modes

### 1. Historical Replay
Replays real NSE bhavcopy data with GBM-interpolated intraday movement. Uses `HistoricalFeed` that downloads data from NSE's public archives.

### 2. Synthetic Simulation  
`MockFeed` generates GBM price paths with injected PCP violations at configurable rates, realistic microstructure noise, and bid-ask spreads.

### 3. Live Paper Trading
`LiveFeed` polls NSE's unofficial public endpoints every 3 seconds with proper session cookie management. Runs the trained agent in real-time.

## Quick Start

### 1. Run Alpha Analysis First
```bash
python main.py --mode alpha --underlying NIFTY --start 2024-01-01 --end 2024-06-30
```
This downloads historical data, analyzes violation frequency and cost impact, runs a baseline agent, and generates an HTML report with a traffic light (GREEN/YELLOW/RED) telling you if training is worthwhile.

### 2. Backtest
```bash
python main.py --mode backtest --underlying NIFTY --start 2024-01-01 --end 2024-06-30
```
Runs the baseline agent over historical sessions and generates equity curve report.

### 3. Train
```bash
python main.py --mode train --steps 3000 --checkpoint checkpoints/latest
```
Starts GRPO training with 4-stage curriculum progression. Logs to WandB.

### 4. Paper Trade
```bash
python main.py --mode paper --feed mock
# or with live NSE data:
python main.py --mode paper --feed live
```
Shows rich terminal dashboard with real-time P&L, violations, and agent decisions.

### 5. Interactive Demo
```bash
python main.py --mode demo
```
Interactive CLI where you manually step through the environment.

### 6. Analyze Session
```bash
python main.py --mode analyze --session logs/session_20240415_103245.jsonl
```
Generates step-by-step HTML analysis of a recorded session.

## Reward System

| Component | Weight | What it rewards |
|-----------|--------|----------------|
| Profitability | 35% | Realized P&L gains, penalizes losses |
| Timing | 25% | Early entry in violations, exit before close |
| Cost Awareness | 25% | Using cost tools before entry, avoiding STT traps |
| Format Compliance | 15% | Valid JSON output with correct fields |

## Curriculum Stages

| Stage | Steps | Violations | Feed | Focus |
|-------|-------|-----------|------|-------|
| OBVIOUS | 0–500 | >1% | Mock | Basic enter/exit loop |
| MODERATE | 500–1500 | 0.5–1% | Mock | Cost tool usage |
| REALISTIC | 1500–3000 | 0.3–0.8% | Mixed | STT trap avoidance |
| LIVE_SIM | 3000+ | 0.1–0.5% | Live/Historical | Production readiness |

## Docker Deployment

```bash
docker-compose up -d
# Set mode via environment variables:
MODE=backtest UNDERLYING=NIFTY docker-compose up app
```

## Project Structure

```
pcp-arb-rl/
├── pcp_arb_env/          # OpenEnv environment
│   ├── environment.py    # Core env with MCP tool-calling
│   ├── observations.py   # Text observation builder
│   ├── rewards.py        # 4-component reward system
│   └── curriculum.py     # Stage management
├── mcp_servers/          # FastAPI MCP tool servers
│   ├── market_data_server.py  # Port 8001
│   ├── risk_server.py         # Port 8002
│   ├── cost_server.py         # Port 8003
│   └── mcp_client.py         # Client with caching
├── data/
│   ├── feeds/            # Mock, Historical, Live feeds
│   ├── processors/       # Options chain, PCP calc, costs
│   └── historical/       # NSE downloader, store
├── training/             # GRPO training with Unsloth
├── backtest/             # Engine, metrics, HTML reports
├── execution/            # Order simulator, risk manager
├── signals/              # Signal generator, filters
├── models/               # Feature engineering, ensemble
├── tools/                # Alpha analyzer, recorder, demo
├── monitoring/           # Rich dashboard, alerts
├── config/               # Settings, instruments.yaml
├── main.py               # Entry point (6 modes)
├── docker-compose.yml    # 3 MCP servers + app
└── openenv.yaml          # OpenEnv manifest
```

## Key Technical Decisions

- **STT on exercise modeled correctly**: 0.125% of intrinsic value — this is the critical cost that makes most apparent arb opportunities unprofitable
- **MCP servers as separate processes**: Enables the agent to actively query for information rather than receiving a passive state dict
- **4-bit quantized Qwen2.5-1.5B**: Small enough for fast inference, large enough for tool-calling reasoning
- **Walk-forward training**: No shuffling — curriculum progression ensures the agent encounters increasingly realistic scenarios

## Known Limitations

- **NSE bhavcopy is EOD only**: Intraday violations are synthesized using GBM interpolation from OHLC data
- **Live feed has 3s latency**: NSE rate limits unofficial API access; poll interval is randomized 2.5–3.5s
- **MCX data via HTML scraping**: MCX has no public JSON API, so we parse HTML tables
- **No real execution**: Paper trading only — no actual order submission
