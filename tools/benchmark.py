"""
Benchmark tool — measures system performance metrics.
"""
from __future__ import annotations
import time
from typing import Dict
from data.feeds.mock_feed import MockFeed
from mcp_servers.mcp_client import MCPClient
from pcp_arb_env.environment import PCPArbEnv
from config.settings import get_settings

def run_benchmark(n_steps: int = 100) -> Dict:
    """Benchmark environment step speed, feed tick speed, and MCP call latency."""
    settings = get_settings()
    feed = MockFeed(underlyings=["NIFTY"], num_strikes=10)
    mcp = MCPClient(timeout=5.0)
    env = PCPArbEnv(feed=feed, mcp_client=mcp)

    # Benchmark feed ticks
    feed.reset()
    start = time.time()
    for _ in range(n_steps):
        feed.next_tick()
    feed_time = time.time() - start

    # Benchmark env steps
    env.reset()
    start = time.time()
    for _ in range(n_steps):
        if env.done:
            break
        action = {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}
        env.step(action)
    env_time = time.time() - start

    results = {
        "feed_ticks_per_sec": round(n_steps / max(feed_time, 0.001), 1),
        "env_steps_per_sec": round(n_steps / max(env_time, 0.001), 1),
        "avg_tick_ms": round(feed_time / n_steps * 1000, 2),
        "avg_step_ms": round(env_time / n_steps * 1000, 2),
        "n_steps": n_steps}
    print(f"[Benchmark] Feed: {results['feed_ticks_per_sec']} ticks/s ({results['avg_tick_ms']}ms/tick)")
    print(f"[Benchmark] Env: {results['env_steps_per_sec']} steps/s ({results['avg_step_ms']}ms/step)")
    return results

if __name__ == "__main__":
    run_benchmark()
