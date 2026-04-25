"""
Rollout collector — runs the LLM agent in the environment and collects trajectories.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pcp_arb_env.environment import PCPArbEnv, StepResult

SYSTEM_PROMPT = """You are a professional NSE option trader using a multi-factor approach.
You MUST output ONLY valid JSON. No other text.

Your job:
1. Analyze Fundamentals: Use news sentiment and valuation data.
2. Analyze Technicals: Use RSI, EMA, and Option Greeks (Delta/Gamma).
3. Execute: Enter PCP violations or directional trades ONLY when both technicals and news align.
4. Risk: Avoid the STT trap near expiry and monitor daily P&L limits.

Available Servers: market_data, risk, cost, technical, news.

Output format:
{"tool_calls": [{"server": "news", "tool": "get_news_summary", "params": {"date_iso": "2024-06-28"}}, {"server": "technical", "tool": "get_rsi", "params": {"symbol": "NIFTY"}}], "action_type": "hold", "strike": null, "qty": 1}
"""

def parse_action(response: str) -> Tuple[Dict, bool]:
    """Parse LLM response into action dict. Returns (action, success)."""
    clean = response.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean)
    if json_match:
        clean = json_match.group()
    try:
        action = json.loads(clean)
        if not isinstance(action, dict):
            return {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}, False
        if "action_type" not in action:
            action["action_type"] = "hold"
        if "tool_calls" not in action:
            action["tool_calls"] = []
        if "strike" not in action:
            action["strike"] = None
        if "qty" not in action:
            action["qty"] = 1
        return action, True
    except (json.JSONDecodeError, ValueError):
        return {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}, False

def collect_rollout(env: PCPArbEnv, model, tokenizer,
                    max_steps: int = 50, device: str = "cuda") -> List[Dict]:
    """
    Run one episode: reset env, generate LLM responses, collect trajectories.
    
    Returns list of (prompt, completion, reward) tuples for GRPO training.
    """
    obs = env.reset()
    trajectories = []
    total_reward = 0.0
    parse_failures = 0
    for step in range(max_steps):
        if env.done:
            break
        prompt = f"{SYSTEM_PROMPT}\n\nCurrent state:\n{obs}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=120, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        action, parsed_ok = parse_action(response)
        if not parsed_ok:
            parse_failures += 1
        result = env.step(action)
        trajectories.append({
            "prompt": prompt, "completion": response,
            "reward": result.reward.total,
            "reward_breakdown": result.reward.to_dict(),
            "action": action, "parsed_ok": parsed_ok,
            "step": step, "daily_pnl": result.info.get("daily_pnl", 0)})
        total_reward += result.reward.total
        obs = result.observation
    return trajectories

def collect_rollout_simple(env: PCPArbEnv, response_text: str) -> Tuple[float, Dict]:
    """Single-step rollout for GRPO reward function. Returns (reward, info)."""
    action, parsed_ok = parse_action(response_text)
    result = env.step(action)
    return result.reward.total, {
        "parsed_ok": parsed_ok, "action": action,
        "reward_breakdown": result.reward.to_dict(),
        "daily_pnl": result.info.get("daily_pnl", 0)}
