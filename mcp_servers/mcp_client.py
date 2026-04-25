"""
MCP Client — wraps all three MCP servers with caching, timeouts, and retries.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

@dataclass
class ToolDefinition:
    server: str
    name: str
    description: str
    params: Dict[str, str]

class MCPClient:
    """Client for communicating with MCP servers via HTTP."""

    SERVER_URLS = {
        "market_data": "http://localhost:8001",
        "risk": "http://localhost:8002",
        "cost": "http://localhost:8003",
        "technical": "http://localhost:8004",
        "news": "http://localhost:8005",
    }

    CACHE_TTL = {
        "get_option_chain": 1.0,
        "get_spot_price": 0.5,
        "get_pcp_deviation": 1.0,
        "get_iv_surface": 2.0,
        "get_historical_violations": 5.0,
        "get_market_regime": 2.0,
        "get_position_state": 0.0,
        "check_entry_allowed": 0.0,
        "get_daily_pnl": 0.0,
        "estimate_exit_pnl": 0.0,
        "get_risk_limits": 5.0,
        "calculate_arb_costs": 5.0,
        "get_breakeven_violation": 5.0,
        "simulate_stt_trap": 5.0,
        "get_cost_history": 2.0,
        "get_rsi": 5.0,
        "get_ema": 5.0,
        "get_greeks": 5.0,
        "get_news_summary": 30.0,
    }

    def __init__(self, server_urls: Dict[str, str] = None, timeout: float = 0.2, max_retries: int = 1):
        self.server_urls = server_urls or self.SERVER_URLS
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, float] = {}
        self._client = httpx.Client(timeout=timeout)
        self._call_count = 0
        self._error_count = 0

    def call_tool(self, server: str, tool: str, params: Dict = None) -> Dict:
        """
        Call an MCP tool endpoint.
        
        Args:
            server: Server name (market_data, risk, cost)
            tool: Tool name (e.g., get_option_chain)
            params: Tool parameters
        
        Returns:
            JSON response from the server.
        """
        if params is None:
            params = {}
        cache_key = f"{server}:{tool}:{str(sorted(params.items()))}"
        ttl = self.CACHE_TTL.get(tool, 0.0)
        if ttl > 0 and cache_key in self._cache:
            age = time.time() - self._cache_times.get(cache_key, 0)
            if age < ttl:
                return self._cache[cache_key]
        base_url = self.server_urls.get(server)
        if not base_url:
            return {"error": f"Unknown server: {server}"}
        url = f"{base_url}/tools/{tool}"
        self._call_count += 1
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, json=params, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json()
                if ttl > 0:
                    self._cache[cache_key] = result
                    self._cache_times[cache_key] = time.time()
                return result
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.05)
                    continue
                self._error_count += 1
                return {"error": str(e), "server": server, "tool": tool}

    def call_internal(self, server: str, endpoint: str, data: Dict = None) -> Dict:
        """Call an internal (non-tool) endpoint on an MCP server."""
        base_url = self.server_urls.get(server)
        if not base_url:
            return {"error": f"Unknown server: {server}"}
        url = f"{base_url}/internal/{endpoint}"
        try:
            resp = self._client.post(url, json=data or {}, timeout=self.timeout * 2)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def push_feed_update(self, chain_dict: Dict) -> Dict:
        """Push a feed update to market data and technical servers."""
        res_md = {"status": "skipped"}
        res_tech = {"status": "skipped"}
        
        # Update Market Data
        url_md = f"{self.server_urls.get('market_data')}/feed/update"
        try:
            resp = self._client.post(url_md, json=chain_dict, timeout=self.timeout * 3)
            res_md = resp.json()
        except Exception as e:
            res_md = {"error": str(e)}
            
        # Update Technical Server
        url_tech = f"{self.server_urls.get('technical')}/feed/update"
        try:
            data_tech = {"symbol": chain_dict.get("underlying"), "price": chain_dict.get("spot_price")}
            resp = self._client.post(url_tech, json=data_tech, timeout=self.timeout * 3)
            res_tech = resp.json()
        except Exception as e:
            res_tech = {"error": str(e)}
            
        return {"market_data": res_md, "technical": res_tech}

    def check_health(self) -> Dict[str, bool]:
        """Check health of all MCP servers."""
        status = {}
        for name, url in self.server_urls.items():
            try:
                resp = self._client.get(f"{url}/health", timeout=1.0)
                status[name] = resp.status_code == 200
            except Exception:
                status[name] = False
        return status

    def get_tool_registry(self) -> List[ToolDefinition]:
        """Return all available tools in the format the LLM prompt expects."""
        return [
            ToolDefinition("market_data", "get_option_chain",
                           "Get full option chain with PCP deviations",
                           {"underlying": "str", "expiry": "str"}),
            ToolDefinition("market_data", "get_spot_price",
                           "Get current spot price with bid/ask",
                           {"symbol": "str"}),
            ToolDefinition("market_data", "get_pcp_deviation",
                           "Get PCP deviation for a specific strike",
                           {"underlying": "str", "strike": "float", "expiry": "str"}),
            ToolDefinition("market_data", "get_iv_surface",
                           "Get IV surface and skew data",
                           {"underlying": "str", "expiry": "str"}),
            ToolDefinition("market_data", "get_historical_violations",
                           "Get historical violation statistics",
                           {"underlying": "str", "lookback_sessions": "int"}),
            ToolDefinition("market_data", "get_market_regime",
                           "Get current market regime classification",
                           {"underlying": "str"}),
            ToolDefinition("risk", "get_position_state",
                           "Get all open positions with P&L and Greeks",
                           {}),
            ToolDefinition("risk", "check_entry_allowed",
                           "Check if a new position entry is allowed",
                           {"underlying": "str", "strike": "float", "qty": "int", "action_type": "str"}),
            ToolDefinition("risk", "get_daily_pnl",
                           "Get daily P&L summary",
                           {}),
            ToolDefinition("risk", "estimate_exit_pnl",
                           "Estimate P&L for exiting a position",
                           {"position_id": "str"}),
            ToolDefinition("risk", "get_risk_limits",
                           "Get current risk limit configuration",
                           {}),
            ToolDefinition("cost", "calculate_arb_costs",
                           "Calculate full arbitrage transaction costs",
                           {"underlying": "str", "strike": "float", "expiry_days": "int",
                            "qty": "int", "gross_violation_pct": "float"}),
            ToolDefinition("cost", "get_breakeven_violation",
                           "Get breakeven violation percentage",
                           {"underlying": "str", "strike": "float", "expiry_days": "int", "qty": "int"}),
            ToolDefinition("cost", "simulate_stt_trap",
                           "Simulate STT trap: costs if held to expiry vs exited early",
                           {"underlying": "str", "strike": "float", "expiry_days": "int",
                            "qty": "int", "hold_to_expiry": "bool"}),
            ToolDefinition("cost", "get_cost_history",
                           "Get historical cost statistics and trends",
                           {}),
            ToolDefinition("technical", "get_rsi",
                           "Get Relative Strength Index (RSI) for momentum",
                           {"symbol": "str", "period": "int"}),
            ToolDefinition("technical", "get_ema",
                           "Get Exponential Moving Average (EMA) for trend",
                           {"symbol": "str", "period": "int"}),
            ToolDefinition("technical", "get_greeks",
                           "Get option Greeks (Delta, Gamma, Theta)",
                           {"symbol": "str", "strike": "float", "expiry_days": "float", "iv": "float"}),
            ToolDefinition("news", "get_news_summary",
                           "Get headlines and sentiment score for a date",
                           {"date_iso": "str", "symbol": "str"}),
        ]

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return [t.name for t in self.get_tool_registry()]

    @property
    def stats(self) -> Dict:
        return {"total_calls": self._call_count, "errors": self._error_count,
                "cache_entries": len(self._cache)}

    def clear_cache(self):
        """Clear all cached responses."""
        self._cache.clear()
        self._cache_times.clear()

    def close(self):
        """Close the HTTP client."""
        self._client.close()
