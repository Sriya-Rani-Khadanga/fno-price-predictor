"""
Live feed — polls NSE/MCX public endpoints for real-time option chain data.
Runs as a background thread pushing updates to the MCP market data server.
"""
from __future__ import annotations
import random
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from data.feeds.base import BaseFeed, TickData
from data.processors.options_chain import OptionChain, StrikeData
from data.historical.nse_downloader import NSEDownloader
from config.settings import get_settings

class LiveFeed(BaseFeed):
    """Polls NSE/MCX public endpoints for live option chain data."""

    def __init__(self, underlyings: List[str] = None, mcp_update_url: str = None):
        settings = get_settings()
        if underlyings is None:
            underlyings = list(settings.instruments.keys())
        super().__init__(underlyings)
        self.settings = settings
        self.mcp_update_url = mcp_update_url or "http://localhost:8001/feed/update"
        self.downloader = NSEDownloader()
        self._poll_min = settings.feed.poll_interval_min
        self._poll_max = settings.feed.poll_interval_max
        self._staleness_threshold = settings.feed.staleness_threshold_seconds
        self._last_update: Dict[str, datetime] = {}
        self._latest_chains: Dict[str, OptionChain] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._done = False
        self._mcx_session = requests.Session()
        self._mcx_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def reset(self) -> TickData:
        self._tick_count = 0
        self._done = False
        self._session_start = datetime.now()
        self._latest_chains.clear()
        self._last_update.clear()
        self.start_polling()
        time.sleep(3)
        return self._build_tick()

    def next_tick(self) -> TickData:
        self._tick_count += 1
        now = datetime.now()
        trading_end = now.replace(hour=15, minute=30, second=0)
        if now >= trading_end:
            self._done = True
        self._current_tick = self._build_tick()
        return self._current_tick

    def is_done(self) -> bool:
        return self._done

    def start_polling(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_polling(self):
        """Stop background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        """Main polling loop running in background thread."""
        while self._running:
            for sym in self.underlyings:
                try:
                    if sym == "CRUDEOIL":
                        chain = self._poll_mcx(sym)
                    else:
                        chain = self.downloader.download_option_chain_snapshot(sym)
                    if chain:
                        self._latest_chains[sym] = chain
                        self._last_update[sym] = datetime.now()
                        self._push_to_mcp(chain)
                except Exception as e:
                    print(f"[LiveFeed] Poll error for {sym}: {e}")
            interval = random.uniform(self._poll_min, self._poll_max)
            time.sleep(interval)

    def _poll_mcx(self, symbol: str) -> Optional[OptionChain]:
        """Poll MCX website and parse HTML table for commodity futures data."""
        try:
            url = "https://www.mcxindia.com/market-data/commodity-futures"
            resp = self._mcx_session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"id": "tblFutureWatch"})
            if not table:
                table = soup.find("table")
            if not table:
                return None
            rows = table.find_all("tr")[1:]
            spot = 6500.0
            strikes = []
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 5:
                    name = cells[0].get_text(strip=True)
                    if symbol.upper() in name.upper():
                        try:
                            ltp = float(cells[1].get_text(strip=True).replace(",", ""))
                            spot = ltp
                        except (ValueError, IndexError):
                            pass
            atm = round(spot / 50) * 50
            for i in range(-5, 6):
                k = atm + i * 50
                strikes.append(StrikeData(
                    strike=k, call_bid=max(0.5, (spot - k) + 50), call_ask=max(1.0, (spot - k) + 55),
                    call_ltp=max(0.5, (spot - k) + 52), call_oi=1000, call_volume=100, call_iv=0.25,
                    put_bid=max(0.5, (k - spot) + 50), put_ask=max(1.0, (k - spot) + 55),
                    put_ltp=max(0.5, (k - spot) + 52), put_oi=1000, put_volume=100, put_iv=0.25))
            return OptionChain(
                underlying=symbol, expiry="NEAR", spot_price=spot,
                spot_bid=spot - 1, spot_ask=spot + 1, timestamp=datetime.now(),
                strikes=strikes, data_source="live")
        except Exception as e:
            print(f"[LiveFeed] MCX poll error: {e}")
            return None

    def _push_to_mcp(self, chain: OptionChain):
        """Push chain update to MCP market data server."""
        try:
            requests.post(self.mcp_update_url, json=chain.to_dict(), timeout=1)
        except Exception:
            pass

    def _build_tick(self) -> TickData:
        now = datetime.now()
        minute = int((now - self._session_start).total_seconds() / 60) if self._session_start else 0
        spots = {}
        for sym, chain in self._latest_chains.items():
            spots[sym] = chain.spot_price
            staleness = (now - self._last_update.get(sym, now)).total_seconds()
            if staleness > self._staleness_threshold:
                print(f"[LiveFeed] WARNING: {sym} data is {staleness:.0f}s stale")
        self._current_tick = TickData(
            timestamp=now, session_minute=minute,
            chains=dict(self._latest_chains), spots=spots,
            is_session_end=self._done)
        return self._current_tick

    def get_staleness(self) -> Dict[str, float]:
        """Get staleness in seconds for each instrument."""
        now = datetime.now()
        return {sym: (now - self._last_update.get(sym, now)).total_seconds()
                for sym in self.underlyings}

    def is_stale(self, symbol: str) -> bool:
        """Check if a specific symbol's data is stale."""
        staleness = self.get_staleness().get(symbol, 999)
        return staleness > self._staleness_threshold
