"""
NSE F&O Data Collector — 3-layer data pipeline.

Layer 1: NSE live option chain scraper (market hours, zero auth)
Layer 2: NSE Bhavcopy downloader (EOD, full history, zero auth)
Layer 3: Breeze API historical 1-min OHLCV (free with ICICI account)

Usage:
    python nse_data_collector.py --mode bhavcopy --start 2023-01-01 --end 2024-12-31
    python nse_data_collector.py --mode live --symbol NIFTY
    python nse_data_collector.py --mode breeze --symbol NIFTY --start 2022-01-01 --end 2024-12-31
    python nse_data_collector.py --mode pcp --symbol NIFTY
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ─── Configuration ────────────────────────────────────────────────────────────

CACHE = Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: NSE LIVE OPTION CHAIN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

class NSELiveScraper:
    """
    Scrapes live option chain from NSE website during market hours.
    No authentication needed — uses session cookies from NSE homepage.
    
    Captures per-strike: LTP, bid, ask, OI, change in OI, volume, IV
    for both calls and puts.
    """
    BASE = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookie_time: Optional[datetime] = None
        self._refresh_cookies()

    def _refresh_cookies(self):
        """NSE requires a valid session cookie — hit homepage first."""
        try:
            self.session.get(self.BASE, timeout=10)
            self.session.get(f"{self.BASE}/option-chain", timeout=10)
            self._cookie_time = datetime.now()
            time.sleep(1)
        except Exception as e:
            print(f"[NSE] Cookie refresh failed: {e}")

    def _ensure_cookies(self):
        """Refresh cookies if they're older than 4 minutes."""
        if (self._cookie_time is None or
                (datetime.now() - self._cookie_time).total_seconds() > 240):
            self._refresh_cookies()

    def get_option_chain(self, symbol: str = "NIFTY") -> Dict:
        """
        Fetch and parse the full option chain for a symbol.
        
        Returns dict with: spot, expiry_dates, chain (DataFrame), timestamp
        """
        self._ensure_cookies()

        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            url = f"{self.BASE}/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"{self.BASE}/api/option-chain-equities?symbol={symbol}"

        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=15)
                if r.status_code == 401:
                    self._refresh_cookies()
                    continue
                if r.status_code != 200:
                    print(f"[NSE] HTTP {r.status_code} for {symbol}")
                    continue
                data = r.json()
                return self._parse_chain(data, symbol)
            except Exception as e:
                print(f"[NSE] Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
                self._refresh_cookies()
        return {}

    def _parse_chain(self, raw: dict, symbol: str) -> Dict:
        """Parse NSE API JSON into a structured DataFrame."""
        records = raw.get("records", {})
        spot = records.get("underlyingValue", 0)
        expiry_dates = records.get("expiryDates", [])
        ts = datetime.now()

        rows = []
        for entry in records.get("data", []):
            strike = entry.get("strikePrice")
            expiry = entry.get("expiryDate")

            for side in ("CE", "PE"):
                d = entry.get(side)
                if not d:
                    continue
                rows.append({
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "spot": spot,
                    "expiry": expiry,
                    "strike": strike,
                    "type": side,
                    "ltp": d.get("lastPrice", 0),
                    "bid_qty": d.get("bidQty", d.get("buyQuantity1", 0)),
                    "bid": d.get("bidprice", d.get("buyPrice1", 0)),
                    "ask": d.get("askPrice", d.get("sellPrice1", 0)),
                    "ask_qty": d.get("askQty", d.get("sellQuantity1", 0)),
                    "oi": d.get("openInterest", 0),
                    "oi_change": d.get("changeinOpenInterest", 0),
                    "volume": d.get("totalTradedVolume", 0),
                    "iv": d.get("impliedVolatility", 0),
                    "ltp_change": d.get("change", 0),
                    "ltp_change_pct": d.get("pchangeinOpenInterest", 0),
                })

        df = pd.DataFrame(rows)
        return {
            "spot": spot,
            "expiry_dates": expiry_dates,
            "chain": df,
            "timestamp": ts.isoformat(),
        }

    def scrape_to_parquet(
        self,
        symbol: str,
        interval_seconds: int = 60,
        duration_minutes: int = 375,
    ) -> pd.DataFrame:
        """
        Scrape live option chain at fixed intervals for a full trading day.
        Run during market hours 9:15 AM – 3:30 PM IST.
        
        Saves combined parquet at end of session.
        """
        today = date.today().isoformat()
        out_dir = CACHE / "live" / symbol
        out_dir.mkdir(parents=True, exist_ok=True)

        snapshots: List[pd.DataFrame] = []
        total = (duration_minutes * 60) // interval_seconds

        print(f"[Live] Scraping {symbol} every {interval_seconds}s "
              f"for {duration_minutes} min ({total} snapshots)...")

        for i in range(total):
            result = self.get_option_chain(symbol)
            if result and "chain" in result and not result["chain"].empty:
                snapshots.append(result["chain"])
                print(f"  [{i + 1}/{total}] {result['timestamp']} "
                      f"spot={result['spot']} rows={len(result['chain'])}")
            else:
                print(f"  [{i + 1}/{total}] EMPTY")
            time.sleep(interval_seconds)

        if snapshots:
            full = pd.concat(snapshots, ignore_index=True)
            path = out_dir / f"{today}.parquet"
            full.to_parquet(path, index=False, engine="pyarrow")
            print(f"[Live] Saved {len(full)} rows to {path}")
            return full

        print("[Live] No data collected")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: NSE BHAVCOPY (EOD, FREE, FULL HISTORY)
# ═══════════════════════════════════════════════════════════════════════════════

class BhavCopyDownloader:
    """
    Downloads NSE F&O bhavcopy (end-of-day settlement data).
    Contains: all traded F&O contracts with OHLC, close, OI, volume.
    Free, no auth needed. Available from ~2020 onwards.
    """
    # New-format URL (2022+)
    BASE_NEW = "https://nsearchives.nseindia.com/content/fo"
    # Legacy format (pre-2022)
    BASE_LEGACY = "https://archives.nseindia.com/content/historical/DERIVATIVES"

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def download_day(self, dt: date) -> pd.DataFrame:
        """Download FNO bhavcopy for a single trading day."""
        cached = CACHE / "bhavcopy" / f"{dt.isoformat()}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)

        df = self._try_new_format(dt)
        if df is None or df.empty:
            df = self._try_legacy_format(dt)

        if df is not None and not df.empty:
            cached.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cached, index=False, engine="pyarrow")
            return df

        return pd.DataFrame()

    def _try_new_format(self, dt: date) -> Optional[pd.DataFrame]:
        """Try new bhavcopy format (2022+)."""
        fname = f"BhavCopy_NSE_FO_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip"
        url = f"{self.BASE_NEW}/{fname}"
        try:
            r = requests.get(url, timeout=30, headers=self.HEADERS)
            if r.status_code != 200:
                return None
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name))
            # Filter options only
            if "FinInstrmTp" in df.columns:
                df = df[df["FinInstrmTp"].isin(["OPTIDX", "OPTSTK"])].copy()
            df["date"] = dt
            return df
        except Exception as e:
            print(f"[Bhavcopy] New format failed for {dt}: {e}")
            return None

    def _try_legacy_format(self, dt: date) -> Optional[pd.DataFrame]:
        """Try legacy bhavcopy format (pre-2022)."""
        month_str = dt.strftime("%b").upper()
        fname = f"fo{dt.strftime('%d')}{month_str}{dt.strftime('%Y')}bhav.csv.zip"
        url = f"{self.BASE_LEGACY}/{dt.year}/{month_str}/{fname}"
        try:
            r = requests.get(url, timeout=30, headers=self.HEADERS)
            if r.status_code != 200:
                return None
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name))
            if "INSTRUMENT" in df.columns:
                df = df[df["INSTRUMENT"].isin(["OPTIDX", "OPTSTK"])].copy()
            df["date"] = dt
            return df
        except Exception as e:
            print(f"[Bhavcopy] Legacy format failed for {dt}: {e}")
            return None

    def download_range(self, start: date, end: date,
                       symbol_filter: str = None) -> pd.DataFrame:
        """
        Download bhavcopies for a full date range, skipping weekends.
        Optionally filter by underlying symbol (e.g., "NIFTY").
        """
        all_dfs: List[pd.DataFrame] = []
        current = start
        success = 0
        skipped = 0

        while current <= end:
            if current.weekday() < 5:  # skip weekends
                print(f"[Bhavcopy] Downloading {current}...", end=" ")
                df = self.download_day(current)
                if not df.empty:
                    if symbol_filter:
                        # Filter by underlying — handle both formats
                        sym_col = None
                        for col in ("TckrSymb", "SYMBOL"):
                            if col in df.columns:
                                sym_col = col
                                break
                        if sym_col:
                            df = df[df[sym_col].str.contains(
                                symbol_filter, case=False, na=False)]
                    all_dfs.append(df)
                    success += 1
                    print(f"OK ({len(df)} rows)")
                else:
                    skipped += 1
                    print("skipped (holiday or missing)")
                time.sleep(0.5)  # be respectful to NSE servers
            current += timedelta(days=1)

        print(f"\n[Bhavcopy] Downloaded {success} days, skipped {skipped}")

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            out = CACHE / "bhavcopy" / "combined.parquet"
            result.to_parquet(out, index=False, engine="pyarrow")
            print(f"[Bhavcopy] Saved {len(result)} total rows to {out}")
            return result

        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: BREEZE API (3 YEARS 1-MIN HISTORY, FREE WITH ICICI ACCOUNT)
# ═══════════════════════════════════════════════════════════════════════════════

class BreezeDownloader:
    """
    Downloads historical 1-minute OHLCV for option contracts via ICICI Breeze API.
    
    Setup:
      1. Open free ICICIDirect demat account
      2. Login to https://api.icicidirect.com/apiuser/login
      3. Create an App → get API_KEY and SECRET_KEY
      4. Generate SESSION_TOKEN daily (or automate with TOTP)
      5. pip install breeze-connect
    """

    def __init__(self, api_key: str = None, secret: str = None,
                 session_token: str = None):
        self.configured = all([api_key, secret, session_token])
        self.breeze = None
        if self.configured:
            try:
                from breeze_connect import BreezeConnect
                self.breeze = BreezeConnect(api_key=api_key)
                self.breeze.generate_session(
                    api_secret=secret, session_token=session_token)
                print("[Breeze] Connected successfully")
            except ImportError:
                print("[Breeze] breeze-connect not installed: pip install breeze-connect")
                self.configured = False
            except Exception as e:
                print(f"[Breeze] Connection failed: {e}")
                self.configured = False

    def get_option_history(
        self,
        symbol: str,
        expiry: date,
        strike: int,
        right: str,
        start: date,
        end: date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        """
        Get 1-minute OHLCV for a specific option contract.
        
        Args:
            symbol: e.g., "NIFTY"
            expiry: expiry date
            strike: strike price (e.g., 22000)
            right: "call" or "put"
            start/end: date range
            interval: "1minute", "5minute", "30minute", "1day"
        """
        if not self.configured:
            print("[Breeze] Not configured. See class docstring for setup.")
            return pd.DataFrame()

        cache_key = f"{symbol}_{expiry}_{strike}_{right}_{interval}"
        cached = CACHE / "breeze" / f"{cache_key}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)

        try:
            resp = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=start.strftime("%Y-%m-%dT07:00:00.000Z"),
                to_date=end.strftime("%Y-%m-%dT07:00:00.000Z"),
                stock_code=symbol,
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry.strftime("%Y-%m-%dT07:00:00.000Z"),
                right=right,
                strike_price=str(strike),
            )

            if resp.get("Status") == 200 and resp.get("Success"):
                df = pd.DataFrame(resp["Success"])
                cached.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cached, index=False, engine="pyarrow")
                return df
            else:
                err = resp.get("Error", "Unknown error")
                print(f"[Breeze] API error: {err}")
                return pd.DataFrame()
        except Exception as e:
            print(f"[Breeze] Request failed: {e}")
            return pd.DataFrame()

    def build_atm_dataset(
        self,
        symbol: str,
        start: date,
        end: date,
        n_strikes: int = 5,
    ) -> pd.DataFrame:
        """
        Download 1-min OHLCV for near-ATM strikes across all expiries.
        
        For NIFTY: n_strikes=5 gives 5 strikes each side of ATM = 10 CE + 10 PE.
        Strike spacing: NIFTY=50, BANKNIFTY=100.
        """
        if not self.configured:
            print("[Breeze] Not configured. See class docstring.")
            return pd.DataFrame()

        strike_spacing = 50 if "NIFTY" in symbol and "BANK" not in symbol else 100

        # Use bhavcopy to find all expiries and approximate ATM
        bhavcopy = BhavCopyDownloader()
        eod_data = bhavcopy.download_range(start, end, symbol_filter=symbol)
        if eod_data.empty:
            print("[Breeze] No bhavcopy data to determine expiries")
            return pd.DataFrame()

        # Find expiry column
        expiry_col = None
        for col in ("XpryDt", "EXPIRY_DT"):
            if col in eod_data.columns:
                expiry_col = col
                break
        if expiry_col is None:
            print("[Breeze] Cannot find expiry column in bhavcopy")
            return pd.DataFrame()

        # Find underlying price column
        price_col = None
        for col in ("UndrlygPric", "CLOSE"):
            if col in eod_data.columns:
                price_col = col
                break

        expiries = eod_data[expiry_col].unique()
        all_dfs: List[pd.DataFrame] = []
        total_contracts = 0

        for expiry_str in expiries:
            try:
                expiry_dt = pd.to_datetime(expiry_str).date()
            except Exception:
                continue

            cycle_data = eod_data[eod_data[expiry_col] == expiry_str]
            if cycle_data.empty or price_col is None:
                continue

            spot_approx = float(cycle_data[price_col].iloc[0])
            atm = round(spot_approx / strike_spacing) * strike_spacing
            strikes = [atm + i * strike_spacing
                       for i in range(-n_strikes, n_strikes + 1)]

            for strike_val in strikes:
                for right in ("call", "put"):
                    tag = f"{symbol} {expiry_dt} {strike_val}{right[0].upper()}E"
                    print(f"  [Breeze] Downloading {tag}...")
                    df = self.get_option_history(
                        symbol, expiry_dt, strike_val, right, start, end)
                    if not df.empty:
                        df["symbol"] = symbol
                        df["expiry"] = expiry_dt
                        df["strike"] = strike_val
                        df["right"] = right
                        all_dfs.append(df)
                        total_contracts += 1
                    time.sleep(0.35)  # rate limit: ~100 calls/min

        print(f"[Breeze] Downloaded {total_contracts} contracts")

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            out = CACHE / "breeze" / f"{symbol}_{start}_{end}_atm.parquet"
            result.to_parquet(out, index=False, engine="pyarrow")
            print(f"[Breeze] Saved {len(result)} rows to {out}")
            return result

        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# PCP DEVIATION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pcp_deviation(
    chain_df: pd.DataFrame,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """
    Compute put-call parity deviation for each strike in the chain.
    
    PCP: C - P = S - K * e^(-rT)
    Deviation = (C - P) - (S - K * e^(-rT))
    
    Large deviations indicate potential arbitrage opportunity.
    """
    if chain_df.empty:
        return pd.DataFrame()

    calls = chain_df[chain_df["type"] == "CE"].copy()
    puts = chain_df[chain_df["type"] == "PE"].copy()

    merge_cols = ["timestamp", "symbol", "expiry", "strike"]
    value_cols = ["ltp", "bid", "ask", "oi", "volume", "iv", "spot"]

    # Only keep columns that exist
    call_cols = [c for c in merge_cols + value_cols if c in calls.columns]
    put_cols = [c for c in merge_cols + [c for c in value_cols if c != "spot"]
                if c in puts.columns]

    merged = pd.merge(
        calls[call_cols],
        puts[put_cols],
        on=[c for c in merge_cols if c in calls.columns and c in puts.columns],
        suffixes=("_call", "_put"),
    )

    if merged.empty:
        return pd.DataFrame()

    # Time to expiry in years
    spot_col = "spot" if "spot" in merged.columns else "spot_call"
    merged["expiry_dt"] = pd.to_datetime(merged["expiry"], errors="coerce")
    merged["timestamp_dt"] = pd.to_datetime(merged["timestamp"])
    merged["T"] = (
        (merged["expiry_dt"] - merged["timestamp_dt"]).dt.total_seconds()
        / (365.25 * 86400)
    )
    merged["T"] = merged["T"].clip(lower=1 / 365)

    # PCP theoretical: S - K * e^(-rT)
    spot = merged[spot_col]
    merged["pcp_theoretical"] = spot - merged["strike"] * np.exp(
        -risk_free_rate * merged["T"]
    )

    # PCP actual: C - P (using mid prices where available)
    if "bid_call" in merged.columns and "ask_call" in merged.columns:
        merged["call_mid"] = (merged["bid_call"] + merged["ask_call"]) / 2
        merged["put_mid"] = (merged["bid_put"] + merged["ask_put"]) / 2
        # Fall back to LTP if bid/ask are zero
        merged.loc[merged["call_mid"] == 0, "call_mid"] = merged.loc[
            merged["call_mid"] == 0, "ltp_call"
        ]
        merged.loc[merged["put_mid"] == 0, "put_mid"] = merged.loc[
            merged["put_mid"] == 0, "ltp_put"
        ]
    else:
        merged["call_mid"] = merged["ltp_call"]
        merged["put_mid"] = merged["ltp_put"]

    merged["pcp_actual"] = merged["call_mid"] - merged["put_mid"]
    merged["pcp_deviation"] = merged["pcp_actual"] - merged["pcp_theoretical"]
    merged["pcp_deviation_pct"] = merged["pcp_deviation"] / spot * 100

    return merged.sort_values("pcp_deviation_pct", key=abs, ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NSE F&O Data Collector — 3-layer pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "bhavcopy", "breeze", "pcp"],
        required=True,
        help="Data collection mode",
    )
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=60,
                        help="Scrape interval in seconds (live mode)")
    parser.add_argument("--duration", type=int, default=375,
                        help="Scrape duration in minutes (live mode)")
    parser.add_argument("--breeze-key", default=None, help="Breeze API key")
    parser.add_argument("--breeze-secret", default=None, help="Breeze secret")
    parser.add_argument("--breeze-token", default=None, help="Breeze session token")
    args = parser.parse_args()

    if args.mode == "live":
        scraper = NSELiveScraper()
        # Single snapshot
        chain = scraper.get_option_chain(args.symbol)
        if chain and "chain" in chain and not chain["chain"].empty:
            print(f"\n[Live] Got {len(chain['chain'])} rows, spot={chain['spot']}")
            print(f"[Live] Expiries: {chain['expiry_dates'][:5]}")

            deviations = compute_pcp_deviation(chain["chain"])
            if not deviations.empty:
                print("\nTop PCP deviations right now:")
                display_cols = [c for c in [
                    "strike", "expiry", "pcp_deviation_pct",
                    "call_mid", "put_mid", "pcp_theoretical"
                ] if c in deviations.columns]
                print(deviations[display_cols].head(10).to_string(index=False))
        else:
            print("[Live] No data received (market may be closed)")

        # Uncomment to run full-day scraper:
        # scraper.scrape_to_parquet(args.symbol, args.interval, args.duration)

    elif args.mode == "bhavcopy":
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        dl = BhavCopyDownloader()
        df = dl.download_range(start, end, symbol_filter=args.symbol)
        if not df.empty:
            print(f"\nShape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nSample data:")
            print(df.head(5).to_string())
        else:
            print("No data downloaded")

    elif args.mode == "breeze":
        dl = BreezeDownloader(
            api_key=args.breeze_key or "YOUR_API_KEY",
            secret=args.breeze_secret or "YOUR_SECRET_KEY",
            session_token=args.breeze_token or "YOUR_SESSION_TOKEN",
        )
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        df = dl.build_atm_dataset(args.symbol, start, end)
        if not df.empty:
            print(f"\nDownloaded {len(df)} rows of 1-min options data")
            print(f"Columns: {list(df.columns)}")

    elif args.mode == "pcp":
        live_dir = CACHE / "live" / args.symbol
        files = sorted(live_dir.glob("*.parquet")) if live_dir.exists() else []
        if not files:
            print("No saved data. Run --mode live first during market hours.")
        else:
            df = pd.read_parquet(files[-1])
            deviations = compute_pcp_deviation(df)
            if not deviations.empty:
                display_cols = [c for c in [
                    "timestamp", "strike", "expiry", "pcp_deviation_pct",
                    "call_mid", "put_mid"
                ] if c in deviations.columns]
                print(deviations[display_cols].head(20).to_string(index=False))
            else:
                print("No deviations computed")
