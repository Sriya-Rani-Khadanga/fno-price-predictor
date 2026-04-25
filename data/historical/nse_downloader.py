"""
NSE bhavcopy downloader and option chain snapshot fetcher.
Downloads real NSE data from public endpoints.
"""
from __future__ import annotations
import io
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests
from data.processors.options_chain import OptionChain, StrikeData
from config.settings import get_settings, CACHE_DIR

class NSEDownloader:
    """Downloads NSE bhavcopy (EOD) data and live option chain snapshots."""

    BASE_URL = "https://www.nseindia.com"
    BHAVCOPY_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/fo{ddmmmyyyy}bhav.csv.zip"
    OPTION_CHAIN_INDICES = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    OPTION_CHAIN_EQUITIES = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._cookies_valid = False
        self._last_cookie_refresh = None
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _refresh_session(self):
        """Hit NSE homepage to get valid session cookies."""
        try:
            resp = self._session.get(self.BASE_URL, timeout=10)
            resp.raise_for_status()
            self._cookies_valid = True
            self._last_cookie_refresh = datetime.now()
        except Exception as e:
            print(f"[NSEDownloader] Cookie refresh failed: {e}")
            self._cookies_valid = False

    def _ensure_cookies(self):
        """Ensure we have valid cookies, refresh if needed."""
        if not self._cookies_valid or self._last_cookie_refresh is None:
            self._refresh_session()
            return
        if (datetime.now() - self._last_cookie_refresh).total_seconds() > 300:
            self._refresh_session()

    def download_bhavcopy(self, dt: date) -> Optional[pd.DataFrame]:
        """Download NSE F&O bhavcopy for a given date. Returns DataFrame or None."""
        cache_path = self._cache_dir / f"bhavcopy_{dt.isoformat()}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        self._ensure_cookies()
        month_map = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
                     7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}
        url = self.BHAVCOPY_URL.format(
            year=dt.year, month=month_map[dt.month],
            ddmmmyyyy=dt.strftime("%d%b%Y").upper())
        try:
            time.sleep(0.5)
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df.to_parquet(cache_path)
            return df
        except Exception as e:
            print(f"[NSEDownloader] Bhavcopy download failed for {dt}: {e}")
            return None

    def download_option_chain_snapshot(self, underlying: str) -> Optional[OptionChain]:
        """Download live option chain snapshot from NSE. Works during market hours."""
        self._ensure_cookies()
        settings = get_settings()
        inst = settings.instruments.get(underlying)
        if inst and inst.option_chain_url:
            url = inst.option_chain_url
        elif underlying in ("NIFTY", "BANKNIFTY"):
            url = self.OPTION_CHAIN_INDICES.format(symbol=underlying)
        else:
            url = self.OPTION_CHAIN_EQUITIES.format(symbol=underlying)
        try:
            time.sleep(0.5)
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_nse_option_chain(data, underlying)
        except Exception as e:
            print(f"[NSEDownloader] Option chain snapshot failed for {underlying}: {e}")
            return None

    def _parse_nse_option_chain(self, data: dict, underlying: str) -> Optional[OptionChain]:
        """Parse NSE API JSON response into OptionChain dataclass."""
        try:
            records = data.get("records", {})
            chain_data = records.get("data", [])
            spot = records.get("underlyingValue", 0.0)
            expiry_dates = records.get("expiryDates", [])
            nearest_expiry = expiry_dates[0] if expiry_dates else "UNKNOWN"
            filtered = [r for r in chain_data if r.get("expiryDate") == nearest_expiry]
            strikes_list = []
            for row in filtered:
                ce = row.get("CE", {})
                pe = row.get("PE", {})
                strike = row.get("strikePrice", 0.0)
                strikes_list.append(StrikeData(
                    strike=strike,
                    call_bid=ce.get("bidprice", 0.0),
                    call_ask=ce.get("askPrice", 0.0),
                    call_ltp=ce.get("lastPrice", 0.0),
                    call_oi=int(ce.get("openInterest", 0)),
                    call_volume=int(ce.get("totalTradedVolume", 0)),
                    call_iv=ce.get("impliedVolatility", 0.0) / 100.0 if ce.get("impliedVolatility") else 0.15,
                    put_bid=pe.get("bidprice", 0.0),
                    put_ask=pe.get("askPrice", 0.0),
                    put_ltp=pe.get("lastPrice", 0.0),
                    put_oi=int(pe.get("openInterest", 0)),
                    put_volume=int(pe.get("totalTradedVolume", 0)),
                    put_iv=pe.get("impliedVolatility", 0.0) / 100.0 if pe.get("impliedVolatility") else 0.15,
                ))
            exp_str = nearest_expiry.replace("-", "")[:7] if isinstance(nearest_expiry, str) else nearest_expiry
            return OptionChain(
                underlying=underlying, expiry=str(exp_str), spot_price=spot,
                spot_bid=spot * 0.9999, spot_ask=spot * 1.0001,
                timestamp=datetime.now(), strikes=strikes_list, data_source="live")
        except Exception as e:
            print(f"[NSEDownloader] Parse error: {e}")
            return None

    def download_historical_chain(self, underlying: str, dt: date) -> List[OptionChain]:
        """Reconstruct option chains from bhavcopy data for a given date."""
        df = self.download_bhavcopy(dt)
        if df is None or df.empty:
            return []
        if "SYMBOL" in df.columns:
            sym_col = "SYMBOL"
        elif "TckrSymb" in df.columns:
            sym_col = "TckrSymb"
        else:
            sym_col = df.columns[0]
        sym_filter = underlying
        if underlying == "NIFTY":
            sym_filter = "NIFTY"
            mask = df[sym_col].str.contains("NIFTY", case=False, na=False)
            mask = mask & ~df[sym_col].str.contains("BANKNIFTY", case=False, na=False)
        elif underlying == "BANKNIFTY":
            mask = df[sym_col].str.contains("BANKNIFTY", case=False, na=False)
        else:
            mask = df[sym_col].str.contains(underlying, case=False, na=False)
        opt_df = df[mask].copy()
        if "OPTION_TYP" in opt_df.columns:
            typ_col = "OPTION_TYP"
        elif "OptnTp" in opt_df.columns:
            typ_col = "OptnTp"
        else:
            return []
        opt_df = opt_df[opt_df[typ_col].isin(["CE", "PE"])]
        if opt_df.empty:
            return []
        if "EXPIRY_DT" in opt_df.columns:
            exp_col = "EXPIRY_DT"
        elif "XpryDt" in opt_df.columns:
            exp_col = "XpryDt"
        else:
            exp_col = "EXPIRY_DT"
        chains = []
        for expiry, group in opt_df.groupby(exp_col):
            strikes = []
            strike_col = "STRIKE_PR" if "STRIKE_PR" in group.columns else "StrkPric"
            close_col = "CLOSE" if "CLOSE" in group.columns else "ClsPric"
            oi_col = "OPEN_INT" if "OPEN_INT" in group.columns else "OpnIntrst"
            vol_col = "CONTRACTS" if "CONTRACTS" in group.columns else "TtlTradgVol"
            for strike_price, sgroup in group.groupby(strike_col):
                ce_row = sgroup[sgroup[typ_col] == "CE"]
                pe_row = sgroup[sgroup[typ_col] == "PE"]
                if ce_row.empty or pe_row.empty:
                    continue
                ce = ce_row.iloc[0]
                pe = pe_row.iloc[0]
                close_ce = float(ce.get(close_col, 0))
                close_pe = float(pe.get(close_col, 0))
                oi_ce = int(ce.get(oi_col, 0))
                oi_pe = int(pe.get(oi_col, 0))
                vol_ce = int(ce.get(vol_col, 0))
                vol_pe = int(pe.get(vol_col, 0))
                strikes.append(StrikeData(
                    strike=float(strike_price), call_bid=close_ce * 0.99, call_ask=close_ce * 1.01,
                    call_ltp=close_ce, call_oi=oi_ce, call_volume=vol_ce, call_iv=0.15,
                    put_bid=close_pe * 0.99, put_ask=close_pe * 1.01, put_ltp=close_pe,
                    put_oi=oi_pe, put_volume=vol_pe, put_iv=0.15))
            if strikes:
                spot_est = strikes[len(strikes) // 2].strike
                chains.append(OptionChain(
                    underlying=underlying, expiry=str(expiry),
                    spot_price=spot_est, spot_bid=spot_est * 0.9999,
                    spot_ask=spot_est * 1.0001,
                    timestamp=datetime.combine(dt, datetime.min.time().replace(hour=15, minute=30)),
                    strikes=sorted(strikes, key=lambda s: s.strike),
                    data_source="historical"))
        return chains

    def download_date_range(self, start: date, end: date, callback=None) -> int:
        """Download bhavcopies for a date range. Returns count of successful downloads."""
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5:
                result = self.download_bhavcopy(current)
                if result is not None:
                    count += 1
                if callback:
                    callback(current, result is not None)
            current += timedelta(days=1)
        return count
