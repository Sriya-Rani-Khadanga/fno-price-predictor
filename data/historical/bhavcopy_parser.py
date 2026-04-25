"""
Bhavcopy parser — transforms raw NSE bhavcopy CSV into structured option chain data.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Dict, List, Optional
import pandas as pd
from data.processors.options_chain import OptionChain, StrikeData

class BhavcopyParser:
    """Parses NSE F&O bhavcopy CSVs into OptionChain objects."""

    COLUMN_MAPPINGS = {
        "new": {"symbol": "TckrSymb", "expiry": "XpryDt", "strike": "StrkPric",
                "type": "OptnTp", "open": "OpnPric", "high": "HghPric",
                "low": "LwPric", "close": "ClsPric", "oi": "OpnIntrst",
                "volume": "TtlTradgVol"},
        "old": {"symbol": "SYMBOL", "expiry": "EXPIRY_DT", "strike": "STRIKE_PR",
                "type": "OPTION_TYP", "open": "OPEN", "high": "HIGH",
                "low": "LOW", "close": "CLOSE", "oi": "OPEN_INT",
                "volume": "CONTRACTS"},
    }

    def detect_format(self, df: pd.DataFrame) -> str:
        if "TckrSymb" in df.columns:
            return "new"
        return "old"

    def parse(self, df: pd.DataFrame, underlying: str, target_date: date) -> List[OptionChain]:
        fmt = self.detect_format(df)
        cols = self.COLUMN_MAPPINGS[fmt]
        sym_col = cols["symbol"]
        if underlying == "NIFTY":
            mask = df[sym_col].str.contains("NIFTY", case=False, na=False)
            mask = mask & ~df[sym_col].str.contains("BANKNIFTY", case=False, na=False)
        else:
            mask = df[sym_col].str.contains(underlying, case=False, na=False)
        opt_mask = df[cols["type"]].isin(["CE", "PE"])
        filtered = df[mask & opt_mask].copy()
        if filtered.empty:
            return []
        chains = []
        for expiry, group in filtered.groupby(cols["expiry"]):
            strikes = []
            for strike_price, sgroup in group.groupby(cols["strike"]):
                ce = sgroup[sgroup[cols["type"]] == "CE"]
                pe = sgroup[sgroup[cols["type"]] == "PE"]
                if ce.empty or pe.empty:
                    continue
                ce_r = ce.iloc[0]
                pe_r = pe.iloc[0]
                c_close = float(ce_r[cols["close"]])
                p_close = float(pe_r[cols["close"]])
                strikes.append(StrikeData(
                    strike=float(strike_price),
                    call_bid=c_close * 0.995, call_ask=c_close * 1.005,
                    call_ltp=c_close, call_oi=int(ce_r[cols["oi"]]),
                    call_volume=int(ce_r[cols["volume"]]), call_iv=0.15,
                    put_bid=p_close * 0.995, put_ask=p_close * 1.005,
                    put_ltp=p_close, put_oi=int(pe_r[cols["oi"]]),
                    put_volume=int(pe_r[cols["volume"]]), put_iv=0.15))
            if not strikes:
                continue
            strikes.sort(key=lambda s: s.strike)
            mid_idx = len(strikes) // 2
            spot_est = strikes[mid_idx].strike
            chains.append(OptionChain(
                underlying=underlying, expiry=str(expiry),
                spot_price=spot_est, spot_bid=spot_est * 0.9999,
                spot_ask=spot_est * 1.0001,
                timestamp=datetime.combine(target_date, datetime.min.time().replace(hour=15, minute=30)),
                strikes=strikes, data_source="historical"))
        return chains

    def extract_ohlc(self, df: pd.DataFrame, underlying: str) -> Dict[str, float]:
        """Extract OHLC data for spot reconstruction from futures data."""
        if df.empty:
            return {}
        fmt = self.detect_format(df)
        cols = self.COLUMN_MAPPINGS[fmt]
        sym_col = cols.get("symbol", "SYMBOL")
        if sym_col not in df.columns:
            return {}
        mask = df[sym_col].str.contains(underlying, case=False, na=False)
        type_col = cols["type"]
        fut_mask = df[type_col].isna() | (df[type_col] == "XX") | (df[type_col] == "")
        fut_df = df[mask & fut_mask]
        if fut_df.empty:
            mask2 = df[sym_col].str.contains(underlying, case=False, na=False)
            fut_df = df[mask2].head(1)
        if fut_df.empty:
            return {}
        row = fut_df.iloc[0]
        return {
            "open": float(row.get(cols["open"], 0)),
            "high": float(row.get(cols["high"], 0)),
            "low": float(row.get(cols["low"], 0)),
            "close": float(row.get(cols["close"], 0)),
        }
