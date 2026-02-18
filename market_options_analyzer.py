#!/usr/bin/env python3
"""
Market Options Analyzer
-----------------------
Analyzes stock options chains for high-volume, liquid tickers
and identifies potential opportunities based on technical signals,
implied volatility, and volume/open interest ratios.

DISCLAIMER: This tool is for EDUCATIONAL and INFORMATIONAL purposes only.
It does NOT constitute financial advice. Options trading involves substantial
risk of loss. Always consult a licensed financial advisor before trading.
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "budget": 5000,
    "watchlist": [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
        "META", "GOOGL", "AMD", "SPY", "QQQ",
        "NFLX", "COIN", "PLTR", "SOFI", "BAC"
    ],
    "max_days_to_expiry": 45,
    "min_days_to_expiry": 7,
    "min_volume": 100,
    "min_open_interest": 500,
    "max_premium_pct_of_budget": 0.10,
    "output_dir": "reports"
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            user = json.load(f)
        cfg = {**DEFAULT_CONFIG, **user}
    else:
        cfg = DEFAULT_CONFIG.copy()
    return cfg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str) -> dict | None:
    """Fetch current price, basic stats, and recent performance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        if hist.empty:
            return None

        info = t.info or {}
        current = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current
        month_ago = hist["Close"].iloc[0]

        return {
            "ticker": ticker,
            "price": round(current, 2),
            "daily_change_pct": round((current - prev_close) / prev_close * 100, 2),
            "monthly_change_pct": round((current - month_ago) / month_ago * 100, 2),
            "avg_volume": int(hist["Volume"].mean()),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "market_cap": info.get("marketCap", None),
            "sector": info.get("sector", "N/A"),
        }
    except Exception as e:
        print(f"  [WARN] Could not fetch data for {ticker}: {e}")
        return None


def analyse_options_chain(ticker: str, cfg: dict) -> list[dict]:
    """Scan the options chain and score each contract."""
    results = []
    try:
        t = yf.Ticker(ticker)
        expirations = t.options  # tuple of date strings
        if not expirations:
            return results

        stock_price = t.history(period="1d")["Close"].iloc[-1]
        now = datetime.now()

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - now).days
            if dte < cfg["min_days_to_expiry"] or dte > cfg["max_days_to_expiry"]:
                continue

            chain = t.option_chain(exp_str)

            for opt_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if df.empty:
                    continue

                df = df.copy()
                df["midPrice"] = (df["bid"] + df["ask"]) / 2
                df["contract_cost"] = df["midPrice"] * 100  # 1 contract = 100 shares

                # Filters
                mask = (
                    (df["volume"].fillna(0) >= cfg["min_volume"])
                    & (df["openInterest"].fillna(0) >= cfg["min_open_interest"])
                    & (df["contract_cost"] > 0)
                    & (df["contract_cost"] <= cfg["budget"])
                    & (df["contract_cost"] <= cfg["budget"] * cfg["max_premium_pct_of_budget"])
                )
                filtered = df[mask]

                for _, row in filtered.iterrows():
                    strike = row["strike"]
                    mid = round(row["midPrice"], 2)
                    cost = round(row["contract_cost"], 2)
                    iv = round(row.get("impliedVolatility", 0) * 100, 1)
                    vol = int(row.get("volume", 0))
                    oi = int(row.get("openInterest", 0))
                    vol_oi = round(vol / oi, 2) if oi > 0 else 0

                    # Distance from current price
                    if opt_type == "CALL":
                        otm_pct = round((strike - stock_price) / stock_price * 100, 2)
                    else:
                        otm_pct = round((stock_price - strike) / stock_price * 100, 2)

                    # Scoring heuristic (higher = more interesting, NOT better)
                    score = 0
                    # High volume relative to OI signals unusual activity
                    if vol_oi > 1.5:
                        score += 3
                    elif vol_oi > 0.8:
                        score += 1

                    # Moderate IV (not too expensive, not too dead)
                    if 30 < iv < 80:
                        score += 2
                    elif iv <= 30:
                        score += 1

                    # Reasonable OTM %
                    if 1 < otm_pct < 10:
                        score += 2
                    elif otm_pct <= 1:
                        score += 1  # near ATM

                    # More contracts affordable = better risk spread
                    contracts_affordable = int(cfg["budget"] // cost)
                    if contracts_affordable >= 5:
                        score += 2
                    elif contracts_affordable >= 2:
                        score += 1

                    results.append({
                        "ticker": ticker,
                        "type": opt_type,
                        "strike": strike,
                        "expiry": exp_str,
                        "dte": dte,
                        "mid_price": mid,
                        "cost_1_contract": cost,
                        "contracts_in_budget": contracts_affordable,
                        "total_cost": round(cost * contracts_affordable, 2),
                        "iv_pct": iv,
                        "volume": vol,
                        "open_interest": oi,
                        "vol_oi_ratio": vol_oi,
                        "otm_pct": otm_pct,
                        "score": score,
                        "stock_price": round(stock_price, 2),
                    })

    except Exception as e:
        print(f"  [WARN] Options chain error for {ticker}: {e}")

    return results


# ---------------------------------------------------------------------------
# Trend signal (simple moving-average based)
# ---------------------------------------------------------------------------

def get_trend_signal(ticker: str) -> str:
    """Return BULLISH / BEARISH / NEUTRAL based on SMA crossover."""
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if len(hist) < 50:
            return "NEUTRAL"
        sma_20 = hist["Close"].rolling(20).mean().iloc[-1]
        sma_50 = hist["Close"].rolling(50).mean().iloc[-1]
        price = hist["Close"].iloc[-1]

        if price > sma_20 > sma_50:
            return "BULLISH"
        elif price < sma_20 < sma_50:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(cfg: dict) -> str:
    """Main entry-point: scan watchlist, analyse, and produce a report."""
    budget = cfg["budget"]
    watchlist = cfg["watchlist"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"  MARKET OPTIONS ANALYZER — {now_str}")
    lines.append(f"  Budget: ${budget:,.2f}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("DISCLAIMER: This is NOT financial advice. Use at your own risk.")
    lines.append("")

    # --- Stock overview ---
    lines.append("-" * 80)
    lines.append("  WATCHLIST OVERVIEW")
    lines.append("-" * 80)

    stock_rows = []
    for sym in watchlist:
        print(f"  Fetching {sym} ...")
        d = get_stock_data(sym)
        if d:
            trend = get_trend_signal(sym)
            stock_rows.append([
                d["ticker"], f"${d['price']:.2f}",
                f"{d['daily_change_pct']:+.2f}%",
                f"{d['monthly_change_pct']:+.2f}%",
                d["sector"], trend
            ])

    lines.append(tabulate(
        stock_rows,
        headers=["Ticker", "Price", "Day %", "Month %", "Sector", "Trend"],
        tablefmt="simple_grid"
    ))
    lines.append("")

    # --- Options scan ---
    lines.append("-" * 80)
    lines.append("  OPTIONS SCAN (filtered by liquidity, premium, budget)")
    lines.append("-" * 80)

    all_options: list[dict] = []
    for sym in watchlist:
        print(f"  Scanning options for {sym} ...")
        opts = analyse_options_chain(sym, cfg)
        all_options.extend(opts)

    if not all_options:
        lines.append("  No options matched the current filters.")
        lines.append("  Try relaxing min_volume, min_open_interest, or max_premium_pct_of_budget.")
    else:
        df = pd.DataFrame(all_options)
        df = df.sort_values("score", ascending=False)

        # Top CALLs
        top_calls = df[df["type"] == "CALL"].head(10)
        if not top_calls.empty:
            lines.append("")
            lines.append("  >> TOP CALL OPTIONS (by score)")
            call_rows = []
            for _, r in top_calls.iterrows():
                call_rows.append([
                    r["ticker"], f"${r['stock_price']}", f"${r['strike']}",
                    r["expiry"], r["dte"],
                    f"${r['mid_price']}", f"${r['cost_1_contract']}",
                    r["contracts_in_budget"], f"${r['total_cost']}",
                    f"{r['iv_pct']}%", r["volume"], r["open_interest"],
                    r["vol_oi_ratio"], f"{r['otm_pct']}%", r["score"]
                ])
            lines.append(tabulate(
                call_rows,
                headers=["Tkr", "Price", "Strike", "Exp", "DTE",
                         "Mid", "Cost/1", "#Ctrs", "Total",
                         "IV", "Vol", "OI", "V/OI", "OTM%", "Score"],
                tablefmt="simple_grid"
            ))

        # Top PUTs
        top_puts = df[df["type"] == "PUT"].head(10)
        if not top_puts.empty:
            lines.append("")
            lines.append("  >> TOP PUT OPTIONS (by score)")
            put_rows = []
            for _, r in top_puts.iterrows():
                put_rows.append([
                    r["ticker"], f"${r['stock_price']}", f"${r['strike']}",
                    r["expiry"], r["dte"],
                    f"${r['mid_price']}", f"${r['cost_1_contract']}",
                    r["contracts_in_budget"], f"${r['total_cost']}",
                    f"{r['iv_pct']}%", r["volume"], r["open_interest"],
                    r["vol_oi_ratio"], f"{r['otm_pct']}%", r["score"]
                ])
            lines.append(tabulate(
                put_rows,
                headers=["Tkr", "Price", "Strike", "Exp", "DTE",
                         "Mid", "Cost/1", "#Ctrs", "Total",
                         "IV", "Vol", "OI", "V/OI", "OTM%", "Score"],
                tablefmt="simple_grid"
            ))

        # Unusual activity
        unusual = df[df["vol_oi_ratio"] > 2.0].sort_values("vol_oi_ratio", ascending=False).head(5)
        if not unusual.empty:
            lines.append("")
            lines.append("  >> UNUSUAL ACTIVITY (Volume/OI > 2.0)")
            ua_rows = []
            for _, r in unusual.iterrows():
                ua_rows.append([
                    r["ticker"], r["type"], f"${r['strike']}",
                    r["expiry"], r["volume"], r["open_interest"],
                    r["vol_oi_ratio"], f"${r['cost_1_contract']}"
                ])
            lines.append(tabulate(
                ua_rows,
                headers=["Tkr", "Type", "Strike", "Exp", "Vol", "OI", "V/OI", "Cost"],
                tablefmt="simple_grid"
            ))

    # --- Footer ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("  SCORING EXPLANATION")
    lines.append("  - Vol/OI > 1.5: +3 pts  (unusual activity)")
    lines.append("  - Vol/OI > 0.8: +1 pt")
    lines.append("  - IV 30-80%:    +2 pts  (moderate implied volatility)")
    lines.append("  - IV <= 30%:    +1 pt")
    lines.append("  - OTM 1-10%:    +2 pts  (reasonable out-of-the-money)")
    lines.append("  - OTM <= 1%:    +1 pt   (near at-the-money)")
    lines.append("  - Affordable 5+: +2 pts (can spread risk)")
    lines.append("  - Affordable 2+: +1 pt")
    lines.append("")
    lines.append("  Higher score = more interesting signals, NOT guaranteed profit.")
    lines.append("  ALWAYS do your own due diligence before trading.")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    report = generate_report(cfg)

    # Print to stdout
    print(report)

    # Save to file
    out_dir = Path(__file__).parent / cfg["output_dir"]
    out_dir.mkdir(exist_ok=True)
    filename = f"options_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
