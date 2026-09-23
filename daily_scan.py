"""台股雷達 PRO - 每日自動雷達
由 GitHub Actions 執行，產生 radar_cache.json。
不修改 Streamlit 介面；網站只讀取最新快取即可快速顯示每日雷達。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data_sources import (
    institutional_bulk_summary,
    revenue_for,
    revenue_table,
    stock_list_all,
)
from kline_engine import bearish_patterns, detect_patterns, prepare, score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "radar_cache.json"
LIMIT = int(os.getenv("RADAR_LIMIT", "80"))
WORKERS = int(os.getenv("RADAR_WORKERS", "8"))


def price_history(code: str) -> pd.DataFrame:
    for suffix in [".TW", ".TWO"]:
        try:
            d = yf.Ticker(f"{code}{suffix}").history(
                period="1y", interval="1d", auto_adjust=False, timeout=10
            )
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                return d.dropna()
        except Exception:
            pass
    return pd.DataFrame()


def analyze_one(code: str, name: str, inst_map: dict, bulk_inst: dict, rev_df: pd.DataFrame):
    raw = price_history(code)
    if raw.empty or len(raw) < 60:
        return None
    try:
        d = prepare(raw)
        bullish = detect_patterns(d)
        bearish = bearish_patterns(d)
        inst = inst_map.get(code, {})
        inst_stats = bulk_inst.get(code, {})
        rev = revenue_for(code, rev_df)
        s = score(
            d,
            bullish,
            bearish,
            inst.get("法人合計", np.nan),
            inst_stats.get("5日", np.nan),
            inst_stats.get("20日", np.nan),
            rev.get("YoY", np.nan),
        )
        last = d.iloc[-1]
        prev_close = d["Close"].iloc[-2]
        change_pct = (float(last.Close) / float(prev_close) - 1) * 100 if prev_close else np.nan
        return {
            "代號": code,
            "名稱": name,
            "收盤": round(float(last.Close), 2),
            "漲跌%": round(float(change_pct), 2),
            "法人買賣超(股)": clean_num(inst.get("法人合計")),
            "法人5日(股)": clean_num(inst_stats.get("5日")),
            "法人20日(股)": clean_num(inst_stats.get("20日")),
            "RSI": clean_num(last.RSI),
            "雷達分數": int(s["分數"]),
            "判斷": s["訊號"],
            "K線訊號": "、".join((bullish[:3] + bearish[:2])),
        }
    except Exception:
        return None


def clean_num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def main():
    stocks = stock_list_all()
    if stocks.empty:
        raise RuntimeError("無法取得台股清單")

    pool = stocks.copy()
    if "成交金額" in pool.columns:
        pool["成交金額"] = pd.to_numeric(pool["成交金額"], errors="coerce")
        pool = pool.sort_values("成交金額", ascending=False, na_position="last")
    pool = pool.head(LIMIT)
    codes = pool["代號"].astype(str).tolist()
    names = dict(zip(pool["代號"].astype(str), pool["名稱"].astype(str)))

    # 法人與營收只各抓一次，避免逐檔重複呼叫官方 API。
    inst_map = {}
    try:
        from data_sources import latest_institutional
        inst_map, inst_date = latest_institutional()
    except Exception:
        inst_date = None

    bulk_inst = institutional_bulk_summary(codes, days=20)
    rev_df = revenue_table()

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(analyze_one, code, names.get(code, ""), inst_map, bulk_inst, rev_df): code
            for code in codes
        }
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                results.append(row)

    results.sort(key=lambda x: x.get("雷達分數", -1), reverse=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "generated_at": now,
        "source": "GitHub Actions daily scan",
        "candidate_count": len(codes),
        "success_count": len(results),
        "institution_date": inst_date,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"寫入 {OUT}: {len(results)}/{len(codes)} 檔")


if __name__ == "__main__":
    main()
