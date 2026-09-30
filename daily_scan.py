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
LIMIT = int(os.getenv("RADAR_LIMIT", "200"))
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
            "法人買賣超(張)": clean_num(inst.get("法人合計"))/1000 if clean_num(inst.get("法人合計")) is not None else None,
            "法人5日(張)": clean_num(inst_stats.get("5日"))/1000 if clean_num(inst_stats.get("5日")) is not None else None,
            "法人20日(張)": clean_num(inst_stats.get("20日"))/1000 if clean_num(inst_stats.get("20日")) is not None else None,
            "RSI": clean_num(last.RSI),
            "雷達分數": int(s["分數"]),
            "早期趨勢分": int(s.get("早期趨勢分", 0)),
            "早期條件數": int(s.get("早期條件數", 0)),
            "60日位階%": round(s.get("60日位階%", np.nan), 1) if pd.notna(s.get("60日位階%", np.nan)) else None,
            "位置距MA20%": round(s.get("位置距MA20%", np.nan), 2) if pd.notna(s.get("位置距MA20%", np.nan)) else None,
            "20日漲幅%": round(s.get("20日漲幅%", np.nan), 2) if pd.notna(s.get("20日漲幅%", np.nan)) else None,
            "進場區": f"{s.get("進場區下緣", np.nan):.2f}～{s.get("進場區上緣", np.nan):.2f}" if pd.notna(s.get("進場區下緣", np.nan)) and pd.notna(s.get("進場區上緣", np.nan)) else "—",
            "判斷": s["訊號"],
            "動作": s["動作"],
            "進場參考": round(s["進場參考"],2),
            "停損": round(s["停損參考"],2),
            "目標1": round(s["目標1"],2),
            "目標2": round(s["目標2"],2),
            "風險報酬": round(s["風險報酬"],2) if pd.notna(s["風險報酬"]) else None,
            "K線訊號": "、".join((bullish[:3] + bearish[:2])),
            "K線出場警戒": "、".join(bearish[:3]) if bearish else "—",
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

    signal_order={"🟢 早期佈局":0,"🟢 買進條件成立":1,"🔵 突破確認":2,"🟡 等待":3,"🟠 不追高":4,"🔴 不買":5}
    results.sort(key=lambda x: (signal_order.get(x.get("判斷"),99), -float(x.get("早期趨勢分",0)), -float(x.get("雷達分數",-1))))
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
