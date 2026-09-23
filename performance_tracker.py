"""交易訊號績效追蹤
每天在收盤後執行：把當日雷達訊號記錄下來，並自動補齊 1/5/20 個交易日績效。
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "radar_cache.json"
HISTORY = ROOT / "signal_history.json"
SUMMARY = ROOT / "performance_summary.json"
WORKERS = 8


def price_history(code: str) -> pd.DataFrame:
    for suffix in [".TW", ".TWO"]:
        try:
            d = yf.Ticker(f"{code}{suffix}").history(period="6mo", interval="1d", auto_adjust=False, timeout=10)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d.index = pd.to_datetime(d.index).tz_localize(None)
                return d.dropna(subset=["Close"])
        except Exception:
            pass
    return pd.DataFrame()


def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def main():
    payload = load_json(CACHE, {})
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not rows:
        print("沒有當日雷達資料，略過績效追蹤")
        return

    history = load_json(HISTORY, [])
    if not isinstance(history, list):
        history = []

    # 今日訊號只記一次；只追蹤可形成交易決策的訊號。
    now = datetime.now().astimezone()
    today = now.date().isoformat()
    existing = {(str(x.get("日期")), str(x.get("代號"))) for x in history}
    for r in rows:
        action = str(r.get("判斷", ""))
        if action not in {"🟢 買進條件成立", "🔵 突破確認", "🟡 等待", "🟠 不追高", "🔴 不買"}:
            continue
        key = (today, str(r.get("代號", "")))
        if key in existing:
            continue
        try:
            entry = float(r.get("收盤"))
        except Exception:
            continue
        history.append({
            "日期": today,
            "代號": str(r.get("代號", "")),
            "名稱": str(r.get("名稱", "")),
            "判斷": action,
            "分數": int(r.get("雷達分數", 0)),
            "動作": str(r.get("動作", "")),
            "訊號價格": entry,
            "進場參考": r.get("進場參考"),
            "停損": r.get("停損"),
            "目標1": r.get("目標1"),
            "目標2": r.get("目標2"),
            "風險報酬": r.get("風險報酬"),
            "1日報酬%": None,
            "5日報酬%": None,
            "20日報酬%": None,
        })

    # 只對尚未完成 1/5/20 日評估的股票抓歷史價格。
    pending_codes = {str(x.get("代號")) for x in history if any(x.get(k) is None for k in ["1日報酬%", "5日報酬%", "20日報酬%"]) }
    prices = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(price_history, c): c for c in pending_codes if c}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                prices[c] = fut.result()
            except Exception:
                prices[c] = pd.DataFrame()

    for rec in history:
        code = str(rec.get("代號", ""))
        d = prices.get(code)
        if d is None or d.empty:
            continue
        try:
            entry_date = pd.Timestamp(rec["日期"])
            entry_price = float(rec["訊號價格"])
        except Exception:
            continue
        idx = d.index[d.index.normalize() >= entry_date]
        if len(idx) == 0:
            continue
        # 訊號日的收盤視為基準；往後第 N 個交易日的收盤計算績效。
        base_pos = d.index.get_loc(idx[0])
        for n, key in [(1, "1日報酬%"), (5, "5日報酬%"), (20, "20日報酬%")]:
            if rec.get(key) is not None:
                continue
            target_pos = base_pos + n
            if target_pos < len(d):
                close = float(d.iloc[target_pos]["Close"])
                rec[key] = round((close / entry_price - 1) * 100, 2)

    # 摘要：只對已經有對應觀察值的訊號計算，避免未到期訊號稀釋結果。
    summary = {"updated_at": now.isoformat(timespec="seconds"), "total_signals": len(history), "by_signal": {}, "by_score": {}}
    for key in ["1日報酬%", "5日報酬%", "20日報酬%"]:
        vals = [float(x[key]) for x in history if x.get(key) is not None]
        summary[key] = {
            "count": len(vals),
            "positive_rate": round(sum(v > 0 for v in vals) / len(vals) * 100, 2) if vals else None,
            "avg_return": round(float(np.mean(vals)), 2) if vals else None,
        }
    for sig in ["🟢 買進條件成立", "🔵 突破確認", "🟡 等待", "🟠 不追高", "🔴 不買"]:
        subset = [x for x in history if x.get("判斷") == sig]
        vals = [float(x["5日報酬%"]) for x in subset if x.get("5日報酬%") is not None]
        summary["by_signal"][sig] = {"count": len(subset), "5d_count": len(vals), "5d_positive_rate": round(sum(v > 0 for v in vals)/len(vals)*100,2) if vals else None, "5d_avg_return": round(float(np.mean(vals)),2) if vals else None}
    for low, high in [(90,101),(80,90),(70,80),(0,70)]:
        subset = [x for x in history if low <= int(x.get("分數",0)) < high]
        vals = [float(x["5日報酬%"]) for x in subset if x.get("5日報酬%") is not None]
        summary["by_score"][f"{low}-{high-1}"] = {"count":len(subset),"5d_count":len(vals),"5d_positive_rate":round(sum(v>0 for v in vals)/len(vals)*100,2) if vals else None,"5d_avg_return":round(float(np.mean(vals)),2) if vals else None}

    HISTORY.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"績效追蹤完成：{len(history)} 筆訊號")


if __name__ == "__main__":
    main()
