import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/126 Safari/537.36"
}

def _get_json(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def twse_stock_list():
    data = _get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
    if not isinstance(data, list):
        return pd.DataFrame(columns=["代號", "名稱"])
    rows = []
    for x in data:
        code = str(x.get("Code", "")).strip()
        name = str(x.get("Name", "")).strip()
        if code.isdigit() and name:
            rows.append({"代號": code, "名稱": name})
    return pd.DataFrame(rows)

def _parse_t86(data):
    if not isinstance(data, dict):
        return {}
    fields = data.get("fields", [])
    rows = data.get("data", [])
    if not fields or not rows:
        return {}

    def idx(keys):
        for key in keys:
            if key in fields:
                return fields.index(key)
        for i, field in enumerate(fields):
            s = str(field)
            if any(k in s for k in keys):
                return i
        return None

    code_i = idx(["證券代號"])
    foreign_i = idx(["外陸資買賣超股數(不含外資自營商)", "外陸資買賣超股數"])
    trust_i = idx(["投信買賣超股數"])
    dealer_i = idx(["自營商買賣超股數"])

    if code_i is None:
        return {}

    def num(v):
        try:
            s = str(v).replace(",", "").strip()
            if s in ("", "-", "--", "—", "nan", "None"):
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    out = {}
    for row in rows:
        if len(row) <= code_i:
            continue
        code = str(row[code_i]).strip()
        if not code:
            continue
        foreign = num(row[foreign_i]) if foreign_i is not None and len(row) > foreign_i else np.nan
        trust = num(row[trust_i]) if trust_i is not None and len(row) > trust_i else np.nan
        dealer = num(row[dealer_i]) if dealer_i is not None and len(row) > dealer_i else np.nan
        vals = [foreign, trust, dealer]
        total = sum(v for v in vals if pd.notna(v)) if any(pd.notna(v) for v in vals) else np.nan
        out[code] = {
            "外資": foreign,
            "投信": trust,
            "自營商": dealer,
            "法人合計": total,
        }
    return out

def institutional_day(date_str):
    data = _get_json(
        "https://www.twse.com.tw/rwd/zh/fund/T86",
        {"date": date_str, "selectType": "ALLBUT0999", "response": "json"},
    )
    return _parse_t86(data)

def latest_institutional():
    # 逐日往回找最近可用交易日
    for i in range(12):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = institutional_day(d)
        if data:
            return data, d
    return {}, None

def institutional_history(code, days=20):
    rows = []
    found = 0
    for i in range(45):
        d = datetime.now() - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        ds = d.strftime("%Y%m%d")
        mp = institutional_day(ds)
        if code in mp:
            row = dict(mp[code])
            row["日期"] = ds
            rows.append(row)
            found += 1
            if found >= days:
                break
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)

def institutional_summary(code):
    h = institutional_history(code, 20)
    if h.empty:
        return {
            "history": h,
            "連買": 0, "連賣": 0,
            "5日": np.nan, "20日": np.nan
        }

    s = pd.to_numeric(h["法人合計"], errors="coerce")
    consecutive_buy = 0
    consecutive_sell = 0

    for v in reversed(s.tolist()):
        if pd.isna(v) or v == 0:
            break
        if v > 0 and consecutive_sell == 0:
            consecutive_buy += 1
        else:
            break

    for v in reversed(s.tolist()):
        if pd.isna(v) or v == 0:
            break
        if v < 0 and consecutive_buy == 0:
            consecutive_sell += 1
        else:
            break

    return {
        "history": h,
        "連買": consecutive_buy,
        "連賣": consecutive_sell,
        "5日": s.tail(5).sum(min_count=1),
        "20日": s.tail(20).sum(min_count=1),
    }

def revenue_table():
    data = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap05_L")
    return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

def revenue_for(code, df):
    if df.empty:
        return {"營收": np.nan, "YoY": np.nan, "MoM": np.nan, "累計": np.nan}

    code_cols = [c for c in df.columns if "公司代號" in str(c) or "證券代號" in str(c)]
    if not code_cols:
        return {"營收": np.nan, "YoY": np.nan, "MoM": np.nan, "累計": np.nan}

    rows = df[df[code_cols[0]].astype(str).str.strip() == str(code)]
    if rows.empty:
        return {"營收": np.nan, "YoY": np.nan, "MoM": np.nan, "累計": np.nan}

    row = rows.iloc[-1]

    def pick(keys):
        for col in df.columns:
            if any(k in str(col) for k in keys):
                return row[col]
        return np.nan

    return {
        "營收": pick(["當月營收", "本月營收"]),
        "YoY": pick(["前期比較增減(%)", "年增率", "年增"]),
        "MoM": pick(["上月比較增減(%)", "月增率", "月增"]),
        "累計": pick(["累計營收"]),
    }
