import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from kline_engine import prepare, detect_patterns, bearish_patterns, score
from data_sources import (
    twse_stock_list, latest_institutional, institutional_summary,
    revenue_table, revenue_for
)

st.set_page_config(page_title="台股雷達 PRO V1", page_icon="📈", layout="wide")

st.title("📈 台股雷達 PRO V1")
st.caption("從零建立｜K線自動辨識 × 法人 × 營收 × 技術雷達")

@st.cache_data(ttl=1800)
def price_history(code):
    for suffix in [".TW", ".TWO"]:
        try:
            t = yf.Ticker(f"{code}{suffix}")
            d = t.history(period="1y", interval="1d", auto_adjust=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                return d.dropna()
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def stock_list_cached():
    return twse_stock_list()

@st.cache_data(ttl=1800)
def institutional_cached():
    return latest_institutional()

@st.cache_data(ttl=3600)
def revenue_cached():
    return revenue_table()

def parse_tokens(text):
    return [x.strip() for x in re.split(r"[\s,，、;；]+", text or "") if x.strip()]

def num(v):
    try:
        s = str(v).replace(",", "").replace("%", "").strip()
        if s in ("", "-", "--", "nan", "None"):
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def analyze(code, name, inst_map, rev_df):
    raw = price_history(code)
    if raw.empty or len(raw) < 60:
        return None

    d = prepare(raw)
    bullish = detect_patterns(d)
    bearish = bearish_patterns(d)

    inst = inst_map.get(code, {})
    inst_stats = institutional_summary(code)
    rev = revenue_for(code, rev_df)

    revenue_yoy = num(rev["YoY"])
    result = score(
        d, bullish, bearish,
        institution_total=inst.get("法人合計", np.nan),
        institution_5d=inst_stats.get("5日", np.nan),
        revenue_yoy=revenue_yoy,
    )

    last = d.iloc[-1]
    return {
        "code": code,
        "name": name,
        "data": d,
        "close": float(last.Close),
        "change_pct": float((last.Close / d.Close.iloc[-2] - 1) * 100),
        "volume": float(last.Volume),
        "rsi": float(last.RSI) if pd.notna(last.RSI) else np.nan,
        "macd": float(last.MACD) if pd.notna(last.MACD) else np.nan,
        "bullish": bullish,
        "bearish": bearish,
        "institution": inst,
        "inst_stats": inst_stats,
        "revenue": rev,
        "score": result,
    }

def candle_chart(result):
    d = result["data"].tail(120).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
        name="K線"
    ))
    for n in [5, 20, 60]:
        if f"MA{n}" in d:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[f"MA{n}"], mode="lines", name=f"MA{n}"
            ))

    # 在最後一天顯示型態標籤
    labels = result["bullish"][:4] + result["bearish"][:2]
    if labels:
        y = float(d.High.iloc[-1]) * 1.03
        fig.add_annotation(
            x=d.index[-1], y=y, text="、".join(labels),
            showarrow=True, arrowhead=2
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ---------- 輸入 ----------
st.subheader("① 輸入股票")
query = st.text_area(
    "股票代號 / 名稱",
    placeholder="例如：2330\n2317\n2454\n或 2330,2317,2454",
    height=100
)

c1, c2 = st.columns([1, 1])
with c1:
    run = st.button("🔍 開始分析", type="primary", use_container_width=True)
with c2:
    clear = st.button("🧹 清除快取", use_container_width=True)

if clear:
    st.cache_data.clear()
    st.rerun()

if not run:
    st.info("""
### 使用方式
1. 輸入股票代號，例如 `2330`
2. 按「開始分析」
3. 系統自動抓 K 線、法人、營收
4. 自動辨識 K 線型態
5. 給出「偏多研究 / 再等等 / 偏弱研究」
6. 點個股查看完整原因

**這個版本是全新專案，不需要昨天任何程式。**
""")
    st.stop()

# ---------- 資料 ----------
with st.spinner("載入股票清單與資料…"):
    stocks = stock_list_cached()
    inst_map, inst_date = institutional_cached()
    rev_df = revenue_cached()

code_to_name = dict(zip(stocks["代號"], stocks["名稱"])) if not stocks.empty else {}
name_to_code = {v: k for k, v in code_to_name.items()}

tokens = parse_tokens(query)
codes = []
unresolved = []

for token in tokens:
    if token in code_to_name:
        codes.append(token)
    elif token in name_to_code:
        codes.append(name_to_code[token])
    else:
        matches = [c for c, n in code_to_name.items() if token in n]
        if matches:
            codes.append(matches[0])
        elif token.isdigit():
            codes.append(token)
        else:
            unresolved.append(token)

codes = list(dict.fromkeys(codes))

if unresolved:
    st.warning("找不到：" + "、".join(unresolved))
if not codes:
    st.error("請輸入有效股票代號或名稱。")
    st.stop()

# ---------- 分析 ----------
results = []
progress = st.progress(0)

for i, code in enumerate(codes):
    result = analyze(code, code_to_name.get(code, ""), inst_map, rev_df)
    if result:
        results.append(result)
    progress.progress((i + 1) / len(codes))

progress.empty()

if not results:
    st.error("目前沒有取得足夠的歷史資料。請確認股票代號或稍後再試。")
    st.stop()

# ---------- 摘要 ----------
rows = []
for r in results:
    s = r["score"]
    inst = r["institution"]
    rows.append({
        "代號": r["code"],
        "名稱": r["name"],
        "收盤": round(r["close"], 2),
        "漲跌%": round(r["change_pct"], 2),
        "法人合計": inst.get("法人合計", np.nan),
        "法人5日": r["inst_stats"].get("5日", np.nan),
        "RSI": round(r["rsi"], 1) if pd.notna(r["rsi"]) else np.nan,
        "分數": s["分數"],
        "訊號": s["訊號"],
        "主要K線": "、".join(r["bullish"][:4]) if r["bullish"] else "—",
    })

summary = pd.DataFrame(rows).sort_values("分數", ascending=False)

st.subheader("② 自動選股結果")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("③ 個股詳細分析")
selected = st.selectbox(
    "選擇股票",
    [r["code"] for r in results],
    format_func=lambda x: f"{x}｜{code_to_name.get(x, '')}"
)
r = next(x for x in results if x["code"] == selected)
s = r["score"]

if s["訊號"] == "偏多研究":
    st.success(f"🟢 {s['訊號']}｜{s['動作']}")
elif s["訊號"] == "再等等":
    st.warning(f"🟡 {s['訊號']}｜{s['動作']}")
else:
    st.error(f"🔴 {s['訊號']}｜{s['動作']}")

a,b,c,d = st.columns(4)
a.metric("雷達分數", s["分數"])
b.metric("收盤", f"{r['close']:.2f}")
c.metric("今日漲跌", f"{r['change_pct']:.2f}%")
d.metric("RSI", f"{r['rsi']:.1f}" if pd.notna(r["rsi"]) else "—")

st.plotly_chart(candle_chart(r), use_container_width=True)

left, right = st.columns(2)

with left:
    st.markdown("### 🕯️ K線辨識")
    st.write("**偏多/反轉型態**")
    st.write("、".join(r["bullish"]) if r["bullish"] else "目前沒有符合的型態")
    st.write("**偏空型態**")
    st.write("、".join(r["bearish"]) if r["bearish"] else "目前沒有明顯偏空型態")

with right:
    st.markdown("### 🧠 系統判斷")
    st.write(f"**技術分：** {s['技術分']}")
    st.write(f"**籌碼分：** {s['籌碼分']}")
    st.write(f"**基本面分：** {s['基本面分']}")
    st.write("**主要原因：** " + ("、".join(s["理由"]) if s["理由"] else "條件不足"))
    st.write("**風險：** " + ("、".join(s["風險"]) if s["風險"] else "目前未偵測到主要風險訊號"))

st.markdown("---")
st.subheader("④ 法人")

inst = r["institution"]
x,y,z,w = st.columns(4)
x.metric("外資", f"{inst.get('外資', np.nan):,.0f}" if pd.notna(inst.get("外資", np.nan)) else "—")
y.metric("投信", f"{inst.get('投信', np.nan):,.0f}" if pd.notna(inst.get("投信", np.nan)) else "—")
z.metric("自營商", f"{inst.get('自營商', np.nan):,.0f}" if pd.notna(inst.get("自營商", np.nan)) else "—")
w.metric("三大法人", f"{inst.get('法人合計', np.nan):,.0f}" if pd.notna(inst.get("法人合計", np.nan)) else "—")

x,y,z = st.columns(3)
x.metric("法人連買", f"{r['inst_stats']['連買']} 天")
y.metric("法人連賣", f"{r['inst_stats']['連賣']} 天")
z.metric("法人20日", f"{r['inst_stats']['20日']:,.0f}" if pd.notna(r["inst_stats"]["20日"]) else "—")

hist = r["inst_stats"]["history"]
if not hist.empty:
    st.dataframe(hist.sort_values("日期", ascending=False), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("⑤ 營收")
rev = r["revenue"]
st.write(f"當月營收：{rev['營收']}")
st.write(f"YoY：{rev['YoY']}")
st.write(f"MoM：{rev['MoM']}")
st.write(f"累計營收：{rev['累計']}")

st.caption(
    f"法人資料最近可用日期：{inst_date or '未取得'}。"
    "訊號是規則化研究結果，不代表保證上漲，也不應單獨作為交易決策。"
)
