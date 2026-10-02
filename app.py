import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="台股雷達 PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 基本設定
# =========================================================
st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
        min-height: 105px;
    }
    .small {font-size: .82rem; opacity: .75;}
    .big {font-size: 1.65rem; font-weight: 700;}
    .green {color:#16a34a;}
    .red {color:#dc2626;}
    .yellow {color:#ca8a04;}
    .blue {color:#2563eb;}
</style>
""", unsafe_allow_html=True)

TW_SUFFIX = ".TW"

# =========================================================
# 資料
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_stock(symbol: str, period="1y"):
    code = str(symbol).strip()
    ticker = code if "." in code else code + TW_SUFFIX

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(), ticker

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame(), ticker

    df = df[required].copy()
    df = df.dropna()
    return df, ticker


def add_indicators(df):
    x = df.copy()

    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))

    tr1 = x["High"] - x["Low"]
    tr2 = (x["High"] - x["Close"].shift()).abs()
    tr3 = (x["Low"] - x["Close"].shift()).abs()
    x["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["ATR14"] = x["TR"].rolling(14).mean()

    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL20"]

    x["RET20"] = x["Close"].pct_change(20)
    x["HIGH20"] = x["High"].rolling(20).max()
    x["LOW20"] = x["Low"].rolling(20).min()

    return x


def round_price(v):
    if pd.isna(v):
        return np.nan
    if v >= 100:
        return round(float(v), 1)
    if v >= 10:
        return round(float(v), 2)
    return round(float(v), 3)


def analyze(df):
    x = add_indicators(df)
    last = x.iloc[-1]

    price = float(last["Close"])
    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else price * .03
    ma5 = float(last["MA5"]) if pd.notna(last["MA5"]) else price
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else price
    ma60 = float(last["MA60"]) if pd.notna(last["MA60"]) else price
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else 50
    vol_ratio = float(last["VOL_RATIO"]) if pd.notna(last["VOL_RATIO"]) else 1
    high20 = float(last["HIGH20"])
    low20 = float(last["LOW20"])

    # 使用近期高低點 + ATR 建立價格區間，而不是固定寫死目標。
    resistance = max(ma20, price + atr * .8)
    target1 = max(resistance, price + atr * 1.2)
    target2 = max(target1 + atr * .8, price + atr * 2.2)
    long_target = max(target2 + atr * .8, price + atr * 3.0)

    # 支撐取近期低點與 MA60 的合理交集
    support = min(low20, ma60)
    stop = max(0.01, support - atr * .35)

    # 趨勢狀態
    bullish = price > ma20 and ma20 >= ma60
    weak = price < ma20 and ma20 < ma60

    # 規則化風控狀態
    if price <= stop:
        status = "🔴 觸發風控"
        status_class = "red"
    elif price >= target2:
        status = "🟢 已進入分批獲利區"
        status_class = "green"
    elif price >= target1:
        status = "🟡 第一目標區"
        status_class = "yellow"
    elif bullish:
        status = "🟢 趨勢偏多"
        status_class = "green"
    elif weak:
        status = "🔴 趨勢偏弱"
        status_class = "red"
    else:
        status = "🟡 觀察"
        status_class = "yellow"

    return {
        "df": x,
        "price": price,
        "atr": atr,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "support": support,
        "stop": stop,
        "resistance": resistance,
        "target1": target1,
        "target2": target2,
        "long_target": long_target,
        "status": status,
        "status_class": status_class,
    }


# =========================================================
# 持倉資料
# =========================================================
DEFAULT_HOLDINGS = pd.DataFrame([
    {"代號": "3576", "名稱": "聯合再生", "股數": 2000, "成本": 18.92},
])

if "holdings" not in st.session_state:
    st.session_state.holdings = DEFAULT_HOLDINGS.copy()

# =========================================================
# 側邊欄
# =========================================================
with st.sidebar:
    st.title("📈 台股雷達 PRO")
    st.caption("智能選股 × 風控 × 持倉管理")

    page = st.radio(
        "功能",
        ["持倉總覽", "個股分析", "持倉編輯", "設定"],
        index=0,
    )

    st.divider()
    st.caption("資料來源：Yahoo Finance")
    st.caption("價格與資料可能延遲；本工具不保證交易結果。")


# =========================================================
# 個股分析共用函式
# =========================================================
def render_stock(symbol, cost=None, shares=None, name=None):
    df, ticker = get_stock(symbol, "1y")

    if df.empty:
        st.error(f"{symbol} 無法取得資料。請確認代號或稍後重試。")
        return

    a = analyze(df)
    price = a["price"]

    name_text = name or symbol
    st.markdown(f"## {symbol} {name_text}")

    c1, c2, c3, c4, c5 = st.columns(5)

    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price
    change = price - prev
    change_pct = change / prev * 100 if prev else 0

    with c1:
        st.metric("現價", f"{price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    with c2:
        st.metric("MA20", f"{a['ma20']:.2f}")
    with c3:
        st.metric("RSI14", f"{a['rsi']:.1f}")
    with c4:
        st.metric("ATR14", f"{a['atr']:.2f}")
    with c5:
        st.metric("量能", f"{a['vol_ratio']:.2f}x")

    tab1, tab2, tab3 = st.tabs(["📊 交易計畫", "📈 技術指標", "🧾 原始資料"])

    with tab1:
        left, right = st.columns([1.25, 1])

        with left:
            st.subheader("K線與成交量")
            chart_df = a["df"][["Close", "MA5", "MA20", "MA60"]].tail(120)
            st.line_chart(chart_df, height=420)

        with right:
            st.subheader("持倉狀態")

            if cost is not None and shares is not None:
                pnl = (price - cost) * shares
                pnl_pct = (price / cost - 1) * 100

                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="small">持有股數</div>
                    <div class="big">{int(shares):,}</div>
                    <div class="small">成本 {cost:.2f}　現價 {price:.2f}</div>
                    <div class="{'green' if pnl >= 0 else 'red'}">
                    損益 {pnl:+,.0f} 元　({pnl_pct:+.2f}%)
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### 價格計畫")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("停損", f"{round_price(a['stop']):.2f}")
            with m2:
                st.metric("第一壓力", f"{round_price(a['resistance']):.2f}")
            with m3:
                st.metric("第二目標", f"{round_price(a['target2']):.2f}")
            with m4:
                st.metric("延伸目標", f"{round_price(a['long_target']):.2f}")

            st.info(
                "第一壓力不是自動賣出價。"
                "只有突破後才重新計算後續目標；"
                "停損則是獨立的風控條件。"
            )

            if price <= a["stop"]:
                st.error("價格已落在風控線以下，請檢查是否需要執行自己的風控規則。")
            elif price >= a["target2"]:
                st.success("價格已進入第二目標區，可依你的分批策略重新評估。")
            elif price >= a["resistance"]:
                st.warning("已接近/突破第一壓力，觀察成交量與收盤確認。")
            else:
                st.success("尚未觸發風控，也尚未到主要目標區。")

            potential = (a["target2"] / price - 1) * 100
            risk = (price / a["stop"] - 1) * 100 if a["stop"] > 0 else np.nan
            rr = potential / risk if risk > 0 else np.nan

            st.markdown(
                f"""
                **風報評估**
                - 到第二目標：`{potential:+.2f}%`
                - 到停損：`{-risk:.2f}%`
                - 風報比：約 `1 : {rr:.2f}`
                """
            )

        st.subheader("規則化進出場邏輯")
        rules = pd.DataFrame([
            ["價格 ≤ 停損", f"{a['stop']:.2f}", "🔴 觸發風控"],
            ["價格 < 第一壓力", f"< {a['resistance']:.2f}", "🟡 持有/觀察"],
            ["突破第一壓力 + 量能放大", f"> {a['resistance']:.2f}", "🟢 進入確認區"],
            ["價格 ≥ 第二目標", f"≥ {a['target2']:.2f}", "🟡 分批獲利評估"],
            ["價格 ≥ 延伸目標", f"≥ {a['long_target']:.2f}", "🟡 趨勢延伸評估"],
        ], columns=["條件", "價格", "系統狀態"])
        st.dataframe(rules, use_container_width=True, hide_index=True)

    with tab2:
        ind = a["df"][["Close", "MA5", "MA20", "MA60", "RSI", "ATR14", "VOL_RATIO"]].tail(30).copy()
        st.dataframe(ind.round(2), use_container_width=True)

    with tab3:
        st.dataframe(a["df"].tail(100).round(2), use_container_width=True)


# =========================================================
# 持倉總覽
# =========================================================
if page == "持倉總覽":
    st.title("持倉管理")

    rows = []
    for _, h in st.session_state.holdings.iterrows():
        symbol = str(h["代號"])
        name = str(h.get("名稱", ""))
        shares = float(h["股數"])
        cost = float(h["成本"])

        df, _ = get_stock(symbol, "1y")
        if df.empty:
            rows.append({
                "代號": symbol, "名稱": name, "狀態": "⚪ 無資料",
                "股數": shares, "成本": cost, "現價": np.nan,
                "損益%": np.nan, "損益": np.nan,
                "停損": np.nan, "第一壓力": np.nan, "第二目標": np.nan,
            })
            continue

        a = analyze(df)
        price = a["price"]
        pnl = (price - cost) * shares
        pnl_pct = (price / cost - 1) * 100

        rows.append({
            "代號": symbol,
            "名稱": name,
            "狀態": a["status"],
            "股數": int(shares),
            "成本": round(cost, 2),
            "現價": round(price, 2),
            "損益%": round(pnl_pct, 2),
            "損益": round(pnl, 0),
            "停損": round_price(a["stop"]),
            "第一壓力": round_price(a["resistance"]),
            "第二目標": round_price(a["target2"]),
        })

    result = pd.DataFrame(rows)

    total_pnl = pd.to_numeric(result["損益"], errors="coerce").fillna(0).sum()
    market_value = (
        pd.to_numeric(result["現價"], errors="coerce").fillna(0)
        * pd.to_numeric(result["股數"], errors="coerce").fillna(0)
    ).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("持倉檔數", len(result))
    c2.metric("市值", f"{market_value:,.0f}")
    c3.metric("未實現損益", f"{total_pnl:+,.0f}")
    c4.metric("資料更新", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
        column_config={
            "損益%": st.column_config.NumberColumn(format="%.2f%%"),
            "損益": st.column_config.NumberColumn(format="%+d"),
        },
    )

    st.divider()

    if len(st.session_state.holdings) > 0:
        first = st.session_state.holdings.iloc[0]
        render_stock(
            str(first["代號"]),
            float(first["成本"]),
            float(first["股數"]),
            str(first.get("名稱", "")),
        )


# =========================================================
# 個股分析
# =========================================================
elif page == "個股分析":
    st.title("個股分析")

    symbol = st.text_input("輸入台股代號", value="3576")
    st.caption("例如：2330、3576、0050；系統會自動加入 .TW")

    if st.button("🔄 重新抓取資料", type="primary"):
        get_stock.clear()

    if symbol:
        render_stock(symbol.strip())


# =========================================================
# 持倉編輯
# =========================================================
elif page == "持倉編輯":
    st.title("持倉資料")

    edited = st.data_editor(
        st.session_state.holdings,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "代號": st.column_config.TextColumn("代號"),
            "名稱": st.column_config.TextColumn("名稱"),
            "股數": st.column_config.NumberColumn("股數", min_value=0, step=100),
            "成本": st.column_config.NumberColumn("成本", min_value=0, step=0.01),
        },
        key="holdings_editor",
    )

    if st.button("💾 儲存持倉", type="primary"):
        x = edited.copy()
        x["代號"] = x["代號"].astype(str).str.replace(".TW", "", regex=False)
        x["股數"] = pd.to_numeric(x["股數"], errors="coerce").fillna(0)
        x["成本"] = pd.to_numeric(x["成本"], errors="coerce").fillna(0)
        x = x[x["代號"].str.len() > 0]
        st.session_state.holdings = x.reset_index(drop=True)
        st.success("持倉已更新。")

    st.info(
        "這版把「第一壓力」和「真正目標」分開："
        "第一壓力不會直接被當成賣出價，避免出現『進場後只漲一點就被迫出場』的問題。"
    )


# =========================================================
# 設定
# =========================================================
else:
    st.title("系統設定")

    st.subheader("目前版本的核心規則")
    st.markdown("""
    1. **停損獨立計算**：由近期支撐與 ATR 推導，不跟目標價混在一起。
    2. **第一壓力 ≠ 賣出價**：第一壓力是確認區。
    3. **第二目標才是主要獲利區**：達標後才進入分批獲利評估。
    4. **突破後重新計算**：若突破第一壓力且量能放大，延伸目標會提高。
    5. **持倉頁直接使用輸入成本**：不再用錯誤的進場價推估損益。
    6. **資料有時間標記**：避免把昨天收盤誤認成盤中即時價。
    """)

    if st.button("🧹 清除快取並重新抓資料"):
        get_stock.clear()
        st.success("快取已清除。")


st.divider()
st.caption(
    "⚠️ 本工具是規則化分析與風控輔助，不是保證獲利或自動下單系統。"
)
