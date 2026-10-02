import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from kline_engine import prepare, detect_patterns, bearish_patterns, score, PATTERN_DESCRIPTIONS
from data_sources import stock_list_all, latest_institutional, institutional_summary, institutional_bulk_summary, revenue_table, revenue_for, financials_yfinance

st.set_page_config(page_title="TW STOCK RADAR PRO", page_icon="📈", layout="wide")

st.title("📈 TW STOCK RADAR PRO")
st.caption("Institutional Flow × Revenue × Technical Analysis × Candlestick Patterns｜Smart Stock Screening & Market Intelligence")

# ---------- Data ----------
@st.cache_data(ttl=1800, show_spinner=False)
def price_history(code, period="1y"):
    for suffix in [".TW", ".TWO"]:
        try:
            d=yf.Ticker(f"{code}{suffix}").history(period=period,interval="1d",auto_adjust=False,timeout=10)
            if not d.empty:
                if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
                return d.dropna()
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def stocks_cached(): return stock_list_all()
@st.cache_data(ttl=1800, show_spinner=False)
def inst_cached(): return latest_institutional()
@st.cache_data(ttl=3600, show_spinner=False)
def revenue_cached(): return revenue_table()


def num(v):
    try:
        s=str(v).replace(",","").replace("%","").strip()
        if s in ("","-","--","nan","None"): return np.nan
        return float(s)
    except Exception: return np.nan

def analyze(code,name,inst_map,rev_df,inst_stats_override=None):
    raw=price_history(code)
    if raw.empty or len(raw)<60: return None
    d=prepare(raw); bull=detect_patterns(d); bear=bearish_patterns(d)
    inst=inst_map.get(code,{})
    insts=inst_stats_override if inst_stats_override is not None else institutional_summary(code)
    rev=revenue_for(code,rev_df)
    s=score(d,bull,bear,inst.get("法人合計",np.nan),insts.get("5日",np.nan),insts.get("20日",np.nan),num(rev.get("YoY")))
    last=d.iloc[-1]
    return {"code":code,"name":name,"data":d,"close":float(last.Close),"change_pct":float((last.Close/d.Close.iloc[-2]-1)*100),"volume":float(last.Volume),"rsi":float(last.RSI) if pd.notna(last.RSI) else np.nan,"macd":float(last.MACD) if pd.notna(last.MACD) else np.nan,"bullish":bull,"bearish":bear,"institution":inst,"inst_stats":insts,"revenue":rev,"score":s}

def candle_chart(r):
    d=r["data"].tail(160)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="K線"))
    for n in [5,20,60]: fig.add_trace(go.Scatter(x=d.index,y=d[f"MA{n}"],mode="lines",name=f"MA{n}"))
    fig.update_layout(height=620,xaxis_rangeslider_visible=False,margin=dict(l=20,r=20,t=40,b=20))
    return fig

def parse_tokens(text): return [x.strip() for x in re.split(r"[\s,，、;；]+",text or "") if x.strip()]

def run_manual(codes,stocks,inst_map,rev_df):
    names=dict(zip(stocks["代號"],stocks["名稱"])) if not stocks.empty else {}
    results=[]; bar=st.progress(0)
    for i,code in enumerate(codes):
        r=analyze(code,names.get(code,""),inst_map,rev_df)
        if r: results.append(r)
        bar.progress((i+1)/len(codes))
    bar.empty(); return results

def _lots(v):
    """TWSE 法人原始資料為股；畫面統一顯示為張。"""
    return v / 1000 if pd.notna(v) else np.nan

def result_row(r):
    """把分析結果安全地轉成表格列；遇到單檔資料缺欄位也不讓整個 Streamlit 頁面崩潰。"""
    if not isinstance(r, dict):
        return {}
    s=r.get("score") or {}
    inst=r.get("institution") or {}
    insts=r.get("inst_stats") or {}
    return {
        "代號":r.get("code","—"),
        "名稱":r.get("name","—"),
        "收盤":round(num(r.get("close")),2) if pd.notna(num(r.get("close"))) else np.nan,
        "漲跌%":round(num(r.get("change_pct")),2) if pd.notna(num(r.get("change_pct"))) else np.nan,
        "法人買賣超(張)":_lots(num(inst.get("法人合計"))),
        "法人5日(張)":_lots(num(insts.get("5日"))),
        "法人20日(張)":_lots(num(insts.get("20日"))),
        "RSI":round(num(r.get("rsi")),1) if pd.notna(num(r.get("rsi"))) else np.nan,
        "雷達分數":s.get("分數",np.nan),
        "早期趨勢分":s.get("早期趨勢分",np.nan),
        "早期條件數":s.get("早期條件數",np.nan),
        "60日位階%":round(num(s.get("60日位階%")),1) if pd.notna(num(s.get("60日位階%"))) else np.nan,
        "位置距MA20%":round(num(s.get("位置距MA20%")),2) if pd.notna(num(s.get("位置距MA20%"))) else np.nan,
        "20日漲幅%":round(num(s.get("20日漲幅%")),2) if pd.notna(num(s.get("20日漲幅%"))) else np.nan,
        "進場區":(f"{num(s.get("進場區下緣")):.2f}～{num(s.get("進場區上緣")):.2f}" if pd.notna(num(s.get("進場區下緣"))) and pd.notna(num(s.get("進場區上緣"))) else "—"),
        "判斷":s.get("訊號","⚪ 資料不足"),
        "動作":s.get("動作","資料不足"),
        "進場參考":round(num(s.get("進場參考")),2) if pd.notna(num(s.get("進場參考"))) else np.nan,
        "停損":round(num(s.get("停損參考")),2) if pd.notna(num(s.get("停損參考"))) else np.nan,
        "目標1":round(num(s.get("目標1")),2) if pd.notna(num(s.get("目標1"))) else np.nan,
        "目標2":round(num(s.get("目標2")),2) if pd.notna(num(s.get("目標2"))) else np.nan,
        "目標1依據":s.get("目標1依據", ""),
        "目標2依據":s.get("目標2依據", ""),
        "風險報酬":round(num(s.get("風險報酬")),2) if pd.notna(num(s.get("風險報酬"))) else np.nan,
        "第一目標空間%":round(num(s.get("第一目標空間%")),2) if pd.notna(num(s.get("第一目標空間%"))) else np.nan,
        "風險報酬達標":"是" if s.get("風險報酬達標") is True else "否" if s.get("風險報酬達標") is False else "—",
        "K線訊號":"、".join((r.get("bullish",[])[:3]+r.get("bearish",[])[:2])),
        "K線出場警戒":"、".join(r.get("bearish",[])[:3]) if r.get("bearish") else "—"
    }

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ 選股設定")
    st.write("系統會先縮小候選池，再逐檔分析K線、量能、法人與營收，避免一次查詢全市場造成逾時。")
    universe=st.selectbox("候選股範圍",["成交金額前 50","成交金額前 100","成交金額前 200","全部市場（較慢）"])
    min_score=st.slider("最低雷達分數",40,90,65)
    only_bull=st.checkbox("只顯示偏多訊號",False)
    max_scan=st.slider("最多實際分析檔數",20,200,200,step=10)
    st.divider()
    st.caption("⚠️ 分數是規則化研究工具，不是獲利保證，也不代表個人化投資建議。")

# 重要：首頁啟動時不抓股票清單、法人、營收。
# 這些資料只在使用者真正按下「自動選股／個股查詢」後才載入，避免 Streamlit 啟動卡住。
stocks=None
inst_map={}
inst_date=None
rev_df=pd.DataFrame()

def load_core_data():
    stocks=stocks_cached()
    if stocks.empty:
        return stocks, {}, None, pd.DataFrame()
    inst_map,inst_date=inst_cached()
    rev_df=revenue_cached()
    st.session_state["stocks"] = stocks
    st.session_state["inst_map"] = inst_map
    st.session_state["inst_date"] = inst_date
    st.session_state["rev_df"] = rev_df
    return stocks,inst_map,inst_date,rev_df

def get_core_data():
    return (st.session_state.get("stocks"),
            st.session_state.get("inst_map",{}),
            st.session_state.get("inst_date"),
            st.session_state.get("rev_df",pd.DataFrame()))

def load_performance_summary():
    path = Path(__file__).resolve().parent / "performance_summary.json"
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def load_signal_history():
    path = Path(__file__).resolve().parent / "signal_history.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def load_daily_radar_cache():
    """讀取 GitHub Actions 每日產生的雷達快取；失敗時完全不影響原本手動選股。"""
    path = Path(__file__).resolve().parent / "radar_cache.json"
    try:
        if not path.exists():
            return pd.DataFrame(), ""
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame(), ""
        return pd.DataFrame(rows), str(payload.get("generated_at", ""))
    except Exception:
        return pd.DataFrame(), ""

# ---------- Tabs ----------
tab_auto,tab_search,tab_portfolio,tab_detail,tab_fin,tab_patterns=st.tabs(["🚀 自動選股","🔎 個股查詢","🛡️ 持倉/出場","📊 詳細分析","📑 財報/法人","🕯️ K線型態庫"])

with tab_auto:
    st.subheader("🚀 全自動選股")
    st.markdown("**目標：不用先告訴系統股票代號，由系統自己從市場候選池找出符合條件的股票。**")
    nmap={"成交金額前 50":50,"成交金額前 100":100,"成交金額前 200":200,"全部市場（較慢）":999999}
    limit=max_scan
    pool_label="全市場" if universe=="全部市場（較慢）" else f"{nmap[universe]} 檔"
    st.info(f"目前候選池：{pool_label}；本次最多深度分析 {limit} 檔。篩選會優先處理成交金額較大的股票。")

    # 每日自動更新：只讀 GitHub Actions 產生的快取，不改原本版面或手動分析流程。
    perf = load_performance_summary()
    st.markdown("### 📊 訊號實戰績效")
    p1, p2, p3, p4 = st.columns(4)
    d1, d5, d20 = perf.get("1日報酬%", {}), perf.get("5日報酬%", {}), perf.get("20日報酬%", {})
    p1.metric("累計訊號", perf.get("total_signals", 0))
    p2.metric("隔日正報酬", f"{d1.get('positive_rate'):.1f}%" if d1.get('positive_rate') is not None else "資料累積中")
    p3.metric("5日正報酬", f"{d5.get('positive_rate'):.1f}%" if d5.get('positive_rate') is not None else "資料累積中")
    p4.metric("5日平均報酬", f"{d5.get('avg_return'):+.2f}%" if d5.get('avg_return') is not None else "資料累積中")
    sig_stats = perf.get("by_signal", {})
    if sig_stats:
        # 這裡是「歷史績效累積」，不是今天的選股數量；避免和本次掃描結果混在一起。
        rows=[]
        for sig, v in sig_stats.items():
            rows.append({"歷史訊號":sig,"累積筆數":v.get("count",0),"已完成5日":v.get("5d_count",0),"5日正報酬率":v.get("5d_positive_rate"),"5日平均報酬":v.get("5d_avg_return")})
        st.markdown("#### 📚 歷史訊號績效（累積）")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("上表只用來驗證策略長期表現，不代表今天有多少檔；今天的分類數量會在下方「本次結果分布」單獨統計。1/5/20 日績效需等實際交易日經過後才會補齊。")

    if "auto_table" not in st.session_state:
        daily_df, daily_time = load_daily_radar_cache()
        if not daily_df.empty:
            show_df = daily_df.copy()
            if only_bull and "判斷" in show_df.columns:
                show_df = show_df[show_df["判斷"].isin(["🟢 早期佈局", "🟢 買進條件成立", "🔵 突破確認", "🟡 等待"])]
            # 早期佈局是獨立的提前模型：即使傳統雷達分數未達門檻，也必須保留。
            if "雷達分數" in show_df.columns:
                total_ok = pd.to_numeric(show_df["雷達分數"], errors="coerce") >= min_score
                early_ok = show_df.get("判斷", pd.Series(index=show_df.index, dtype=str)).eq("🟢 早期佈局")
                show_df = show_df[total_ok | early_ok]
            signal_order = {
                "🟢 早期佈局": 0,
                "🟢 買進條件成立": 1,
                "🔵 突破確認": 2,
                "🟡 等待": 3,
                "🟠 不追高": 4,
                "🔴 不買": 5,
            }
            if "判斷" in show_df.columns:
                show_df["_signal_order"] = show_df["判斷"].map(signal_order).fillna(99)
                show_df["_score_num"] = pd.to_numeric(show_df["雷達分數"], errors="coerce").fillna(-999) if "雷達分數" in show_df.columns else -999
                show_df = show_df.sort_values(["_signal_order", "_score_num"], ascending=[True, False], kind="stable")
                show_df = show_df.drop(columns=["_signal_order", "_score_num"], errors="ignore")
            st.session_state["auto_table"] = show_df.reset_index(drop=True)
            st.caption(f"📅 每日自動更新：{daily_time or '最近一次成功更新'}｜資料由 GitHub Actions 產生")

    if st.button("🚀 開始全市場自動選股",type="primary",width="stretch"):
        with st.spinner("正在載入台股清單、法人與營收資料…"):
            stocks,inst_map,inst_date,rev_df=load_core_data()
        if stocks.empty:
            st.error("目前抓不到台股清單，請稍後再試。")
            st.stop()
        pool=stocks.copy()
        if "成交金額" in pool.columns: pool=pool.sort_values("成交金額",ascending=False,na_position="last")
        pool=pool.head(limit)
        codes=pool["代號"].astype(str).tolist()
        with st.spinner("正在掃描市場：K線、均線、量能、法人、營收…"):
            # 法人先一次抓最近幾個交易日，避免每檔股票重複打API。
            bulk_inst=institutional_bulk_summary(codes, days=5)
            names=dict(zip(stocks["代號"],stocks["名稱"]))
            results=[]
            bar=st.progress(0)
            for i,code in enumerate(codes):
                r=analyze(code,names.get(code,""),inst_map,rev_df,bulk_inst.get(code))
                if r: results.append(r)
                bar.progress((i+1)/len(codes))
            bar.empty()
        rows=[result_row(r) for r in results]
        if rows:
            df=pd.DataFrame(rows).sort_values("雷達分數",ascending=False)
            if only_bull:
                df=df[df["判斷"].isin(["🟢 早期佈局","🟢 買進條件成立","🔵 突破確認","🟡 等待"])]
            # 早期佈局不受傳統最低雷達分數限制；它本身就是獨立的提前模型。
            total_ok = pd.to_numeric(df["雷達分數"], errors="coerce") >= min_score
            early_ok = df["判斷"].eq("🟢 早期佈局")
            df=df[total_ok | early_ok].copy()
            st.session_state["auto_results"]=results
            st.session_state["auto_table"]=df
            st.success(f"掃描完成：成功分析 {len(results)} 檔，符合目前分數條件 {len(df)} 檔。")
        else: st.error("沒有取得足夠資料，請稍後再試。")
    if "auto_table" in st.session_state:
        df=st.session_state["auto_table"].copy()
        # 固定訊號優先順序：先找早期轉強，再看確認型訊號，最後才是不追高/不買。
        # 同一訊號內優先早期趨勢分，再依雷達分數，讓手機上第一眼就看到「還沒噴」的標的。
        signal_order = {
            "🟢 早期佈局": 0,
            "🟢 買進條件成立": 1,
            "🔵 突破確認": 2,
            "🟡 等待": 3,
            "🟠 不追高": 4,
            "🔴 不買": 5,
        }
        if not df.empty and "判斷" in df.columns:
            df["_signal_order"] = df["判斷"].map(signal_order).fillna(99)
            df["_early_num"] = pd.to_numeric(df["早期趨勢分"], errors="coerce").fillna(-999) if "早期趨勢分" in df.columns else -999
            df["_score_num"] = pd.to_numeric(df["雷達分數"], errors="coerce").fillna(-999) if "雷達分數" in df.columns else -999
            df = df.sort_values(["_signal_order", "_early_num", "_score_num"], ascending=[True, False, False], kind="stable")
            df = df.drop(columns=["_signal_order", "_early_num", "_score_num"], errors="ignore").reset_index(drop=True)
            st.session_state["auto_table"] = df

        early_df = df[df.get("判斷", pd.Series(dtype=str)) == "🟢 早期佈局"].copy() if "判斷" in df.columns else pd.DataFrame()
        buy_df = df[df.get("判斷", pd.Series(dtype=str)).isin(["🟢 買進條件成立", "🔵 突破確認"])] if "判斷" in df.columns else pd.DataFrame()
        e1,e2,e3 = st.columns(3)
        e1.metric("🟢 早期佈局", len(early_df))
        e2.metric("🟢/🔵 確認型訊號", len(buy_df))
        e3.metric("📊 本次顯示", len(df))

        # 這張表只統計「目前畫面 df」；六種分類互斥，因此總數必須等於本次顯示筆數。
        signal_order = ["🟢 早期佈局","🟢 買進條件成立","🔵 突破確認","🟡 等待","🟠 不追高","🔴 不買"]
        current_counts = df["判斷"].value_counts() if "判斷" in df.columns else pd.Series(dtype=int)
        current_rows = [{"本次最終分類": sig, "本次檔數": int(current_counts.get(sig, 0))} for sig in signal_order]
        current_summary = pd.DataFrame(current_rows)
        st.markdown("#### 📌 本次選股結果分布")
        st.dataframe(current_summary, width="stretch", hide_index=True)
        if not early_df.empty:
            top = early_df.head(8)[[c for c in ["代號","名稱","收盤","早期趨勢分","早期條件數","60日位階%","位置距MA20%","20日漲幅%","進場區"] if c in early_df.columns]]
            st.markdown("### 🟢 最值得先看的早期轉強區")
            st.dataframe(top, width="stretch", hide_index=True)
        else:
            st.info("今天沒有股票同時滿足目前的早期轉強門檻；這不代表市場沒有機會，而是目前規則刻意避免把高檔或弱勢股硬列成早期佈局。")

        st.markdown("### 📋 自動選股結果")
        st.caption("排序：🟢早期佈局 → 🟢買進 → 🔵突破 → 🟡等待 → 🟠不追高 → 🔴不買；早期區內優先依早期趨勢分排序。點一下股票那一列，下方會完整顯示該檔資料。")
        event = st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="auto_result_table",
        )

        # 點選任一股票後，把原本橫向表格的完整內容改成手機容易閱讀的
        # 「股票名稱 + 欄位/數值」卡片，避免使用者滑到最右邊後不知道是哪一檔。
        selected_rows = getattr(getattr(event, "selection", None), "rows", [])
        if selected_rows:
            idx = selected_rows[0]
            if 0 <= idx < len(df):
                selected = df.iloc[idx]
                code = str(selected.get("代號", ""))
                name = str(selected.get("名稱", ""))
                signal = str(selected.get("判斷", ""))
                st.markdown(f"### 🎯 已選取：{code} {name}")
                st.success(f"{signal}｜雷達分數：{selected.get('雷達分數', '—')}")
                st.markdown("#### 🛡️ 交易風控區")
                q1,q2,q3,q4=st.columns(4)
                q1.metric("進場參考", f"{num(selected.get('進場參考')):.2f}" if pd.notna(num(selected.get('進場參考'))) else "—")
                q2.metric("防守／停損", f"{num(selected.get('停損')):.2f}" if pd.notna(num(selected.get('停損'))) else "—")
                q3.metric("目標1", f"{num(selected.get('目標1')):.2f}" if pd.notna(num(selected.get('目標1'))) else "—")
                q4.metric("目標2", f"{num(selected.get('目標2')):.2f}" if pd.notna(num(selected.get('目標2'))) else "—")
                st.caption("風控規則：第一目標必須有至少 2R 的報酬空間才列為可進場；若上方壓力太近，系統會改列「🟡 等待」，避免為了小利承擔較大風險。已持有則到目標1可考慮分批落袋，跌破停損優先處理風險。")
                detail_items = []
                for col, val in selected.items():
                    if col.startswith("_"):
                        continue
                    if pd.isna(val):
                        val = "—"
                    detail_items.append({"項目": col, "內容": str(val)})
                st.dataframe(pd.DataFrame(detail_items), width="stretch", hide_index=True)

        if not df.empty:
            st.markdown("### 自動選股解讀")
            st.write("**🟢 早期佈局**＝尚未明顯過熱，但多個領先條件正在改善；**🟢 買進條件成立**＝主要條件同時偏多；**🔵 突破確認**＝突破型態成立；**🟡 等待**＝條件尚未完整；**🔴 不買**＝目前條件偏弱；**🟠 不追高**＝位置過熱。")
            st.caption("排序只是依照本工具的規則分組與分數排序，不代表未來報酬排名。現在買進訊號另加 2R 風險報酬閘門：上方第一壓力空間不足時，不會因分數高就硬列買進。")

with tab_search:
    st.subheader("🔎 查詢指定個股")
    q=st.text_area("輸入代號或名稱（可一次多檔）",placeholder="2330\n2317\n2454",height=90)
    if st.button("🔍 分析指定股票",width="stretch"):
        with st.spinner("正在載入股票、法人與營收資料…"):
            stocks,inst_map,inst_date,rev_df=load_core_data()
        if stocks.empty:
            st.error("目前抓不到台股清單，請稍後再試。")
            st.stop()
        code_to_name=dict(zip(stocks["代號"],stocks["名稱"]))
        name_to_code={v:k for k,v in code_to_name.items()}
        codes=[]
        for t in parse_tokens(q):
            if t in code_to_name: codes.append(t)
            elif t in name_to_code: codes.append(name_to_code[t])
            elif t.isdigit(): codes.append(t)
            else:
                m=[c for c,n in code_to_name.items() if t in n]
                if m: codes.append(m[0])
        codes=list(dict.fromkeys(codes))
        if not codes: st.error("找不到有效股票。")
        else: st.session_state["manual_results"]=run_manual(codes,stocks,inst_map,rev_df)
    if st.session_state.get("manual_results"):
        st.dataframe(pd.DataFrame([result_row(r) for r in st.session_state["manual_results"]]),width="stretch",hide_index=True)

with tab_portfolio:
    st.subheader("🛡️ 持倉／出場管理")
    st.markdown("**目的：已經買進的股票，不只看它會不會漲，也要知道什麼時候該停利、停損，避免一路抱到貪心變回吐。**")
    st.info("輸入格式：一行一檔，使用逗號分隔 `代號,成本價,股數`。例如 `2330,1200,1000`。系統會直接重新抓你輸入股票的最新價格、K線、法人與雷達風控；**不要求它先進每日候選池**。")
    portfolio_text = st.text_area("目前持倉", placeholder="2330,1200,1000\n2454,980,2000", height=120, key="portfolio_text")
    if st.button("🔎 檢查持倉／出場訊號", type="primary", width="stretch"):
        # 持倉管理與每日選股完全分離：持有哪一檔，就直接分析哪一檔。
        # 持倉按鈕是獨立入口；若本次 Session 尚未載入核心資料，這裡主動載入，
        # 避免 stocks 尚未初始化成 None 時直接呼叫 .empty 導致 AttributeError。
        stocks, inst_map, inst_date, rev_df = get_core_data()
        if stocks is None or stocks.empty:
            stocks, inst_map, inst_date, rev_df = load_core_data()
        if stocks is None or stocks.empty:
            st.error("目前無法取得股票清單，請稍後再試。")
        else:
            code_to_name = dict(zip(stocks["代號"].astype(str), stocks["名稱"]))
            positions=[]
            errors=[]
            lines=(portfolio_text or "").splitlines()

            for line_no, line in enumerate(lines, 1):
                parts=[x.strip() for x in re.split(r"[,，、;；]+", line.strip()) if x.strip()]
                if len(parts)<3:
                    if line.strip(): errors.append(f"第 {line_no} 行格式錯誤：請輸入 代號,成本價,股數")
                    continue

                code=str(parts[0])
                try:
                    cost=float(parts[1]); shares=float(parts[2])
                except Exception:
                    errors.append(f"第 {line_no} 行成本／股數不是數字")
                    continue

                name=code_to_name.get(code, "")
                if not name:
                    errors.append(f"{code} 找不到股票名稱，請確認代號")
                    continue

                try:
                    r=analyze(code, name, inst_map, rev_df)
                except Exception as exc:
                    r=None
                    errors.append(f"{code} 分析失敗：{type(exc).__name__}")

                if not r:
                    errors.append(f"{code} 暫時取得不到足夠的價格／K線資料")
                    continue

                row=result_row(r)
                price=num(r.get("close"))
                stop=num(row.get("停損"))
                t1=num(row.get("目標1"))
                t2=num(row.get("目標2"))
                radar=str(row.get("判斷", ""))
                pnl=(price/cost-1)*100 if cost and pd.notna(price) else np.nan

                # 持倉頁與「新進場」分開處理：新進場可以要求 2R，
                # 但已買進的股票不能只看目前結構目標，還要對照自己的成本，
                # 避免出現「成本18.92、目標1卻只有19.12」卻被誤解成很好的停利目標。
                cost_to_t1=((t1/cost-1)*100) if cost and pd.notna(t1) else np.nan
                cost_to_t2=((t2/cost-1)*100) if cost and pd.notna(t2) else np.nan
                target1_too_close=bool(pd.notna(cost_to_t1) and cost_to_t1 < 3.0)

                current_to_t1=((t1/price-1)*100) if price and pd.notna(t1) else np.nan
                current_to_stop=((price/stop-1)*100) if stop and pd.notna(price) else np.nan

                # 已有明顯獲利時，不能繼續只看原始停損；否則像 +130% 的持倉，
                # 仍可能顯示「持有觀察」，讓大幅獲利回吐。建立「獲利保護線」供持倉管理。
                d=r.get("data", pd.DataFrame())
                last=d.iloc[-1] if isinstance(d,pd.DataFrame) and not d.empty else None
                ma20=num(last.get("MA20")) if last is not None else np.nan
                atr=num(last.get("ATR14")) if last is not None else np.nan
                recent_low=np.nan
                try:
                    if isinstance(d,pd.DataFrame) and len(d)>=20:
                        recent_low=float(d["Low"].tail(20).min())
                except Exception:
                    recent_low=np.nan
                profit_protect=np.nan
                if pd.notna(pnl) and pnl >= 10 and pd.notna(price):
                    candidates=[]
                    if pd.notna(stop): candidates.append(float(stop))
                    # 至少保護一部分既有獲利；同時參考 MA20/ATR 與近20日低點。
                    candidates.append(float(cost)*1.05)
                    if pd.notna(ma20) and pd.notna(atr) and atr>0:
                        candidates.append(float(ma20)-0.75*float(atr))
                    if pd.notna(recent_low): candidates.append(float(recent_low))
                    profit_protect=max(candidates) if candidates else np.nan
                protect_gap=((price/profit_protect-1)*100) if pd.notna(profit_protect) and profit_protect>0 else np.nan

                if pd.notna(stop) and price <= stop:
                    status="🔴 出場警示"
                    reason="已到／跌破系統防守價。這是風險控制訊號，不是預測股價一定會繼續跌。"
                elif pd.notna(t2) and price >= t2:
                    status="🟠 目標2達成"
                    reason="已到第二結構目標區；可重新檢查趨勢與防守線，避免因還可能上漲而失去既有獲利。"
                elif pd.notna(t1) and price >= t1:
                    status="🟡 第一壓力到達"
                    reason=(f"現價已到第一結構壓力；從成本看仍有約{cost_to_t1:.1f}%空間。"
                            "這是壓力區，不是強制賣出點；若突破且量價配合，再重新評估下一目標。"
                            if pd.notna(cost_to_t1) else "現價已到第一結構壓力，觀察突破或轉弱。")
                elif r.get("bearish"):
                    patterns="、".join(r.get("bearish",[])[:3])
                    status="🟠 K線出場警戒"
                    reason=f"偵測到轉弱／反轉型態：{patterns}。這是警戒，不代表單一K線就必須賣出；應搭配停損與整體雷達。"
                elif pd.notna(pnl) and pnl >= 30 and radar == "🟠 不追高":
                    status="🟠 高獲利防守"
                    reason=(f"目前已有{pnl:.1f}%獲利，但雷達判定為不追高；這時重點不是再追目標，而是保護已經賺到的部位。"
                            + (f" 目前建議關注獲利保護線約{profit_protect:.2f}，距現價約{protect_gap:.1f}%。" if pd.notna(profit_protect) else ""))
                elif pd.notna(pnl) and pnl > 0 and radar in ("🔴 不買", "🟡 等待", "🟠 不追高"):
                    status="⚠️ 獲利部位轉弱"
                    reason=(f"目前仍有{pnl:.1f}%獲利，但雷達已不是強勢進場訊號；不要只因為還在賺就忽略趨勢惡化。"
                            + (f" 建議把獲利保護線放在約{profit_protect:.2f}附近，避免大幅回吐。" if pd.notna(profit_protect) else ""))
                elif pd.notna(pnl) and pnl < 0 and target1_too_close:
                    status="🟡 反彈觀察"
                    reason=(f"目前虧損{abs(pnl):.1f}%；最近結構壓力約{t1:.2f}，距離你的成本只有{cost_to_t1:.1f}%，"
                            f"但距離目前股價約{current_to_t1:.1f}%。因此『反彈觀察』的意思是：先觀察股價能否反彈到壓力並有效突破，"
                            "不是叫你現在賣，也不是把這個壓力當成漂亮的停利目標。")
                elif radar == "🟡 等待":
                    status="🟡 持有觀察"
                    reason="目前沒有觸發停損或明確出場K線，但雷達尚未重新轉強；先觀察支撐、量價與後續雷達變化。"
                else:
                    status="🟢 持有觀察"
                    reason="尚未觸發系統停損／目標，也沒有新的K線轉弱警戒；依原風控計畫觀察。"

                positions.append({
                    "代號":code,"名稱":name,"狀態":status,
                    "現價":round(price,2) if pd.notna(price) else np.nan,
                    "成本":cost,"損益%":round(pnl,2) if pd.notna(pnl) else np.nan,"股數":shares,
                    "停損":stop,"獲利保護線":round(profit_protect,2) if pd.notna(profit_protect) else np.nan,"目標1":t1,"目標2":t2,
                    "目標1依據":row.get("目標1依據", ""),"目標2依據":row.get("目標2依據", ""),
                    "現價→目標1%":round(current_to_t1,2) if pd.notna(current_to_t1) else np.nan,
                    "成本→目標1%":round(cost_to_t1,2) if pd.notna(cost_to_t1) else np.nan,
                    "成本→目標2%":round(cost_to_t2,2) if pd.notna(cost_to_t2) else np.nan,
                    "雷達":radar,"理由":reason
                })

            if errors:
                for e in errors: st.warning(e)

            if positions:
                pdf=pd.DataFrame(positions)
                order={"🔴 出場警示":0,"🟠 目標2達成":1,"🟡 第一壓力到達":2,"🟠 K線出場警戒":3,"🟠 高獲利防守":4,"⚠️ 獲利部位轉弱":5,"🟡 反彈觀察":6,"🟡 持有觀察":7,"🟢 持有觀察":8}
                pdf["_o"]=pdf["狀態"].map(order).fillna(99)
                pdf=pdf.sort_values(["_o","損益%"],ascending=[True,False]).drop(columns="_o")
                # 手機橫向表格容易把「理由」推到最右邊；因此摘要表只放關鍵欄位，
                # 每檔股票再用獨立區塊完整顯示狀態與理由，避免使用者看不到重要訊息。
                summary_cols=["代號","名稱","狀態","現價","成本","損益%","停損","獲利保護線","目標1","目標2","雷達"]
                st.dataframe(pdf[[c for c in summary_cols if c in pdf.columns]], width="stretch", hide_index=True)
                for _, pos in pdf.iterrows():
                    st.markdown(f"### {pos.get('代號','')} {pos.get('名稱','')}｜{pos.get('狀態','')}")
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("現價", f"{num(pos.get('現價')):.2f}" if pd.notna(num(pos.get('現價'))) else "—")
                    c2.metric("成本", f"{num(pos.get('成本')):.2f}" if pd.notna(num(pos.get('成本'))) else "—")
                    c3.metric("損益", f"{num(pos.get('損益%')):.2f}%" if pd.notna(num(pos.get('損益%'))) else "—")
                    c4.metric("現價→目標1", f"{num(pos.get('現價→目標1%')):.2f}%" if pd.notna(num(pos.get('現價→目標1%'))) else "—")
                    st.info(str(pos.get("理由", "")))
                    st.caption(
                        f"停損 {pos.get('停損','—')}｜獲利保護線 {pos.get('獲利保護線','—')}｜目標1 {pos.get('目標1','—')}（{pos.get('目標1依據','')}）｜目標2 {pos.get('目標2','—')}（{pos.get('目標2依據','')}）｜"
                        f"成本→目標1 {pos.get('成本→目標1%','—')}%｜成本→目標2 {pos.get('成本→目標2%','—')}%"
                    )
                st.caption(f"資料時間：{inst_date or '最新可取得資料'}。持倉分析會直接針對輸入股票重新計算，不受每日雷達前50／100／200候選池限制。以上是規則化風控與技術結構判讀，不是保證性指令。")

with tab_detail:
    stocks,inst_map,inst_date,rev_df=get_core_data()
    all_results=st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[])
    # dedupe
    unique={r["code"]:r for r in all_results}
    if not unique:
        st.info("先到「自動選股」或「個股查詢」分析股票。")
    else:
        code=st.selectbox("選擇股票",list(unique.keys()),format_func=lambda x:f"{x}｜{unique[x]['name']}")
        r=unique[code]; s=r["score"]
        if s.get("訊號") in ["🟢 早期佈局","🟢 買進條件成立","🔵 突破確認"]: st.success(f"{s.get('訊號')}｜{s.get('動作','')}")
        elif s.get("訊號")=="🟡 等待": st.warning(f"{s.get('訊號')}｜{s.get('動作','')}")
        else: st.error(f"{s.get('訊號','⚪ 資料不足')}｜{s.get('動作','')}")
        a,b,c,d=st.columns(4); a.metric("雷達分數",s["分數"]); b.metric("收盤",f"{r['close']:.2f}"); c.metric("今日漲跌",f"{r['change_pct']:.2f}%"); d.metric("RSI",f"{r['rsi']:.1f}" if pd.notna(r['rsi']) else "—")
        st.plotly_chart(candle_chart(r),width="stretch")
        l,rr=st.columns(2)
        with l:
            st.markdown("### 🕯️ 自動辨識")
            st.write("**偏多/反轉：** "+("、".join(r["bullish"]) if r["bullish"] else "無"))
            st.write("**偏空：** "+("、".join(r["bearish"]) if r["bearish"] else "無"))
            st.write("**解釋：**")
            for p in (r["bullish"]+r["bearish"])[:10]: st.write(f"- **{p}**：{PATTERN_DESCRIPTIONS.get(p,'依規則偵測')}")
        with rr:
            st.markdown("### 🧠 判斷依據")
            st.write(f"技術分：**{s['技術分']}**｜籌碼分：**{s['籌碼分']}**｜基本面分：**{s['基本面分']}**")
            st.write("主要原因："+("、".join(s["理由"]) if s["理由"] else "條件不足"))
            st.write("風險："+("、".join(s["風險"]) if s["風險"] else "目前未偵測到主要風險"))
        st.subheader("法人買賣超")
        inst=r["institution"]; x,y,z,w=st.columns(4)
        x.metric("外資",f"{_lots(inst.get('外資',np.nan)):,.1f} 張" if pd.notna(inst.get('外資',np.nan)) else "—")
        y.metric("投信",f"{_lots(inst.get('投信',np.nan)):,.1f} 張" if pd.notna(inst.get('投信',np.nan)) else "—")
        z.metric("自營商",f"{_lots(inst.get('自營商',np.nan)):,.1f} 張" if pd.notna(inst.get('自營商',np.nan)) else "—")
        w.metric("三大法人合計",f"{_lots(inst.get('法人合計',np.nan)):,.1f} 張" if pd.notna(inst.get('法人合計',np.nan)) else "—")
        hist=r["inst_stats"]["history"].copy()
        if not hist.empty:
            for col in ["外資","投信","自營商","法人合計"]:
                if col in hist.columns:
                    hist[col]=hist[col].map(_lots)
            hist=hist.rename(columns={"外資":"外資(張)","投信":"投信(張)","自營商":"自營商(張)","法人合計":"法人合計(張)"})
            st.dataframe(hist.sort_values("日期",ascending=False),width="stretch",hide_index=True)
        st.subheader("營收")
        st.dataframe(pd.DataFrame([r["revenue"]]),width="stretch",hide_index=True)

with tab_fin:
    st.subheader("📑 財報 / 法人")
    result_pool=st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[])
    codes=[r["code"] for r in result_pool]
    name_map={r["code"]:r.get("name","") for r in result_pool}
    codes=list(dict.fromkeys(codes))
    if not codes: st.info("先分析至少一檔股票，再查看財報。")
    else:
        c=st.selectbox("選擇股票",codes,key="fin_code",format_func=lambda x:f"{x}｜{name_map.get(x,'')}")
        ticker=f"{c}.TW"
        st.write("### 法人最近交易日")
        inst=next((r["institution"] for r in st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[]) if r["code"]==c),{})
        st.dataframe(pd.DataFrame([inst]),width="stretch",hide_index=True)
        st.write("### 財務報表（公開資料，來源依 yfinance 可取得內容）")
        fin=financials_yfinance(ticker)
        if fin.empty: st.warning("目前抓不到財報資料，可能是資料源暫時沒有回應。")
        else: st.dataframe(fin,width="stretch")
        st.caption(f"法人最近可用日期：{inst_date or '未取得'}；股數正值＝買超，負值＝賣超。")

with tab_patterns:
    st.subheader("🕯️ K線型態庫")
    st.write("系統會把下列型態轉成規則，並在個股分析時自動標記。圖像為你提供的參考圖庫；真正判斷以OHLC數據規則為準。")
    assets=[]
    import os
    for f in sorted(os.listdir("assets/patterns")):
        if f.lower().endswith((".png",".jpg",".jpeg")): assets.append(f)
    if assets:
        cols=st.columns(3)
        for i,f in enumerate(assets): cols[i%3].image("assets/patterns/"+f,width="stretch",caption=f)
    st.markdown("### 已納入自動判斷的型態")
    st.dataframe(pd.DataFrame([{"型態":k,"系統解讀":v} for k,v in PATTERN_DESCRIPTIONS.items()]),width="stretch",hide_index=True)

st.sidebar.caption(f"法人資料最近可用日：{st.session_state.get('inst_date') or '—'}")
