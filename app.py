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
    s=r["score"]; inst=r["institution"]
    return {"代號":r["code"],"名稱":r["name"],"收盤":round(r["close"],2),"漲跌%":round(r["change_pct"],2),"法人買賣超(張)":_lots(inst.get("法人合計",np.nan)),"法人5日(張)":_lots(r["inst_stats"].get("5日",np.nan)),"法人20日(張)":_lots(r["inst_stats"].get("20日",np.nan)),"RSI":round(r["rsi"],1) if pd.notna(r["rsi"]) else np.nan,"雷達分數":s["分數"],"判斷":s["訊號"],"動作":s["動作"],"進場參考":round(s["進場參考"],2),"停損":round(s["停損參考"],2),"目標1":round(s["目標1"],2),"目標2":round(s["目標2"],2),"風險報酬":round(s["風險報酬"],2) if pd.notna(s["風險報酬"]) else np.nan,"K線訊號":"、".join((r["bullish"][:3]+r["bearish"][:2]))}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ 選股設定")
    st.write("系統會先縮小候選池，再逐檔分析K線、量能、法人與營收，避免一次查詢全市場造成逾時。")
    universe=st.selectbox("候選股範圍",["成交金額前 50","成交金額前 100","成交金額前 200","全部市場（較慢）"])
    min_score=st.slider("最低雷達分數",40,90,65)
    only_bull=st.checkbox("只顯示偏多/再等等",False)
    max_scan=st.slider("最多實際分析檔數",20,200,80,step=10)
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
tab_auto,tab_search,tab_detail,tab_fin,tab_patterns=st.tabs(["🚀 自動選股","🔎 個股查詢","📊 詳細分析","📑 財報/法人","🕯️ K線型態庫"])

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
        rows=[]
        for sig, v in sig_stats.items():
            rows.append({"訊號":sig,"訊號數":v.get("count",0),"已完成5日":v.get("5d_count",0),"5日正報酬率":v.get("5d_positive_rate"),"5日平均報酬":v.get("5d_avg_return")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("績效由每日收盤後自動記錄；1/5/20 日績效需等實際交易日經過後才會補齊。")

    if "auto_table" not in st.session_state:
        daily_df, daily_time = load_daily_radar_cache()
        if not daily_df.empty:
            show_df = daily_df.copy()
            if only_bull and "判斷" in show_df.columns:
                show_df = show_df[show_df["判斷"].isin(["🟢 買進條件成立", "🔵 突破確認", "🟡 等待"])]
            if "雷達分數" in show_df.columns:
                show_df = show_df[pd.to_numeric(show_df["雷達分數"], errors="coerce") >= min_score]
                show_df = show_df.sort_values("雷達分數", ascending=False)
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
            if only_bull: df=df[df["判斷"].isin(["可研究","再等等"])]
            df=df[df["雷達分數"]>=min_score]
            st.session_state["auto_results"]=results
            st.session_state["auto_table"]=df
            st.success(f"掃描完成：成功分析 {len(results)} 檔，符合目前分數條件 {len(df)} 檔。")
        else: st.error("沒有取得足夠資料，請稍後再試。")
    if "auto_table" in st.session_state:
        df=st.session_state["auto_table"]
        st.dataframe(df,width="stretch",hide_index=True)
        if not df.empty:
            st.markdown("### 自動選股解讀")
            st.write("**可研究**＝同時有較多技術/籌碼/基本面正向條件；**再等等**＝有訊號但缺確認；**偏弱**＝目前條件較弱。")
            st.caption("排序只是依照本工具的規則分數排序，不代表未來報酬排名。")

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
        if s["訊號"]=="可研究": st.success(f"🟢 {s['訊號']}｜{s['動作']}")
        elif s["訊號"]=="再等等": st.warning(f"🟡 {s['訊號']}｜{s['動作']}")
        else: st.error(f"🔴 {s['訊號']}｜{s['動作']}")
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
