import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from kline_engine import prepare, detect_patterns, bearish_patterns, score, PATTERN_DESCRIPTIONS
from data_sources import stock_list_all, latest_institutional, institutional_summary, institutional_bulk_summary, revenue_table, revenue_for, financials_yfinance

st.set_page_config(page_title="台股雷達 PRO｜自動選股", page_icon="📈", layout="wide")

st.title("📈 台股雷達 PRO")
st.caption("自動選股 × K線自動辨識 × 法人買賣超 × 營收 × 財報 × 技術雷達")

# ---------- Data ----------
@st.cache_data(ttl=1800, show_spinner=False)
def price_history(code, period="1y"):
    for suffix in [".TW", ".TWO"]:
        try:
            d=yf.Ticker(f"{code}{suffix}").history(period=period,interval="1d",auto_adjust=False)
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

def result_row(r):
    s=r["score"]; inst=r["institution"]
    return {"代號":r["code"],"名稱":r["name"],"收盤":round(r["close"],2),"漲跌%":round(r["change_pct"],2),"法人買賣超(股)":inst.get("法人合計",np.nan),"法人5日(股)":r["inst_stats"].get("5日",np.nan),"法人20日(股)":r["inst_stats"].get("20日",np.nan),"RSI":round(r["rsi"],1) if pd.notna(r["rsi"]) else np.nan,"雷達分數":s["分數"],"判斷": {"可研究":"可以", "再等等":"再等等", "偏弱":"不可以"}.get(s["訊號"], s["訊號"]),"K線訊號":"、".join((r["bullish"][:3]+r["bearish"][:2])) or "無明顯型態"}

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

stocks=stocks_cached()
inst_map,inst_date=inst_cached()
rev_df=revenue_cached()
if stocks.empty:
    st.error("目前抓不到台股清單，請稍後重新整理。")
    st.stop()

# ---------- Tabs ----------
tab_auto,tab_search,tab_detail,tab_fin=st.tabs(["🚀 自動選股","🔎 個股查詢","📊 詳細分析","📑 財報/法人"])

with tab_auto:
    st.subheader("🚀 全自動選股｜直接告訴你：可以 / 再等等 / 不可以")
    st.markdown("**目標：不用先告訴系統股票代號，由系統自己從市場候選池找出符合條件的股票。**")
    nmap={"成交金額前 50":50,"成交金額前 100":100,"成交金額前 200":200,"全部市場（較慢）":len(stocks)}
    limit=min(nmap[universe],max_scan)
    st.info(f"目前候選池：{nmap[universe]} 檔；本次最多深度分析 {limit} 檔。篩選會優先處理成交金額較大的股票。")
    if st.button("🚀 開始全市場自動選股",type="primary",use_container_width=True):
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
            if only_bull: df=df[df["判斷"].isin(["可以","再等等"])]
            df=df[df["雷達分數"]>=min_score]
            st.session_state["auto_results"]=results
            st.session_state["auto_table"]=df
            st.success(f"掃描完成：成功分析 {len(results)} 檔，符合目前分數條件 {len(df)} 檔。")
        else: st.error("沒有取得足夠資料，請稍後再試。")
    if "auto_table" in st.session_state:
        df=st.session_state["auto_table"]
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.markdown("### 判斷規則")
        st.write("🟢 **可以**：技術面、量能、法人/基本面綜合條件達標。\n🟡 **再等等**：有轉強訊號，但突破、量能或籌碼尚未確認。\n🔴 **不可以**：目前偏弱或風險訊號較多。")
        if not df.empty:
            st.markdown("### 自動選股解讀")
            st.write("**可以**＝規則條件大致轉強；**再等等**＝有訊號但缺確認；**不可以**＝目前條件偏弱。")
            st.caption("排序只代表本工具的規則分數；訊號不是獲利保證，仍需自行確認風險。")

with tab_search:
    st.subheader("🔎 查詢指定個股")
    q=st.text_area("輸入代號或名稱（可一次多檔）",placeholder="2330\n2317\n2454",height=90)
    if st.button("🔍 分析指定股票",use_container_width=True):
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
        st.dataframe(pd.DataFrame([result_row(r) for r in st.session_state["manual_results"]]),use_container_width=True,hide_index=True)

with tab_detail:
    all_results=st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[])
    # dedupe
    unique={r["code"]:r for r in all_results}
    if not unique:
        st.info("先到「自動選股」或「個股查詢」分析股票。")
    else:
        code=st.selectbox("選擇股票",list(unique.keys()),format_func=lambda x:f"{x}｜{unique[x]['name']}")
        r=unique[code]; s=r["score"]
        display_signal={"可研究":"可以","再等等":"再等等","偏弱":"不可以"}.get(s["訊號"],s["訊號"])
        if display_signal=="可以": st.success(f"🟢 {display_signal}｜{s['動作']}")
        elif display_signal=="再等等": st.warning(f"🟡 {display_signal}｜{s['動作']}")
        else: st.error(f"🔴 {display_signal}｜{s['動作']}")
        a,b,c,d=st.columns(4); a.metric("雷達分數",s["分數"]); b.metric("收盤",f"{r['close']:.2f}"); c.metric("今日漲跌",f"{r['change_pct']:.2f}%"); d.metric("RSI",f"{r['rsi']:.1f}" if pd.notna(r['rsi']) else "—")
        st.plotly_chart(candle_chart(r),use_container_width=True)
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
        x.metric("外資",f"{inst.get('外資',np.nan):,.0f}" if pd.notna(inst.get('外資',np.nan)) else "—")
        y.metric("投信",f"{inst.get('投信',np.nan):,.0f}" if pd.notna(inst.get('投信',np.nan)) else "—")
        z.metric("自營商",f"{inst.get('自營商',np.nan):,.0f}" if pd.notna(inst.get('自營商',np.nan)) else "—")
        w.metric("三大法人合計",f"{inst.get('法人合計',np.nan):,.0f}" if pd.notna(inst.get('法人合計',np.nan)) else "—")
        hist=r["inst_stats"]["history"]
        if not hist.empty:
            st.dataframe(hist.sort_values("日期",ascending=False),use_container_width=True,hide_index=True)
        st.subheader("營收")
        st.dataframe(pd.DataFrame([r["revenue"]]),use_container_width=True,hide_index=True)

with tab_fin:
    st.subheader("📑 財報 / 法人")
    codes=[r["code"] for r in (st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[]))]
    codes=list(dict.fromkeys(codes))
    if not codes: st.info("先分析至少一檔股票，再查看財報。")
    else:
        c=st.selectbox("選擇股票",codes,key="fin_code",format_func=lambda x:f"{x}｜{dict(zip(stocks['代號'],stocks['名稱'])).get(x,'')}")
        ticker=f"{c}.TW"
        st.write("### 法人最近交易日")
        inst=next((r["institution"] for r in st.session_state.get("manual_results",[])+st.session_state.get("auto_results",[]) if r["code"]==c),{})
        st.dataframe(pd.DataFrame([inst]),use_container_width=True,hide_index=True)
        st.write("### 財務報表（公開資料，來源依 yfinance 可取得內容）")
        fin=financials_yfinance(ticker)
        if fin.empty: st.warning("目前抓不到財報資料，可能是資料源暫時沒有回應。")
        else: st.dataframe(fin,use_container_width=True)
        st.caption(f"法人最近可用日期：{inst_date or '未取得'}；股數正值＝買超，負值＝賣超。")
