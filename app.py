import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="台股法人 × 財報 × 技術雷達", page_icon="📈", layout="wide")

st.title("📈 台股法人 × 財報 × 技術雷達")
st.caption("法人籌碼 × 營收 × 技術面 × K線型態｜資料抓不到時明確顯示，不把未知當 0")

HEADERS = {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Safari/605.1.15"}

def num(v):
    try:
        s = str(v).replace(",", "").replace("%", "").strip()
        if s in ("", "-", "--", "—", "nan", "None"): return np.nan
        return float(s)
    except: return np.nan

@st.cache_data(ttl=1800)
def get_json(url, params=None, timeout=5):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ---------- 股票清單 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_list():
    fallback = {
        "1101":"台泥","1102":"亞泥","1216":"統一","1301":"台塑","1303":"南亞",
        "2002":"中鋼","2207":"和泰車","2303":"聯電","2308":"台達電","2317":"鴻海",
        "2330":"台積電","2344":"華邦電","2357":"華碩","2379":"瑞昱","2382":"廣達",
        "2454":"聯發科","2603":"長榮","2609":"陽明","2615":"萬海","2881":"富邦金",
        "2882":"國泰金","2884":"玉山金","2886":"兆豐金","2891":"中信金",
        "3008":"大立光","3034":"聯詠","3037":"欣興","3231":"緯創","3443":"創意",
        "3711":"日月光投控","4938":"和碩","6669":"緯穎","8046":"南電","9904":"寶成"
    }
    data = get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=6)
    rows=[]
    if isinstance(data, list):
        for x in data:
            c=str(x.get("Code","")).strip(); n=str(x.get("Name","")).strip()
            if c.isdigit() and n: rows.append({"代號":c,"名稱":n})
    if rows:
        return pd.DataFrame(rows).drop_duplicates("代號")
    return pd.DataFrame([{"代號":c,"名稱":n} for c,n in fallback.items()])

# ---------- 法人：TWT38U + T86 fallback ----------
def parse_twt38(data):
    if not isinstance(data,list) or not data: return {}
    fields=list(data[0].keys())
    def find(keys):
        for k in keys:
            if k in fields: return k
        for f in fields:
            if any(k in str(f) for k in keys): return f
        return None
    cf=find(["證券代號"])
    ff=find(["外陸資買賣超股數(不含外資自營商)","外資買賣超股數"])
    tf=find(["投信買賣超股數"])
    df=find(["自營商買賣超股數(自行買賣)","自營商買賣超股數"])
    out={}
    if not cf: return out
    for x in data:
        c=str(x.get(cf,"")).strip()
        if not c: continue
        f=num(x.get(ff)) if ff else np.nan
        t=num(x.get(tf)) if tf else np.nan
        d=num(x.get(df)) if df else np.nan
        vals=[f,t,d]
        total=sum(v for v in vals if pd.notna(v)) if any(pd.notna(v) for v in vals) else np.nan
        out[c]={"外資":f,"投信":t,"自營商":d,"法人合計":total}
    return out

def parse_t86(data):
    # T86 JSON: {"data":[...],"fields":[...]}
    if not isinstance(data,dict): return {}
    fields=data.get("fields",[])
    rows=data.get("data",[])
    if not fields or not rows: return {}
    def fi(keys):
        for k in keys:
            if k in fields: return fields.index(k)
        for i,f in enumerate(fields):
            if any(k in str(f) for k in keys): return i
        return None
    ci=fi(["證券代號"])
    # T86 typically has foreign buy/sell, trust, dealer columns; use column names rather than position.
    fbuy=fi(["外陸資買賣超股數(不含外資自營商)","外陸資買賣超股數"])
    t=fi(["投信買賣超股數"])
    d=fi(["自營商買賣超股數"])
    out={}
    if ci is None: return out
    for row in rows:
        if len(row)<=ci: continue
        c=str(row[ci]).strip()
        if not c: continue
        f=num(row[fbuy]) if fbuy is not None and len(row)>fbuy else np.nan
        tv=num(row[t]) if t is not None and len(row)>t else np.nan
        dv=num(row[d]) if d is not None and len(row)>d else np.nan
        vals=[f,tv,dv]
        total=sum(v for v in vals if pd.notna(v)) if any(pd.notna(v) for v in vals) else np.nan
        out[c]={"外資":f,"投信":tv,"自營商":dv,"法人合計":total}
    return out

@st.cache_data(ttl=1800)
def institutional_latest():
    data=get_json("https://openapi.twse.com.tw/v1/exchangeReport/TWT38U")
    out=parse_twt38(data)
    if out: return out, "TWSE OpenAPI TWT38U"
    # Find latest available trading date, looking back 10 calendar days.
    for i in range(4):
        d=(datetime.now()-timedelta(days=i)).strftime("%Y%m%d")
        data=get_json("https://www.twse.com.tw/rwd/zh/fund/T86",
                      {"date":d,"selectType":"ALLBUT0999","response":"json"}, timeout=6)
        out=parse_t86(data)
        if out: return out, f"TWSE T86 {d}"
    return {}, "未取得"

@st.cache_data(ttl=900, show_spinner=False)
def institutional_day(date_str):
    data=get_json("https://www.twse.com.tw/rwd/zh/fund/T86",
                  {"date":date_str,"selectType":"ALLBUT0999","response":"json"}, timeout=5)
    return parse_t86(data)

@st.cache_data(ttl=900, show_spinner=False)
def institutional_history_many(codes, lookback=20):
    """一次並行取得最近交易日法人資料，避免每支股票重複打 20 次 API。"""
    codes=tuple(str(c) for c in codes)
    wanted=set(codes)
    rows_by_code={c:[] for c in codes}
    dates=[]
    end=datetime.now()
    for i in range(35):
        d=end-timedelta(days=i)
        if d.weekday()<5:
            dates.append(d.strftime("%Y%m%d"))
        if len(dates)>=lookback+5:
            break

    def fetch(ds):
        return ds, institutional_day(ds)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures=[ex.submit(fetch, ds) for ds in dates]
        for fut in as_completed(futures):
            try:
                ds,mp=fut.result()
            except Exception:
                continue
            if not mp:
                continue
            for code in wanted:
                if code in mp and len(rows_by_code[code])<lookback:
                    r=mp[code].copy(); r["日期"]=ds
                    rows_by_code[code].append(r)

    out={}
    for code,rows in rows_by_code.items():
        if rows:
            out[code]=pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)
        else:
            out[code]=pd.DataFrame()
    return out

def institution_stats_from_history(h):
    if h is None or h.empty:
        return {"法人連買":0,"法人連賣":0,"法人5日":np.nan,"法人20日":np.nan}
    total=pd.to_numeric(h["法人合計"],errors="coerce")
    buy=sell=0
    for v in total.iloc[::-1]:
        if pd.isna(v) or v==0: break
        if v>0:
            if sell: break
            buy+=1
        else:
            if buy: break
            sell+=1
    return {"法人連買":buy,"法人連賣":sell,
            "法人5日":total.tail(5).sum(min_count=1),
            "法人20日":total.tail(20).sum(min_count=1)}

def institution_stats(code, latest):
    h=institutional_history_many((code,),20).get(code,pd.DataFrame())
    return institution_stats_from_history(h)

# ---------- 營收 ----------
@st.cache_data(ttl=3600)
def get_revenue():
    data=get_json("https://openapi.twse.com.tw/v1/opendata/t187ap05_L")
    return pd.DataFrame(data) if isinstance(data,list) else pd.DataFrame()

def revenue_for(code, rev):
    if rev.empty: return {"營收":"—","YoY":"—","MoM":"—","累計":"—"}
    cc=[c for c in rev.columns if "公司代號" in str(c) or "證券代號" in str(c)]
    if not cc: return {"營收":"—","YoY":"—","MoM":"—","累計":"—"}
    code_col=cc[0]
    code_s=rev[code_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    target=str(code).strip()
    rows=rev[code_s.isin({target, target.zfill(4)})]
    if rows.empty: return {"營收":"—","YoY":"—","MoM":"—","累計":"—"}
    r=rows.iloc[-1]
    def pick(keys):
        for c in rev.columns:
            s=str(c)
            if any(k in s for k in keys): return r[c]
        return "—"
    return {"營收":pick(["當月營收","本月營收"]), "YoY":pick(["前期比較增減(%)","年增率","年增"]), 
            "MoM":pick(["上月比較增減(%)","月增率","月增"]), "累計":pick(["累計營收"])}

# ---------- 技術 ----------
@st.cache_data(ttl=1800, show_spinner=False)
def price_history(code, period="1y"):
    try:
        d = yf.download(
            f"{code}.TW", period=period, interval="1d",
            auto_adjust=False, progress=False, timeout=8
        )
        if d is None or d.empty:
            return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols=["Open","High","Low","Close","Volume"]
        if not all(c in d.columns for c in cols):
            return pd.DataFrame()
        return d[cols].dropna()
    except Exception:
        return pd.DataFrame()

def technical(d):
    d=d.copy()
    for n in [5,10,20,60]: d[f"MA{n}"]=d.Close.rolling(n).mean()
    d["V5"]=d.Volume.rolling(5).mean()
    delta=d.Close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
    e12=d.Close.ewm(span=12,adjust=False).mean(); e26=d.Close.ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26; d["MACDSignal"]=d.MACD.ewm(span=9,adjust=False).mean()
    mid=d.Close.rolling(20).mean(); sd=d.Close.rolling(20).std()
    d["BBUpper"]=mid+2*sd; d["BBLower"]=mid-2*sd
    return d

def patterns(d):
    p=[]
    if len(d)<10:return p
    a,b=d.iloc[-2],d.iloc[-1]; body=abs(b.Close-b.Open); rng=max(b.High-b.Low,1e-9)
    lower=min(b.Open,b.Close)-b.Low; upper=b.High-max(b.Open,b.Close)
    if lower>body*2 and lower>upper*1.5 and b.Close>b.Low+rng*.5:p.append("金針探底")
    if len(d)>=3:
        x=d.iloc[-3:]; c1,c2,c3=x.iloc[0],x.iloc[1],x.iloc[2]
        if all(r.Close>r.Open for _,r in x.iterrows()):p.append("紅三兵")
        if c1.Close<c1.Open and abs(c2.Close-c2.Open)<abs(c1.Close-c1.Open)*.5 and c3.Close>c3.Open and c3.Close>(c1.Open+c1.Close)/2:p.append("早晨之星")
        l1=min(c1.Open,c1.Close)-c1.Low; l2=min(c2.Open,c2.Close)-c2.Low
        if l1>abs(c1.Close-c1.Open)*1.5 and l2>abs(c2.Close-c2.Open)*1.5:p.append("雙針探底")
    if a.Close<a.Open and b.Close>b.Open and b.Close>a.Close:p.append("反攻")
    if a.Close<a.Open and b.Close>b.Open and b.Close>(a.Open+a.Close)/2:p.append("曙光出現")
    if len(d)>=20:
        if b.Close>d.High.iloc[-21:-1].max():p.append("突破前高")
        if pd.notna(b.V5) and b.Close>b.Open and b.Volume>b.V5*1.5:p.append("量增紅K")
        if b.MA5>b.MA10>b.MA20:p.append("均線多頭")
        if d.MA5.iloc[-2]<=d.MA20.iloc[-2] and b.MA5>b.MA20:p.append("均線黃金交叉")
        if b.Close>b.MA5 and b.Close>b.MA10 and b.Close>b.MA20 and a.Close<a.MA20:p.append("一陽穿三線")
        if pd.notna(b.V5) and b.Volume>b.V5*2:p.append("爆量")
    if len(d)>=60 and b.Close>b.MA5 and b.Close>b.MA10 and b.Close>b.MA20 and b.Close>b.MA60 and a.Close<a.MA20:p.append("出水芙蓉")
    return p

# ---------- 訊號分類 ----------
def signal_label(row):
    """
    純規則化技術/籌碼訊號，不代表保證獲利。
    「可買」是符合預設條件的研究訊號；「注意」代表條件不足或風險較高。
    """
    score = float(row.get("雷達分數", 0))
    rsi = row.get("RSI", np.nan)
    total = row.get("法人合計", np.nan)
    patterns_text = str(row.get("K線型態", ""))

    bullish = any(x in patterns_text for x in [
        "突破前高", "出水芙蓉", "一陽穿三線",
        "均線多頭", "均線黃金交叉", "紅三兵",
        "早晨之星", "曙光出現", "反攻", "金針探底"
    ])

    overheat = pd.notna(rsi) and rsi >= 75
    institution_ok = pd.notna(total) and total > 0

    if score >= 75 and bullish and not overheat and institution_ok:
        return "可買"

    return "注意"


def signal_reason(row):
    reasons = []
    if float(row.get("雷達分數", 0)) >= 75:
        reasons.append("雷達分數達75以上")
    if pd.notna(row.get("法人合計", np.nan)) and row.get("法人合計", 0) > 0:
        reasons.append("三大法人合計買超")
    if row.get("法人連買", 0) and row.get("法人連買", 0) >= 3:
        reasons.append(f"法人連買{int(row['法人連買'])}天")
    p = str(row.get("K線型態", ""))
    for x in ["突破前高","出水芙蓉","一陽穿三線","均線多頭","紅三兵","早晨之星"]:
        if x in p:
            reasons.append(x)
    if pd.notna(row.get("RSI", np.nan)) and row.get("RSI", 0) >= 75:
        reasons.append("RSI偏熱")
    return "、".join(reasons[:4]) if reasons else "目前條件不足"


# ---------- 分析 ----------
def analyze(code,name,inst,rev,hist=None):
    d=price_history(code)
    if d.empty or len(d)<60:
        return {"代號":code,"名稱":name,"收盤":np.nan,"外資":inst.get("外資",np.nan),
                "投信":inst.get("投信",np.nan),"自營商":inst.get("自營商",np.nan),
                "法人合計":inst.get("法人合計",np.nan),"法人連買":0,"法人連賣":0,
                "法人5日":np.nan,"法人20日":np.nan,"營收":"—","營收YoY":"—","營收MoM":"—",
                "累計營收":"—","RSI":np.nan,"MACD":np.nan,"K線型態":"資料不足","雷達分數":0}
    d=technical(d); ps=patterns(d)
    stats=institution_stats_from_history(hist if hist is not None else pd.DataFrame())
    rv=revenue_for(code,rev)
    total=inst.get("法人合計",np.nan)
    score=50+min(25,len(ps)*5)
    if pd.notna(total): score += 8 if total>0 else -5 if total<0 else 0
    if stats["法人連買"] and stats["法人連買"]>=3: score+=5
    if pd.notna(d.Close.iloc[-1]) and pd.notna(d.MA20.iloc[-1]): score += 5 if d.Close.iloc[-1]>d.MA20.iloc[-1] else -3
    score=int(max(0,min(100,score)))
    return {"代號":code,"名稱":name,"收盤":round(float(d.Close.iloc[-1]),2),
            "外資":inst.get("外資",np.nan),"投信":inst.get("投信",np.nan),"自營商":inst.get("自營商",np.nan),
            "法人合計":total,"法人連買":stats["法人連買"],"法人連賣":stats["法人連賣"],
            "法人5日":stats["法人5日"],"法人20日":stats["法人20日"],
            "營收":rv["營收"],"營收YoY":rv["YoY"],"營收MoM":rv["MoM"],"累計營收":rv["累計"],
            "RSI":round(float(d.RSI.iloc[-1]),1) if pd.notna(d.RSI.iloc[-1]) else np.nan,
            "MACD":round(float(d.MACD.iloc[-1]),3) if pd.notna(d.MACD.iloc[-1]) else np.nan,
            "K線型態":"、".join(ps) if ps else "—","雷達分數":score}

# ---------- 專業 UI ----------
st.markdown("""
<style>
.main-title{font-size:34px;font-weight:800;margin-bottom:0}
.sub-title{color:#6b7280;margin:2px 0 18px}
.search-box{padding:18px;border:1px solid #e5e7eb;border-radius:16px;background:#fff;box-shadow:0 3px 12px rgba(0,0,0,.05)}
.section{font-size:22px;font-weight:800;margin-top:18px}
.signal-buy{padding:8px 16px;border-radius:999px;font-size:19px;font-weight:800;display:inline-block;background:#e8f7ee;color:#15803d}
.signal-watch{padding:8px 16px;border-radius:999px;font-size:19px;font-weight:800;display:inline-block;background:#fff7df;color:#a16207}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 台股雷達 PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">法人籌碼 × 財報 × 營收 × 技術分析 × K線型態</div>', unsafe_allow_html=True)

# 主畫面直接輸入，不必操作左側選單
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 🔎 輸入股票")
codes_text = st.text_area(
    "股票代號或名稱",
    placeholder="例如：\n2330\n2317\n2454\n\n或輸入：2330,2317,2454",
    height=110,
    label_visibility="collapsed"
)
run = st.button("🔍 開始分析", type="primary", width="stretch")
st.caption("可輸入多檔：一行一檔，或用逗號／空格／頓號分隔；例如 2330, 2317, 2454")
st.markdown('</div>', unsafe_allow_html=True)

# 進階設定留在側欄，日常使用不必碰
with st.sidebar.expander("⚙️ 進階設定", expanded=False):
    scan_market = st.checkbox("掃描 TWSE 股票清單", False)
    limit = st.slider("市場測試檔數", 10, 100, 30)
    min_score = st.slider("最低雷達分數", 0, 100, 50)

if run:
    with st.status("🔄 正在分析，請稍候…", expanded=True) as status:
        st.write("① 取得股票清單…")
        market = get_twse_list()
        st.write("② 取得法人資料…")
        inst_map, inst_source = institutional_latest()
        st.write("③ 取得營收資料…")
        rev = get_revenue()

    name_to_code = {}
    code_to_name = {}
    if not market.empty:
        for _, rr in market.iterrows():
            c = str(rr["代號"]).strip()
            n = str(rr["名稱"]).strip()
            code_to_name[c] = n
            name_to_code[n] = c

    if scan_market:
        codes = market["代號"].head(limit).tolist() if not market.empty else []
        st.info(f"市場模式目前先測試 {len(codes)} 檔。")
    else:
        raw = codes_text.strip()
        if not raw:
            st.warning("請先輸入股票代號或名稱。")
            st.stop()

        # 支援：換行、空格、半形/全形逗號、頓號、分號
        tokens = [
            x.strip()
            for x in re.split(r"[\\s,，、;；]+", raw)
            if x.strip()
        ]

        codes = []
        unresolved = []

        for token in tokens:
            if token in code_to_name:
                codes.append(token)
            elif token in name_to_code:
                codes.append(name_to_code[token])
            else:
                # 支援部分名稱搜尋，例如「台積」
                matches = [c for c, n in code_to_name.items() if token in n]
                if matches:
                    codes.append(matches[0])
                elif token.isdigit():
                    codes.append(token)
                else:
                    unresolved.append(token)

        codes = list(dict.fromkeys(codes))

        if unresolved:
            st.warning("以下項目找不到： " + "、".join(unresolved))

        if not codes:
            st.error("找不到股票。請輸入正確的股票代號或名稱。")
            st.stop()

    st.caption(f"法人資料來源：{inst_source}")

    # 一次並行取得所有查詢股票的法人歷史，避免每檔股票各打 20 次 API。
    with st.status("🏦 正在取得法人歷史資料…", expanded=False):
        history_map = institutional_history_many(tuple(codes), 20)

    results = []
    bar = st.progress(0, text="📈 正在分析股價、技術與 K 線…")

    for i, c in enumerate(codes):
        r = analyze(c, code_to_name.get(c, ""), inst_map.get(c, {}), rev, history_map.get(c, pd.DataFrame()))
        if r:
            r["訊號"] = signal_label(r)
            r["訊號原因"] = signal_reason(r)
            results.append(r)
        bar.progress((i + 1) / max(len(codes), 1))

    bar.empty()

    if results:
        df = pd.DataFrame(results).sort_values("雷達分數", ascending=False)
        show = df[df["雷達分數"] >= min_score].copy()

        buy_count = int((df["訊號"] == "可買").sum())
        watch_count = int((df["訊號"] == "注意").sum())

        a,b,c,d = st.columns(4)
        a.metric("分析股票", len(df))
        b.metric("符合分數", len(show))
        c.metric("🟢 可買", buy_count)
        d.metric("🟡 注意", watch_count)

        st.markdown('<div class="section">📊 分析結果</div>', unsafe_allow_html=True)

        display_cols = [
            "代號","名稱","收盤","訊號","雷達分數",
            "外資","投信","自營商","法人合計",
            "法人連買","法人5日","法人20日",
            "營收YoY","RSI","K線型態"
        ]
        display_cols = [x for x in display_cols if x in show.columns]

        st.dataframe(
            show[display_cols],
            width="stretch",
            hide_index=True
        )

        st.markdown('<div class="section">🔎 個股詳細分析</div>', unsafe_allow_html=True)

        selected = st.selectbox(
            "選擇分析結果",
            df["代號"].tolist(),
            format_func=lambda x: f"{x}｜{code_to_name.get(x, '')}"
        )
        r = df[df["代號"] == selected].iloc[0]

        badge = "signal-buy" if r["訊號"] == "可買" else "signal-watch"
        st.markdown(f'<div class="{badge}">{r["訊號"]}</div>', unsafe_allow_html=True)
        st.caption(f"訊號依據：{r['訊號原因']}")
        st.caption(f"資料狀態｜法人：{inst_source}｜價格：yfinance／TWSE 備援｜營收：TWSE 官方資料")

        a,b,c,d = st.columns(4)
        a.metric("收盤", r["收盤"])
        b.metric("雷達分數", r["雷達分數"])
        c.metric("法人連買", f"{int(r['法人連買'])} 天" if pd.notna(r["法人連買"]) else "—")
        d.metric("RSI", r["RSI"] if pd.notna(r["RSI"]) else "—")

        tab1,tab2,tab3 = st.tabs(["🏦 法人","📈 技術 / K線","💰 營收"])

        with tab1:
            st.write(f"外資：{r['外資'] if pd.notna(r['外資']) else '—'}")
            st.write(f"投信：{r['投信'] if pd.notna(r['投信']) else '—'}")
            st.write(f"自營商：{r['自營商'] if pd.notna(r['自營商']) else '—'}")
            st.write(f"三大法人合計：{r['法人合計'] if pd.notna(r['法人合計']) else '—'}")
            st.write(f"法人連買：{r['法人連買'] if pd.notna(r['法人連買']) else '—'} 天")
            st.write(f"法人5日：{r['法人5日'] if pd.notna(r['法人5日']) else '—'}")
            st.write(f"法人20日：{r['法人20日'] if pd.notna(r['法人20日']) else '—'}")

        with tab2:
            st.write(f"RSI：{r['RSI']}")
            st.write(f"MACD：{r['MACD']}")
            st.write(f"**K線型態：{r['K線型態']}**")
            st.info("訊號為固定規則產生的研究結果；「可買」不代表保證上漲。")

        with tab3:
            st.write(f"當月營收：{r['營收']}")
            st.write(f"YoY：{r['營收YoY']}")
            st.write(f"MoM：{r['營收MoM']}")
            st.write(f"累計營收：{r['累計營收']}")

    else:
        st.warning("沒有取得足夠的歷史資料。")

else:
    st.markdown('<div class="section">📌 使用方式</div>', unsafe_allow_html=True)
    st.markdown("""
**直接在上方輸入：**

- `2330`
- `台積電`
- `2330, 2317, 2454`

按 **🔍 開始分析** 即可。

### 🔥 訊號
🟢 **可買**：符合預設的技術＋法人條件  
🟡 **注意**：條件不足或出現偏熱等風險訊號

### 四大雷達
🏦 法人｜💰 營收｜📈 技術｜🔥 K線型態
""")
