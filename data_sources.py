import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}

def _get_json(url, params=None, timeout=8):
    try:
        r=requests.get(url,params=params,headers=HEADERS,timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception: return None

def _num(v):
    try:
        s=str(v).replace(",","").replace("%","").strip()
        if s in ("","-","--","—","nan","None"): return np.nan
        return float(s)
    except Exception: return np.nan

def twse_stock_list():
    data=_get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
    rows=[]
    if isinstance(data,list):
        for x in data:
            code=str(x.get("Code","")).strip(); name=str(x.get("Name","")).strip()
            if code.isdigit() and name:
                rows.append({"代號":code,"名稱":name,"成交量":_num(x.get("TradeVolume")),"成交金額":_num(x.get("TradeValue"))})
    return pd.DataFrame(rows)

def tpex_stock_list():
    data=_get_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes")
    rows=[]
    if isinstance(data,list):
        for x in data:
            code=str(x.get("SecuritiesCompanyCode",x.get("SecuritiesCompanyCode", ""))).strip()
            name=str(x.get("CompanyName","")).strip()
            if code.isdigit() and name:
                rows.append({"代號":code,"名稱":name,"成交量":_num(x.get("TradingShares")),"成交金額":_num(x.get("TransactionAmount"))})
    return pd.DataFrame(rows)

def stock_list_all():
    a=twse_stock_list(); b=tpex_stock_list()
    frames=[x for x in [a,b] if not x.empty]
    if not frames: return pd.DataFrame(columns=["代號","名稱","成交量","成交金額"])
    d=pd.concat(frames,ignore_index=True).drop_duplicates("代號")
    return d

def _parse_t86(data):
    if not isinstance(data,dict): return {}
    fields=data.get("fields",[]); rows=data.get("data",[])
    if not fields or not rows: return {}
    def idx(keys):
        for k in keys:
            if k in fields: return fields.index(k)
        for i,f in enumerate(fields):
            if any(k in str(f) for k in keys): return i
        return None
    ci=idx(["證券代號"]); fi=idx(["外陸資買賣超股數(不含外資自營商)","外陸資買賣超股數"]); ti=idx(["投信買賣超股數"]); di=idx(["自營商買賣超股數"])
    if ci is None: return {}
    out={}
    for row in rows:
        if len(row)<=ci: continue
        code=str(row[ci]).strip()
        vals=[_num(row[fi]) if fi is not None and len(row)>fi else np.nan,_num(row[ti]) if ti is not None and len(row)>ti else np.nan,_num(row[di]) if di is not None and len(row)>di else np.nan]
        total=sum(v for v in vals if pd.notna(v)) if any(pd.notna(v) for v in vals) else np.nan
        out[code]={"外資":vals[0],"投信":vals[1],"自營商":vals[2],"法人合計":total}
    return out

def institutional_day(date_str):
    return _parse_t86(_get_json("https://www.twse.com.tw/rwd/zh/fund/T86",{"date":date_str,"selectType":"ALLBUT0999","response":"json"}))

def latest_institutional():
    for i in range(15):
        ds=(datetime.now()-timedelta(days=i)).strftime("%Y%m%d"); data=institutional_day(ds)
        if data: return data,ds
    return {},None

def institutional_history(code,days=20):
    rows=[]; found=0
    for i in range(50):
        d=datetime.now()-timedelta(days=i)
        if d.weekday()>=5: continue
        ds=d.strftime("%Y%m%d"); mp=institutional_day(ds)
        if code in mp:
            row=dict(mp[code]); row["日期"]=ds; rows.append(row); found+=1
            if found>=days: break
    return pd.DataFrame(rows).sort_values("日期").reset_index(drop=True) if rows else pd.DataFrame()

def institutional_summary(code):
    h=institutional_history(code,20)
    if h.empty: return {"history":h,"連買":0,"連賣":0,"5日":np.nan,"20日":np.nan}
    s=pd.to_numeric(h["法人合計"],errors="coerce")
    buy=sell=0
    for v in reversed(s.tolist()):
        if pd.isna(v) or v==0: break
        if v>0 and sell==0: buy+=1
        else: break
    for v in reversed(s.tolist()):
        if pd.isna(v) or v==0: break
        if v<0 and buy==0: sell+=1
        else: break
    return {"history":h,"連買":buy,"連賣":sell,"5日":s.tail(5).sum(min_count=1),"20日":s.tail(20).sum(min_count=1)}



def institutional_bulk_summary(codes, days=5):
    """Fetch the latest N available institutional days once per day and summarize many codes.
    This is used by the market scanner to avoid N*days API calls.
    """
    wanted=set(str(c) for c in codes)
    buckets={c:[] for c in wanted}
    found_days=0
    for i in range(20):
        d=datetime.now()-timedelta(days=i)
        if d.weekday()>=5:
            continue
        ds=d.strftime("%Y%m%d")
        mp=institutional_day(ds)
        if not mp:
            continue
        found_days += 1
        for code in wanted:
            if code in mp:
                row=dict(mp[code]); row["日期"]=ds; buckets[code].append(row)
        if found_days>=days:
            break
    out={}
    for code,rows in buckets.items():
        h=pd.DataFrame(rows).sort_values("日期") if rows else pd.DataFrame()
        if h.empty:
            out[code]={"history":h,"連買":0,"連賣":0,"5日":np.nan,"20日":np.nan}
            continue
        s=pd.to_numeric(h["法人合計"],errors="coerce")
        buy=sell=0
        for v in reversed(s.tolist()):
            if pd.isna(v) or v==0: break
            if v>0 and sell==0: buy+=1
            else: break
        for v in reversed(s.tolist()):
            if pd.isna(v) or v==0: break
            if v<0 and buy==0: sell+=1
            else: break
        out[code]={"history":h,"連買":buy,"連賣":sell,"5日":s.tail(5).sum(min_count=1),"20日":s.tail(20).sum(min_count=1)}
    return out

def revenue_table():
    data=_get_json("https://openapi.twse.com.tw/v1/opendata/t187ap05_L")
    return pd.DataFrame(data) if isinstance(data,list) else pd.DataFrame()

def revenue_for(code,df):
    if df.empty: return {"營收":np.nan,"YoY":np.nan,"MoM":np.nan,"累計":np.nan}
    cc=[c for c in df.columns if "公司代號" in str(c) or "證券代號" in str(c)]
    if not cc: return {"營收":np.nan,"YoY":np.nan,"MoM":np.nan,"累計":np.nan}
    rows=df[df[cc[0]].astype(str).str.strip()==str(code)]
    if rows.empty: return {"營收":np.nan,"YoY":np.nan,"MoM":np.nan,"累計":np.nan}
    row=rows.iloc[-1]
    def pick(keys):
        for c in df.columns:
            if any(k in str(c) for k in keys): return _num(row[c])
        return np.nan
    return {"營收":pick(["當月營收","本月營收"]),"YoY":pick(["前期比較增減(%)","年增率","年增"]),"MoM":pick(["上月比較增減(%)","月增率","月增"]),"累計":pick(["累計營收"])}

def financials_yfinance(ticker):
    """Returns a normalized financial statement table. yfinance is used as a secondary public-data source."""
    try:
        import yfinance as yf
        t=yf.Ticker(ticker)
        annual=t.financials
        q=t.quarterly_financials
        return annual if isinstance(annual,pd.DataFrame) and not annual.empty else q
    except Exception:
        return pd.DataFrame()
