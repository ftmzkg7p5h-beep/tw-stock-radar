"""Daily Taiwan-stock radar builder for GitHub Actions.
Creates radar_cache.json used by Streamlit.
"""
import json, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'radar_cache.json'
UA={'User-Agent':'Mozilla/5.0'}


def http_json(url, params=None, timeout=20):
    try:
        r=requests.get(url,params=params,headers=UA,timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception:
        return None


def num(x):
    if x is None or x in ('','--'): return np.nan
    try: return float(str(x).replace(',','').replace('%',''))
    except: return np.nan


def latest_dates(n=12):
    now=datetime.now()
    return [(now-timedelta(days=i)).strftime('%Y%m%d') for i in range(n)]


def universe():
    rows=[]
    for ep in ('STOCK_DAY_ALL','ETF_DAY_ALL'):
        j=http_json(f'https://openapi.twse.com.tw/v1/exchangeReport/{ep}')
        if isinstance(j,list): rows.extend(j)
    out=[]
    for r in rows:
        code=str(r.get('Code',r.get('股票代號',''))).strip()
        if not re.fullmatch(r'\d{4,6}',code): continue
        close=num(r.get('ClosingPrice',r.get('收盤價')))
        if np.isnan(close): continue
        out.append({'code':code,'name':str(r.get('Name',r.get('股票名稱',''))),'close':close,
                    'change':num(r.get('Change',r.get('漲跌價差'))),
                    'turnover':num(r.get('TradeValue',r.get('成交金額')))})
    return pd.DataFrame(out).drop_duplicates('code')


def t86_all():
    result={}
    for d in latest_dates():
        j=http_json('https://www.twse.com.tw/rwd/zh/fund/T86',
                    {'date':d,'selectType':'ALLBUT0999','response':'json'})
        if not isinstance(j,dict) or not j.get('data'): continue
        day={}
        for row in j['data']:
            try:
                code=str(row[0]).strip()
                if re.fullmatch(r'\d{4,6}',code):
                    day[code]={'foreign':num(row[2]),'trust':num(row[9]),'dealer':num(row[12])}
            except: pass
        if day:
            result[d]=day
    return result


def institution_tables(hist, codes):
    out={}
    days=sorted(hist.keys(),reverse=True)
    for code in codes:
        rows=[]
        for d in days[:20]:
            x=hist[d].get(code)
            if x:
                x=x.copy(); x['total']=sum(v for v in x.values() if not np.isnan(v)); x['date']=d; rows.append(x)
        if not rows: continue
        total=[r['total'] for r in rows]
        sign=1 if total[0]>0 else -1 if total[0]<0 else 0
        streak=0
        for v in total:
            if sign==1 and v>0: streak+=1
            elif sign==-1 and v<0: streak+=1
            else: break
        out[code]={'latest':rows[0],'streak':streak,'five':float(np.nansum(total[:5])),'twenty':float(np.nansum(total[:20]))}
    return out


def technical(df):
    c=df['Close']; d=df.copy()
    d['MA5']=c.rolling(5).mean(); d['MA10']=c.rolling(10).mean(); d['MA20']=c.rolling(20).mean(); d['MA60']=c.rolling(60).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=-delta.clip(upper=0).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); d['RSI']=100-100/(1+rs)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); d['MACD']=e12-e26; d['Signal']=d['MACD'].ewm(span=9,adjust=False).mean()
    return d


def pattern(d):
    if len(d)<5:return []
    x=d.iloc[-5:]; a=x.iloc[-1]; p=x.iloc[-2]; out=[]
    body=abs(a.Close-a.Open); rng=max(a.High-a.Low,1e-9)
    if min(a.Open,a.Close)-a.Low > body*1.8 and a.High-max(a.Open,a.Close)<rng*.25: out.append('金針探底')
    if all(x.iloc[i].Close>x.iloc[i].Open for i in range(2,5)): out.append('紅三兵')
    if a.Close>a.MA20 and p.Close<=p.MA20 and a.Volume>x.Volume.tail(5).mean()*1.3: out.append('突破均線')
    if not np.isnan(a.MA60) and a.MA5>a.MA10>a.MA20>a.MA60: out.append('五線順上')
    if a.Close>p.High and a.Volume>x.Volume.tail(5).mean()*1.5: out.append('強勢突破')
    if a.Close>a.MA20 and abs(a.Close-a.MA20)/a.MA20<.025 and a.Low<=a.MA20*1.005: out.append('回踩支撐確認')
    if p.Close<p.Open and a.Close>a.Open and a.Close>p.High: out.append('反攻')
    return list(dict.fromkeys(out))


def score(code,name,row,inst,p):
    if p.empty or len(p)<60:return None
    d=technical(p); a=d.iloc[-1]; it=inst.get(code,{})
    total=it.get('latest',{}).get('total',np.nan)
    five=it.get('five',np.nan); twenty=it.get('twenty',np.nan)
    s=50; reasons=[]
    if not np.isnan(total):
        if total>0:s+=10;reasons.append('法人買超')
        elif total<0:s-=8;reasons.append('法人賣超')
    if not np.isnan(five): s+=7 if five>0 else -5
    if not np.isnan(twenty): s+=5 if twenty>0 else -4
    if a.Close>a.MA20:s+=7;reasons.append('站上MA20')
    else:s-=5
    if not np.isnan(a.MA60): s+=5 if a.Close>a.MA60 else -3
    if a.MA5>a.MA20:s+=4
    if a.MACD>a.Signal:s+=5;reasons.append('MACD多方')
    else:s-=3
    if not np.isnan(a.RSI):
        if 45<=a.RSI<=70:s+=4
        elif a.RSI>80:s-=4
    pats=pattern(d)
    s+=min(8,2*len(pats)); reasons.extend(pats[:2])
    s=int(max(0,min(100,round(s))))
    return {'code':code,'name':name,'score':s,'label':'🟢 偏多訊號' if s>=75 else '🟡 注意',
            'close':float(a.Close),'change':float(row.get('change',np.nan)) if not np.isnan(row.get('change',np.nan)) else None,
            'foreign':float(it.get('latest',{}).get('foreign',np.nan)) if not np.isnan(it.get('latest',{}).get('foreign',np.nan)) else None,
            'trust':float(it.get('latest',{}).get('trust',np.nan)) if not np.isnan(it.get('latest',{}).get('trust',np.nan)) else None,
            'dealer':float(it.get('latest',{}).get('dealer',np.nan)) if not np.isnan(it.get('latest',{}).get('dealer',np.nan)) else None,
            'inst_total':float(total) if not np.isnan(total) else None,
            'streak':int(it.get('streak',0)),'five':float(five) if not np.isnan(five) else None,
            'twenty':float(twenty) if not np.isnan(twenty) else None,'rsi':float(a.RSI) if not np.isnan(a.RSI) else None,
            'patterns':'、'.join(pats) if pats else '—','revenue_signal':'—','revenue_yoy':None,
            'reason':'、'.join(dict.fromkeys(reasons))}


def batch_prices(codes):
    out={}
    if not codes:return out
    for start in range(0,len(codes),50):
        batch=codes[start:start+50]
        try:
            df=yf.download([f'{c}.TW' for c in batch],period='6mo',interval='1d',auto_adjust=False,progress=False,group_by='ticker',threads=True)
            if isinstance(df,pd.DataFrame) and not df.empty:
                if isinstance(df.columns,pd.MultiIndex):
                    for code in batch:
                        key=f'{code}.TW'
                        if key not in df.columns.get_level_values(0):continue
                        x=df[key].copy()
                        if 'Close' in x and x['Close'].notna().sum()>=60:out[code]=x.dropna(subset=['Close']).tail(180)
                else:
                    code=batch[0]; x=df.dropna(subset=['Close']);
                    if len(x)>=60:out[code]=x.tail(180)
        except Exception as e:
            print('batch error',start,e)
    return out


def main():
    u=universe()
    if u.empty: raise SystemExit('TWSE stock universe unavailable')
    # Focus daily scan on liquid names to keep GitHub Actions reliable.
    u=u.sort_values('turnover',ascending=False).head(220).reset_index(drop=True)
    codes=u.code.tolist()
    ih=t86_all(); inst=institution_tables(ih,codes)
    prices=batch_prices(codes)
    results=[]
    for _,r in u.iterrows():
        z=score(r.code,r['name'],r,inst,prices.get(r.code,pd.DataFrame()))
        if z: results.append(z)
    results=sorted(results,key=lambda x:x['score'],reverse=True)[:60]
    now=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S %z')
    payload={'generated_at':now,'market':'TWSE','scan_universe':len(u),'results':results}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'generated {len(results)} radar results at {now}')

if __name__=='__main__': main()
