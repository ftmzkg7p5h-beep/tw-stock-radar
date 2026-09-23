import re
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title='台股雷達 PRO', page_icon='📈', layout='wide')

BASE = Path(__file__).resolve().parent
CACHE_FILE = BASE / 'radar_cache.json'
UA = {'User-Agent': 'Mozilla/5.0'}

def http_json(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def num(x):
    if x is None or x == '' or x == '--':
        return np.nan
    try:
        return float(str(x).replace(',', '').replace(' ', '').replace('%', ''))
    except Exception:
        return np.nan

def lots(x):
    return np.nan if pd.isna(x) else x / 1000

def fmt(x, digits=1):
    if x is None or pd.isna(x):
        return '—'
    return f'{x:,.{digits}f}'

def roc_to_ad(s):
    s = str(s).strip()
    m = re.search(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', s)
    if not m:
        return s
    y, mo, d = map(int, m.groups())
    if y < 1911:
        y += 1911
    return f'{y:04d}/{mo:02d}/{d:02d}'

def latest_business_dates(days=12):
    today = datetime.now()
    return [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(days)]

@st.cache_data(ttl=3600, show_spinner=False)
def stock_universe():
    rows = []
    for endpoint in ['STOCK_DAY_ALL', 'ETF_DAY_ALL']:
        data = http_json(f'https://openapi.twse.com.tw/v1/exchangeReport/{endpoint}')
        if isinstance(data, list):
            rows.extend(data)

    out = []
    for r in rows:
        code = str(r.get('Code', r.get('股票代號', ''))).strip()
        name = str(r.get('Name') or r.get('股票名稱') or '').strip()
        if not re.fullmatch(r'\d{4,6}', code):
            continue
        close = num(r.get('ClosingPrice', r.get('收盤價')))
        change = num(r.get('Change', r.get('漲跌價差')))
        vol = num(r.get('TradeVolume', r.get('成交股數')))
        turnover = num(r.get('TradeValue', r.get('成交金額')))
        out.append({'code': code, 'name': name, 'close': close, 'change': change,
                    'volume': vol, 'turnover': turnover})

    fallback_names = {
        '1101':'台泥','1102':'亞泥','1216':'統一','1301':'台塑','1303':'南亞',
        '2002':'中鋼','2207':'和泰車','2303':'聯電','2308':'台達電','2313':'華通',
        '2317':'鴻海','2327':'國巨','2330':'台積電','2344':'華邦電','2357':'華碩',
        '2379':'瑞昱','2382':'廣達','2395':'研華','2408':'南亞科','2454':'聯發科',
        '2603':'長榮','2609':'陽明','2615':'萬海','2881':'富邦金','2882':'國泰金',
        '2883':'開發金','2884':'玉山金','2886':'兆豐金','2887':'台新新光金',
        '2890':'永豐金','2891':'中信金','2892':'第一金','3008':'大立光',
        '3034':'聯詠','3037':'欣興','3045':'台灣大','3231':'緯創','3443':'創意',
        '3711':'日月光投控','4904':'遠傳','4938':'和碩','5347':'世界',
        '5871':'中租-KY','5880':'合庫金','6239':'力成','6669':'緯穎',
        '8046':'南電','8454':'富邦媒','9904':'寶成'
    }

    if out:
        df = pd.DataFrame(out).drop_duplicates('code')
        df['name'] = df.apply(
            lambda r: r['name'] if r['name'] else fallback_names.get(r['code'], r['code']),
            axis=1
        )
        return df

    return pd.DataFrame([
        {'code': c, 'name': n, 'close': np.nan, 'change': np.nan,
         'volume': np.nan, 'turnover': np.nan}
        for c, n in fallback_names.items()
    ])

T86_ALIASES = {
    'foreign_net': ['外陸資買賣超股數(不含外資自營商)', '外陸資買賣超股數'],
    'trust_net': ['投信買賣超股數'],
    'dealer_net': ['自營商買賣超股數'],
    'all_net': ['三大法人買賣超股數'],
}

def _norm_field(x):
    return re.sub(r'[\s_（）()]', '', str(x)).replace('－','-').replace('—','-')

def _find_field(fields, aliases):
    norm = {_norm_field(f): f for f in fields}
    for alias in aliases:
        a = _norm_field(alias)
        for k, original in norm.items():
            if k == a:
                return fields.index(original)
    return None

def parse_t86_table(j):
    if not isinstance(j, dict) or not j.get('data') or not j.get('fields'):
        return {}
    fields = list(j.get('fields') or [])
    fi = {name: _find_field(fields, aliases) for name, aliases in T86_ALIASES.items()}
    if fi['foreign_net'] is None or fi['trust_net'] is None or fi['dealer_net'] is None:
        return {}

    code_idx = _find_field(fields, ['證券代號'])
    if code_idx is None:
        code_idx = 0

    all_idx = _find_field(fields, T86_ALIASES['all_net'])
    out = {}

    for row in j['data']:
        if not row:
            continue
        max_idx = max(code_idx, fi['foreign_net'], fi['trust_net'], fi['dealer_net'])
        if len(row) <= max_idx:
            continue

        code = str(row[code_idx]).strip()
        if not re.fullmatch(r'\d{4,6}', code):
            continue

        foreign = num(row[fi['foreign_net']])
        trust = num(row[fi['trust_net']])
        dealer = num(row[fi['dealer_net']])

        if pd.isna(foreign) and pd.isna(trust) and pd.isna(dealer):
            continue

        total = np.nansum([foreign, trust, dealer])

        if all_idx is not None and all_idx < len(row):
            all_net = num(row[all_idx])
            if not pd.isna(all_net) and abs(all_net - total) > 1:
                continue

        out[code] = {
            'foreign': foreign,
            'trust': trust,
            'dealer': dealer,
            'total': total
        }

    return out

@st.cache_data(ttl=900, show_spinner=False)
def institutional_history(code, lookback=25):
    frames = []
    for d in latest_business_dates(lookback + 10):
        j = http_json(
            'https://www.twse.com.tw/rwd/zh/fund/T86',
            params={'date': d, 'selectType': 'ALLBUT0999', 'response': 'json'},
            timeout=12
        )
        day = parse_t86_table(j)
        if code in day:
            x = day[code]
            frames.append({
                'date': d,
                'foreign': x['foreign'],
                'trust': x['trust'],
                'dealer': x['dealer'],
                'total': x['total']
            })
        if len(frames) >= lookback:
            break

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames).sort_values('date', ascending=False).reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def institutional_latest():
    for d in latest_business_dates():
        j = http_json(
            'https://www.twse.com.tw/rwd/zh/fund/T86',
            params={'date': d, 'selectType': 'ALLBUT0999', 'response': 'json'},
            timeout=12
        )
        day = parse_t86_table(j)
        if day:
            return day
    return {}

def institution_stats(hist):
    if hist is None or hist.empty:
        return {'streak': 0, 'five': np.nan, 'twenty': np.nan}

    vals = hist['total'].dropna().tolist()
    streak = 0

    if vals:
        sign = 1 if vals[0] > 0 else -1 if vals[0] < 0 else 0
        for v in vals:
            if sign == 0:
                break
            if (sign == 1 and v > 0) or (sign == -1 and v < 0):
                streak += 1
            else:
                break

    return {
        'streak': streak,
        'five': hist['total'].head(5).sum(),
        'twenty': hist['total'].head(20).sum()
    }

@st.cache_data(ttl=900, show_spinner=False)
def price_history(code, days=260):
    if yf is not None:
        try:
            df = yf.download(
                f'{code}.TW',
                period='2y',
                interval='1d',
                auto_adjust=False,
                progress=False,
                timeout=12
            )

            if isinstance(df, pd.DataFrame) and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if code + '.TW' in df.columns.get_level_values(-1):
                        try:
                            df = df.xs(code + '.TW', axis=1, level=-1)
                        except Exception:
                            pass
                    if isinstance(df.columns, pd.MultiIndex):
                        if code + '.TW' in df.columns.get_level_values(0):
                            try:
                                df = df.xs(code + '.TW', axis=1, level=0)
                            except Exception:
                                pass
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [str(c[0]) for c in df.columns]

                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                df = df[cols].copy()

                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

                df = df.dropna(subset=['Close']).copy()

                if len(df):
                    return df.tail(days)
        except Exception:
            pass

    frames = []
    today = datetime.now().replace(day=1)

    for m in range(0, 14):
        month = today - timedelta(days=31*m)
        d = month.strftime('%Y%m01')

        j = http_json(
            'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY',
            params={'date': d, 'stockNo': code, 'response': 'json'},
            timeout=15
        )

        if not isinstance(j, dict) or not j.get('data'):
            continue

        for row in j['data']:
            try:
                date = pd.to_datetime(roc_to_ad(row[0]), errors='coerce')
                if pd.isna(date):
                    continue

                # TWSE STOCK_DAY:
                # 0日期 1成交股數 2成交金額 3開盤 4最高 5最低 6收盤 7漲跌 8成交筆數
                frames.append({
                    'Date': date,
                    'Open': num(row[3]),
                    'High': num(row[4]),
                    'Low': num(row[5]),
                    'Close': num(row[6]),
                    'Volume': num(row[1])
                })
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    df = (
        pd.DataFrame(frames)
        .drop_duplicates('Date')
        .sort_values('Date')
        .set_index('Date')
    )

    return df.tail(days)

@st.cache_data(ttl=21600, show_spinner=False)
def revenue_history(code, months=60):
    frames = []
    today = datetime.now().replace(day=1)

    for m in range(0, min(months, 72)):
        dt = today - timedelta(days=31*m)

        j = http_json(
            'https://www.twse.com.tw/rwd/zh/afterTrading/FMSRFK',
            params={
                'date': dt.strftime('%Y%m01'),
                'stockNo': code,
                'response': 'json'
            },
            timeout=15
        )

        if not isinstance(j, dict) or not j.get('data'):
            continue

        for row in j['data']:
            if not row:
                continue

            vals = [str(x).strip() for x in row]
            date_idx = next(
                (i for i, x in enumerate(vals) if re.search(r'\d{3,4}/\d{1,2}', x)),
                0
            )

            date = pd.to_datetime(
                roc_to_ad(vals[date_idx]),
                errors='coerce'
            )

            if pd.isna(date):
                continue

            nums = [num(x) for x in vals]
            candidates = [x for x in nums if not pd.isna(x)]

            if not candidates:
                continue

            revenue = candidates[0]
            frames.append({'date': date, 'revenue': revenue})

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames).drop_duplicates('date').sort_values('date')
    return df.tail(months)

def revenue_summary(df):
    if df is None or df.empty:
        return {
            'yoy': np.nan, 'mom': np.nan,
            'high5': np.nan, 'low5': np.nan,
            'high12': np.nan, 'low12': np.nan,
            'signal': '—'
        }

    s = df['revenue'].astype(float)

    yoy = (
        (s.iloc[-1] / s.iloc[-13] - 1) * 100
        if len(s) >= 13 and s.iloc[-13]
        else np.nan
    )

    mom = (
        (s.iloc[-1] / s.iloc[-2] - 1) * 100
        if len(s) >= 2 and s.iloc[-2]
        else np.nan
    )

    last5 = s.tail(60)
    last12 = s.tail(12)
    cur = s.iloc[-1]

    sig = '—'

    if cur >= last5.max():
        sig = '🔥 歷史新高（近5年）'
    elif cur <= last5.min():
        sig = '❄️ 歷史新低（近5年）'
    elif cur >= last12.max():
        sig = '🔥 近12月新高'
    elif cur <= last12.min():
        sig = '❄️ 近12月新低'

    return {
        'yoy': yoy,
        'mom': mom,
        'high5': s.max(),
        'low5': s.min(),
        'high12': last12.max(),
        'low12': last12.min(),
        'signal': sig
    }

def technicals(df):
    if df is None or df.empty or len(df) < 30:
        return {}

    d = df.copy()
    c, h, l = d['Close'], d['High'], d['Low']

    d['MA5'] = c.rolling(5).mean()
    d['MA10'] = c.rolling(10).mean()
    d['MA20'] = c.rolling(20).mean()
    d['MA60'] = c.rolling(60).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100 - 100/(1+rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    d['MACD'] = ema12 - ema26
    d['Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()

    d['BBU'] = mid + 2*std
    d['BBL'] = mid - 2*std
    d['BBWidth'] = (d['BBU'] - d['BBL']) / mid.replace(0, np.nan)

    d['VolMA5'] = d['Volume'].rolling(5).mean()
    d['VolRatio'] = d['Volume'] / d['VolMA5'].replace(0, np.nan)

    return d

def patterns(d):
    if d is None or len(d) < 5:
        return []

    x = d.iloc[-5:].copy()
    out = []

    last = x.iloc[-1]
    prev = x.iloc[-2]

    body = abs(last.Close-last.Open)
    rng = max(last.High-last.Low, 1e-9)

    if (
        (min(last.Open,last.Close)-last.Low) > body*1.8
        and (last.High-max(last.Open,last.Close)) < rng*0.25
    ):
        out.append('金針探底')

    if all(x.iloc[i].Close > x.iloc[i].Open for i in range(2,5)):
        out.append('紅三兵')

    if (
        last.Close > last.MA20
        and prev.Close <= prev.MA20
        and last.Volume > x.Volume.tail(5).mean()*1.3
    ):
        out.append('突破均線')

    if len(d) >= 60 and last.MA5 > last.MA10 > last.MA20 > last.MA60:
        out.append('五線順上')

    if (
        last.Close > prev.High
        and last.Volume > x.Volume.tail(5).mean()*1.5
    ):
        out.append('突破缺口／強勢突破')

    if (
        last.Close > last.MA20
        and abs(last.Close-last.MA20)/last.MA20 < .025
        and last.Low <= last.MA20*1.005
    ):
        out.append('回踩支撐確認')

    if prev.Close < prev.Open and last.Close > last.Open and last.Close > prev.High:
        out.append('反攻')

    return list(dict.fromkeys(out))

def score_stock_v17(inst, hist, t, rev):
    if t is None or t.empty:
        return None

    if 'VolRatio' not in t.columns and 'Volume' in t.columns:
        t = t.copy()
        t['VolMA5'] = t['Volume'].rolling(5).mean()
        t['VolRatio'] = t['Volume'] / t['VolMA5'].replace(0, np.nan)

    a = t.iloc[-1]
    stt = institution_stats(hist)
    pats = patterns(t)

    ip = 12.5
    ir = []

    latest_total = inst.get('total', np.nan) if inst else np.nan

    if not pd.isna(latest_total):
        if latest_total > 0:
            ip += 6
            ir.append('法人當日買超')
        elif latest_total < 0:
            ip -= 6
            ir.append('法人當日賣超')

    if not pd.isna(stt['five']):
        ip += 4 if stt['five'] > 0 else -4 if stt['five'] < 0 else 0

    if not pd.isna(stt['twenty']):
        ip += 2.5 if stt['twenty'] > 0 else -2.5 if stt['twenty'] < 0 else 0

    ip = max(0, min(25, ip))

    tp = 10
    tr = []

    if a.Close > a.MA20:
        tp += 3
        tr.append('站上MA20')
    elif a.Close < a.MA20:
        tp -= 3

    if not pd.isna(a.MA60):
        tp += 2 if a.Close > a.MA60 else -2

    if a.MA5 > a.MA10 > a.MA20:
        tp += 2
        tr.append('均線多頭')

    if a.MACD > a.Signal:
        tp += 2
        tr.append('MACD多方')
    else:
        tp -= 2

    if 45 <= a.RSI <= 70:
        tp += 1
    elif a.RSI > 80:
        tp -= 2
    elif a.RSI < 35:
        tp -= 1

    tp = max(0, min(20, tp))

    kp = min(20, 10 + min(10, len(pats)*2)) if pats else 10
    kr = pats[:4]

    rp = 7.5
    rr = []

    yoy = rev.get('yoy', np.nan) if rev else np.nan

    if not pd.isna(yoy):
        if yoy >= 10:
            rp += 4
            rr.append('營收YoY成長')
        elif yoy <= -10:
            rp -= 4

    if rev and '🔥' in str(rev.get('signal','')):
        rp += 3
        rr.append('營收創高')
    elif rev and '❄️' in str(rev.get('signal','')):
        rp -= 3

    rp = max(0, min(15, rp))

    fp = 7.5

    vp = 2.5
    vr = []

    if not pd.isna(a.VolRatio):
        if a.VolRatio >= 1.5:
            vp += 2.5
            vr.append('成交量放大')
        elif a.VolRatio < 0.7:
            vp -= 1

    vp = max(0, min(5, vp))

    total = round(ip+tp+kp+rp+fp+vp)
    reasons = list(dict.fromkeys(ir+tr+kr+rr+vr))

    return {
        'score': int(total),
        'label': '🟢 偏多訊號' if total >= 75 else '🟡 注意',
        'inst_points': round(ip,1),
        'tech_points': round(tp,1),
        'k_points': round(kp,1),
        'revenue_points': round(rp,1),
        'fund_points': round(fp,1),
        'volume_points': round(vp,1),
        'patterns': '、'.join(pats) if pats else '—',
        'reason': '、'.join(reasons)
    }

def score_stock(code, name, market_row, inst, hist, price_df, rev_df):
    t = technicals(price_df)

    if t is None or t.empty:
        return None

    rev = revenue_summary(rev_df)
    latest = inst or {}

    s = score_stock_v17(latest, hist, t, rev)

    if s is None:
        return None

    stats = institution_stats(hist)
    total = latest.get('total', np.nan) if latest else np.nan
    last = t.iloc[-1]

    return {
        'code': code,
        'name': name,
        'score': s['score'],
        'label': s['label'],
        'close': float(last.Close),
        'change': market_row.get('change',np.nan),
        'foreign': latest.get('foreign',np.nan),
        'trust': latest.get('trust',np.nan),
        'dealer': latest.get('dealer',np.nan),
        'inst_total': total,
        'streak': stats['streak'],
        'five': stats['five'],
        'twenty': stats['twenty'],
        'rsi': last.RSI,
        'patterns': s['patterns'],
        'revenue_signal': rev['signal'],
        'revenue_yoy': rev['yoy'],
        'reason': s['reason'],
        'inst_points': s['inst_points'],
        'tech_points': s['tech_points'],
        'k_points': s['k_points'],
        'revenue_points': s['revenue_points'],
        'fund_points': s['fund_points'],
        'volume_points': s['volume_points']
    }

def load_cache():
    try:
        return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def live_radar():
    universe = stock_universe()

    if universe.empty:
        return pd.DataFrame()

    # 先掃成交金額最高 20 檔，避免首頁因大量 API 請求 timeout。
    u = universe.sort_values('turnover', ascending=False).head(20).copy()

    inst = institutional_latest()
    results = []

    for _, r in u.iterrows():
        code = r.code

        try:
            p = price_history(code, 180)

            if p.empty:
                continue

            h = institutional_history(code, 20)
            rv = revenue_history(code, 24)

            z = score_stock(
                code,
                r['name'],
                r.to_dict(),
                inst.get(code, {}),
                h,
                p,
                rv
            )

            if z:
                results.append(z)

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values('score', ascending=False)

def market_only_radar():
    u = stock_universe()

    if u.empty:
        return pd.DataFrame()

    x = u.copy()
    x = x[(x["close"].notna()) & (x["turnover"].notna())]

    if x.empty:
        return pd.DataFrame()

    med_turn = x["turnover"].replace(0, np.nan).median()

    if pd.isna(med_turn) or med_turn <= 0:
        med_turn = 1.0

    x["score"] = 50.0
    x["score"] += x["change"].fillna(0).clip(-10, 10) * 2.0
    x["score"] += np.log1p(
        x["turnover"].clip(lower=0) / med_turn
    ).clip(-3, 3) * 4.0

    x["score"] = x["score"].clip(0, 100).round().astype(int)
    x["label"] = np.where(
        x["score"] >= 75,
        "🟢 偏多訊號",
        "🟡 注意"
    )

    x["inst_total"] = np.nan
    x["five"] = np.nan
    x["twenty"] = np.nan
    x["rsi"] = np.nan
    x["revenue_yoy"] = np.nan
    x["revenue_signal"] = "—"
    x["patterns"] = "—"

    x["reason"] = np.where(
        x["change"].fillna(0) > 0,
        "當日股價上漲＋成交金額",
        "成交金額活躍"
    )

    return x[
        [
            "code","name","score","label","close","change",
            "inst_total","five","twenty","rsi","patterns",
            "revenue_signal","revenue_yoy","reason"
        ]
    ].sort_values("score", ascending=False).head(30).reset_index(drop=True)

def cached_or_live_radar():
    c = load_cache()

    if isinstance(c, dict) and c.get('results'):
        return pd.DataFrame(c['results']), c.get('generated_at', '')

    live = live_radar()

    if not live.empty:
        return live, datetime.now().strftime('%Y-%m-%d %H:%M')

    fallback = market_only_radar()

    return (
        fallback,
        datetime.now().strftime('%Y-%m-%d %H:%M')
        if not fallback.empty else ''
    )

def cached_or_live_radar():
    c = load_cache()
    if isinstance(c, dict) and c.get('results'):
        return pd.DataFrame(c['results']), c.get('generated_at', '')

    # 重要：首頁絕對不要在啟動時 live_radar()。
    # 沒有 GitHub Actions 快取時，改用輕量市場資料，避免 App 卡在 Loading。
    fallback = market_only_radar()
    return (fallback, datetime.now().strftime('%Y-%m-%d %H:%M') if not fallback.empty else '')

# -----------------------------
# UI
# -----------------------------
st.title('📈 台股雷達 PRO')
st.caption('法人籌碼 × 營收 × 技術分析 × K線型態｜自動選股 + 個股查詢')

# 側邊選單：保留上一版的操作方式
with st.sidebar:
    st.markdown('## 📈 台股雷達 PRO')
    page = st.radio(
        '功能選單',
        ['🔥 今日雷達', '🔎 個股分析', '⚙️ 系統狀態'],
        index=0
    )
    st.divider()
    st.caption('資料來源：TWSE 官方資料 + yfinance')


def render_radar():
    st.subheader('🔥 今日自動雷達')
    st.caption('首頁只讀快取或輕量市場資料，不會啟動時掃描 20 檔股票。')

    cache_df, cache_time = cached_or_live_radar()

    if cache_df.empty:
        st.warning('目前沒有可用的市場資料，請稍後重新整理。')
        return

    if cache_time:
        st.caption(f'資料時間：{cache_time}')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('偏多訊號', int((cache_df.score >= 75).sum()))
    c2.metric('觀察名單', int(((cache_df.score >= 60) & (cache_df.score < 75)).sum()))
    c3.metric('法人5日偏多', int((cache_df.five.fillna(0) > 0).sum()))
    c4.metric('營收創高', int(cache_df.revenue_signal.astype(str).str.contains('新高').sum()))

    display = cache_df.head(30).copy()
    display['法人合計'] = display.inst_total.map(lambda x: f'{lots(x):,.1f} 張' if not pd.isna(x) else '—')
    display['法人5日'] = display.five.map(lambda x: f'{lots(x):,.1f} 張' if not pd.isna(x) else '—')
    display['RSI'] = display.rsi.map(lambda x: f'{x:.1f}' if not pd.isna(x) else '—')
    display['營收YoY'] = display.revenue_yoy.map(lambda x: f'{x:.1f}%' if not pd.isna(x) else '—')

    st.dataframe(
        display[['code','name','label','score','close','法人合計','法人5日','RSI','營收YoY','revenue_signal','patterns','reason']],
        width='stretch', hide_index=True,
        column_config={
            'code':'代號','name':'名稱','label':'雷達訊號','score':'總分',
            'close':'收盤','revenue_signal':'營收訊號','patterns':'K線型態','reason':'主要原因'
        }
    )

    st.info('💡 自動雷達如果有 GitHub Actions 產生的 radar_cache.json，首頁會直接顯示完整法人＋技術＋營收結果；沒有快取時則先顯示輕量市場雷達，避免首頁卡住。')


def render_stock_analysis():
    st.subheader('🔎 個股分析')
    st.caption('輸入代號或名稱後按「開始分析」，資料才會開始抓取。')

    query = st.text_area(
        '股票代號／名稱',
        placeholder='例如：2330, 2317, 2454\n或：台積電',
        height=90,
        key='stock_query'
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.button('🔍 開始分析', type='primary', width='stretch')

    if not run:
        st.info('請輸入股票代號，例如 2330，然後按「開始分析」。')
        return

    if not query.strip():
        st.warning('請先輸入股票代號或名稱。')
        return

    with st.status('正在準備分析資料…', expanded=True) as status:
        st.write('① 取得股票清單')
        universe = stock_universe()

        tokens = [x.strip() for x in re.split(r'[\s,，、;；|]+', query.strip()) if x.strip()]
        seen = set()
        tokens = [x for x in tokens if not (x in seen or seen.add(x))]

        selected = []
        for token in tokens:
            m = universe[universe['code'].eq(token)]
            if m.empty:
                m = universe[universe['name'].str.contains(token, case=False, na=False)]
            if not m.empty:
                selected.append(m.iloc[0].to_dict())

        if not selected:
            status.update(label='找不到股票', state='error')
            st.warning('找不到符合的股票代號／名稱。')
            return

        st.write(f'② 找到 {len(selected)} 檔：' + '、'.join(x['code'] for x in selected))
        st.write('③ 取得法人資料')
        inst_all = institutional_latest()
        st.write('④ 開始取得股價、法人歷史與營收')

        results=[]
        progress=st.progress(0)
        for i,r in enumerate(selected):
            code,name=r['code'],r['name']
            p=price_history(code,260)
            if p.empty:
                st.warning(f'{code} {name}：價格資料暫時無法取得。')
                progress.progress((i+1)/len(selected))
                continue
            h=institutional_history(code,20)
            rv=revenue_history(code,60)
            t=technicals(p)
            inst=inst_all.get(code,{})
            stats=institution_stats(h)
            rev=revenue_summary(rv)
            pats=patterns(t)
            score_obj=score_stock(code,name,r,inst,h,p,rv)
            results.append((r,p,h,rv,t,inst,stats,rev,pats,score_obj))
            progress.progress((i+1)/len(selected))
        progress.empty()
        status.update(label='分析完成', state='complete')

    if not results:
        st.error('沒有取得足夠的價格資料。')
        return

    for r,p,h,rv,t,inst,stats,rev,pats,score_obj in results:
        code,name=r['code'],r['name']
        last=t.iloc[-1]
        score=score_obj['score'] if score_obj else 50
        label=score_obj['label'] if score_obj else '🟡 注意'
        total=inst.get('foreign',np.nan)+inst.get('trust',np.nan)+inst.get('dealer',np.nan) if inst else np.nan

        st.divider()
        st.markdown(f'## 📌 {code} {name or "（名稱未取得）"}')
        st.caption(f'{label}｜雷達分數 {score}')
        if score_obj:
            st.write(
                f'籌碼 {score_obj.get("inst_points",0)}/25 ｜ 技術 {score_obj.get("tech_points",0)}/20 ｜ '
                f'K線 {score_obj.get("k_points",0)}/20 ｜ 營收 {score_obj.get("revenue_points",0)}/15 ｜ '
                f'財報 {score_obj.get("fund_points",0)}/15 ｜ 量價 {score_obj.get("volume_points",0)}/5'
            )
            st.caption('主要訊號：' + (score_obj.get('reason') or '—'))

        a,b,c,d=st.columns(4)
        a.metric('收盤',fmt(last.Close,2))
        b.metric('雷達分數',score)
        c.metric('法人連買/賣',f'{stats["streak"]} 天')
        d.metric('RSI',fmt(last.RSI,1))

        tabs=st.tabs(['🏦 法人','📈 技術 / K線','💰 營收','📊 財報'])
        with tabs[0]:
            st.write(f'外資：{fmt(lots(inst.get("foreign",np.nan)),1)} 張')
            st.write(f'投信：{fmt(lots(inst.get("trust",np.nan)),1)} 張')
            st.write(f'自營商：{fmt(lots(inst.get("dealer",np.nan)),1)} 張')
            st.write(f'法人合計：{fmt(lots(total),1)} 張')
            st.write(f'法人連買／賣：{stats["streak"]} 天')
            st.write(f'法人5日：{fmt(lots(stats["five"]),1)} 張')
            st.write(f'法人20日：{fmt(lots(stats["twenty"]),1)} 張')
            if not h.empty:
                hh=h.copy()
                for col in ['foreign','trust','dealer','total']:
                    hh[col]=hh[col].map(lots)
                st.dataframe(hh.rename(columns={'date':'日期','foreign':'外資(張)','trust':'投信(張)','dealer':'自營商(張)','total':'合計(張)'}),width='stretch',hide_index=True)
        with tabs[1]:
            cols=st.columns(4)
            cols[0].metric('MA5',fmt(last.MA5,2)); cols[1].metric('MA20',fmt(last.MA20,2)); cols[2].metric('MA60',fmt(last.MA60,2)); cols[3].metric('MACD',fmt(last.MACD,2))
            st.write(f'K線型態：**{("、".join(pats) if pats else "—")}**')
            st.dataframe(t.tail(40)[['Open','High','Low','Close','Volume','MA5','MA10','MA20','MA60','RSI','MACD','Signal']],width='stretch')
        with tabs[2]:
            st.write(f'本月營收：{fmt(rv.revenue.iloc[-1],0) if not rv.empty else "—"}')
            st.write(f'YoY：{fmt(rev["yoy"],1)}%｜MoM：{fmt(rev["mom"],1)}%')
            st.write(f'最高：{fmt(rev["high5"],0)}｜最低：{fmt(rev["low5"],0)}')
            st.write(f'近12月最高：{fmt(rev["high12"],0)}｜近12月最低：{fmt(rev["low12"],0)}')
            st.write(f'訊號：**{rev["signal"]}**')
            if not rv.empty: st.line_chart(rv.set_index('date')['revenue'])
        with tabs[3]:
            st.info('財報資料目前維持「未取得＝中性」原則；未取得經驗證的官方欄位，不參與加分。')


def render_status():
    st.subheader('⚙️ 系統狀態')
    st.write(f'Python / Streamlit：正常啟動')
    st.write(f'yfinance：{"已載入" if yf is not None else "未載入，將使用 TWSE 備援"}')
    cache=load_cache()
    if isinstance(cache,dict) and cache.get('results'):
        st.success(f'已找到 radar_cache.json：{len(cache["results"])} 筆雷達資料')
        st.caption(f'快取時間：{cache.get("generated_at", "未知")}')
    else:
        st.warning('目前沒有 radar_cache.json；首頁會使用輕量市場資料，不會因此卡住。')


if page == '🔥 今日雷達':
    render_radar()
elif page == '🔎 個股分析':
    render_stock_analysis()
else:
    render_status()

st.divider()
st.caption('資料來源以 TWSE 官方資料為主；yfinance 作為價格歷史備援。雷達屬條件篩選與研究工具，不代表個別投資建議。')
