import numpy as np
import pandas as pd

def prepare(df):
    d = df.copy()
    d.columns = [str(c).title() for c in d.columns]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    for n in [5, 10, 20, 60]:
        d[f"MA{n}"] = d["Close"].rolling(n).mean()

    d["V5"] = d["Volume"].rolling(5).mean()
    d["V20"] = d["Volume"].rolling(20).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    mid = d["Close"].rolling(20).mean()
    sd = d["Close"].rolling(20).std()
    d["BBMid"] = mid
    d["BBUpper"] = mid + 2 * sd
    d["BBLower"] = mid - 2 * sd

    return d

def _candle(row):
    body = abs(row.Close - row.Open)
    rng = max(row.High - row.Low, 1e-9)
    upper = row.High - max(row.Open, row.Close)
    lower = min(row.Open, row.Close) - row.Low
    bull = row.Close > row.Open
    bear = row.Close < row.Open
    return body, rng, upper, lower, bull, bear

def detect_patterns(d):
    p = []
    if len(d) < 5:
        return p

    r = d.iloc[-1]
    body, rng, upper, lower, bull, bear = _candle(r)
    body_ratio = body / rng

    # 單根 K 線
    if body_ratio <= 0.12:
        p.append("十字星")
    elif body_ratio <= 0.25 and r.Close >= r.Open:
        p.append("小陽線")
    elif body_ratio <= 0.25 and r.Close < r.Open:
        p.append("小陰線")

    if lower >= max(body * 2, rng * 0.45) and upper <= rng * 0.2:
        p.append("錘頭／錘子線")
    if upper >= max(body * 2, rng * 0.45) and lower <= rng * 0.2:
        p.append("倒錘子線")

    if bull and body_ratio >= 0.75 and upper <= rng * 0.08 and lower <= rng * 0.08:
        p.append("光頭大陽線")
    if bear and body_ratio >= 0.75 and upper <= rng * 0.08 and lower <= rng * 0.08:
        p.append("光頭大陰線")

    if lower >= rng * 0.6 and body_ratio <= 0.35:
        p.append("下影線偏長")
    if upper >= rng * 0.6 and body_ratio <= 0.35:
        p.append("上影線偏長")

    # 兩根 / 三根
    if len(d) >= 3:
        a, b, c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
        abody, arng, aup, alow, abull, abear = _candle(a)
        bbody, brng, bup, blow, bbull, bbear = _candle(b)
        cbody, crng, cup, clow, cbull, cbear = _candle(c)

        if abull and bbull and cbull and a.Close < b.Close < c.Close:
            p.append("紅三兵")

        if abear and bbody <= abody * 0.5 and cbull and c.Close > (a.Open + a.Close) / 2:
            p.append("早晨之星")

        if abear and cbull and c.Close > a.Open and c.Open <= a.Close:
            p.append("反攻")

        if abear and cbull and c.Close > (a.Open + a.Close) / 2:
            p.append("曙光出現")

        if alow > abody * 1.5 and blow > bbody * 1.5:
            p.append("雙針探底")

        # 吞噬
        if abear and cbull and c.Open <= a.Close and c.Close >= a.Open:
            p.append("多頭吞噬")
        if abull and cbear and c.Open >= a.Close and c.Close <= a.Open:
            p.append("空頭吞噬")

    # 趨勢 / 均線
    if len(d) >= 60:
        last = d.iloc[-1]
        if last.MA5 > last.MA10 > last.MA20 > last.MA60:
            p.append("五線順上")
        if last.Close > last.MA5 > last.MA10 > last.MA20:
            p.append("均線多頭")
        if last.Close < last.MA5 < last.MA10 < last.MA20:
            p.append("均線空頭")

        if d.MA5.iloc[-2] <= d.MA20.iloc[-2] and d.MA5.iloc[-1] > d.MA20.iloc[-1]:
            p.append("均線黃金交叉")
        if d.MA5.iloc[-2] >= d.MA20.iloc[-2] and d.MA5.iloc[-1] < d.MA20.iloc[-1]:
            p.append("均線死亡交叉")

        if r.Close > r.MA5 and r.Close > r.MA10 and r.Close > r.MA20 and d.Close.iloc[-2] < d.MA20.iloc[-2]:
            p.append("一陽穿三線")

        if r.Close > r.BBUpper and r.Volume > r.V5:
            p.append("布林突破")

        # 回踩支撐
        recent20 = d["Low"].iloc[-21:-1]
        if len(recent20) >= 10:
            support = recent20.min()
            if r.Low <= support * 1.02 and r.Close > r.Open and r.Close > support:
                p.append("回踩支撐確認")

        # 突破前高
        if r.Close > d["High"].iloc[-21:-1].max():
            p.append("突破前高")

        # 量增
        if pd.notna(r.V5) and r.Volume > r.V5 * 1.5:
            p.append("量增")
        if pd.notna(r.V5) and r.Volume > r.V5 * 2:
            p.append("爆量")

    # 多日結構
    if len(d) >= 20:
        recent = d["Close"].iloc[-10:]
        older = d["Close"].iloc[-20:-10]
        if recent.mean() > older.mean() and recent.iloc[-1] > recent.iloc[0]:
            p.append("短線上升")
        if recent.iloc[-1] < recent.max() * 0.97 and recent.iloc[-1] > older.mean():
            p.append("高檔整理")

    # 去重
    return list(dict.fromkeys(p))

def bearish_patterns(d):
    p = []
    if len(d) < 5:
        return p
    r = d.iloc[-1]
    body, rng, upper, lower, bull, bear = _candle(r)

    if upper >= rng * 0.6 and body / rng <= 0.35:
        p.append("高檔長上影")
    if bear and body / rng >= 0.7:
        p.append("長黑K")

    if len(d) >= 3:
        a, b, c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
        if a.Close > a.Open and b.Close > b.Open and c.Close < c.Open:
            if c.Close < (a.Open + a.Close) / 2:
                p.append("烏雲蓋頂")
        if a.Close > a.Open and c.Close < c.Open and c.Open >= a.Close and c.Close <= a.Open:
            p.append("空頭吞噬")

    if len(d) >= 60:
        if r.Close < r.MA5 < r.MA10 < r.MA20:
            p.append("均線空頭")
        if d.MA5.iloc[-2] >= d.MA20.iloc[-2] and d.MA5.iloc[-1] < d.MA20.iloc[-1]:
            p.append("均線死亡交叉")

    return list(dict.fromkeys(p))

def score(d, bullish, bearish, institution_total=np.nan, institution_5d=np.nan,
          revenue_yoy=np.nan):
    last = d.iloc[-1]
    technical = 0
    reasons = []
    risks = []

    bullish_weights = {
        "紅三兵": 7, "早晨之星": 7, "多頭吞噬": 6,
        "錘頭／錘子線": 5, "金針探底": 5, "雙針探底": 5,
        "一陽穿三線": 7, "五線順上": 6, "均線多頭": 5,
        "均線黃金交叉": 6, "布林突破": 6, "突破前高": 8,
        "回踩支撐確認": 7, "量增": 4, "爆量": 3,
        "短線上升": 4, "曙光出現": 5, "反攻": 5,
        "光頭大陽線": 5,
    }

    bearish_weights = {
        "烏雲蓋頂": -8, "空頭吞噬": -8, "長黑K": -6,
        "均線空頭": -7, "均線死亡交叉": -7, "高檔長上影": -5,
    }

    for x in bullish:
        if x in bullish_weights:
            technical += bullish_weights[x]
            reasons.append(x)
    for x in bearish:
        if x in bearish_weights:
            technical += bearish_weights[x]
            risks.append(x)

    # 趨勢基礎分
    if pd.notna(last.MA20):
        if last.Close > last.MA20:
            technical += 5
        else:
            technical -= 5

    # RSI
    rsi = last.RSI
    if pd.notna(rsi):
        if 45 <= rsi <= 68:
            technical += 4
        elif rsi > 75:
            technical -= 5
            risks.append("RSI過熱")
        elif rsi < 30:
            risks.append("RSI超賣，需等待反轉確認")

    # 法人
    chip = 0
    if pd.notna(institution_total):
        chip += 8 if institution_total > 0 else -8 if institution_total < 0 else 0
    if pd.notna(institution_5d):
        chip += 6 if institution_5d > 0 else -6 if institution_5d < 0 else 0

    # 營收
    fundamental = 0
    if pd.notna(revenue_yoy):
        fundamental += 6 if revenue_yoy > 10 else 3 if revenue_yoy > 0 else -5

    total = int(np.clip(50 + technical + chip + fundamental, 0, 100))

    if total >= 78 and not (pd.notna(rsi) and rsi >= 75):
        signal = "偏多研究"
        action = "可進一步研究"
    elif total >= 60:
        signal = "再等等"
        action = "等待更多確認"
    else:
        signal = "偏弱研究"
        action = "暫不追價"

    if pd.notna(rsi) and rsi >= 75:
        signal = "再等等"
        action = "短線偏熱，等待拉回或突破確認"

    return {
        "分數": total,
        "訊號": signal,
        "動作": action,
        "理由": reasons[:6],
        "風險": risks[:6],
        "技術分": technical,
        "籌碼分": chip,
        "基本面分": fundamental,
    }
