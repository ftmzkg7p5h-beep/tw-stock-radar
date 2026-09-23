import numpy as np
import pandas as pd

PATTERN_DESCRIPTIONS = {
    "十字星": "實體很小，多空暫時僵持，單獨出現不代表一定反轉。",
    "小陽線": "小幅收紅，買方略占優勢。",
    "小陰線": "小幅收黑，賣方略占優勢。",
    "錘頭／錘子線": "下影較長，若出現在下跌後並獲量價確認，可能代表支撐。",
    "倒錘子線": "上影較長，若出現在下跌後需等待隔日確認。",
    "光頭大陽線": "實體長且上下影短，單日買盤強。",
    "光頭大陰線": "實體長且上下影短，單日賣壓強。",
    "紅三兵": "連續收紅且逐步墊高，屬多方延續型態。",
    "早晨之星": "下跌後的三K反轉型態，仍需量價確認。",
    "反攻": "空方後出現強勢紅K收復部分跌幅。",
    "曙光出現": "下跌後紅K深入前一根黑K實體，偏反轉訊號。",
    "雙針探底": "連續出現明顯下影，代表低檔承接。",
    "多頭吞噬": "紅K實體覆蓋前一根黑K，偏多反轉訊號。",
    "空頭吞噬": "黑K實體覆蓋前一根紅K，偏空反轉訊號。",
    "五線順上": "短中期均線呈多頭排列，趨勢偏強。",
    "均線多頭": "收盤站在主要短中期均線之上。",
    "均線空頭": "收盤跌破主要短中期均線。",
    "均線黃金交叉": "MA5向上穿越MA20，屬趨勢轉強的觀察訊號。",
    "均線死亡交叉": "MA5向下跌破MA20，屬趨勢轉弱的觀察訊號。",
    "一陽穿三線": "單日紅K重新站回MA5/10/20，偏多確認訊號。",
    "布林突破": "收盤突破布林上軌且量能放大。",
    "突破前高": "收盤突破近20交易日高點。",
    "回踩支撐確認": "回測近期支撐後收紅，代表支撐暫時有效。",
    "量增": "今日成交量高於5日均量。",
    "爆量": "今日成交量大幅高於5日均量，需搭配K線判斷。",
    "短線上升": "短期平均價格高於前一段，且近期價格墊高。",
    "高檔整理": "高檔震盪但尚未明顯跌破趨勢。",
    "高檔長上影": "高檔出現長上影，代表上方賣壓。",
    "長黑K": "單日黑K實體偏長，短線賣壓明顯。",
    "烏雲蓋頂": "高檔出現黑K深入前一紅K，偏空反轉警訊。",
}


def prepare(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [str(c).title() for c in d.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    for n in [5, 10, 20, 60, 120, 240]:
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
    d["MACDHist"] = d["MACD"] - d["MACDSignal"]
    mid = d["Close"].rolling(20).mean()
    sd = d["Close"].rolling(20).std()
    d["BBMid"] = mid
    d["BBUpper"] = mid + 2 * sd
    d["BBLower"] = mid - 2 * sd
    d["Return5"] = d["Close"].pct_change(5) * 100
    d["Return20"] = d["Close"].pct_change(20) * 100
    tr = pd.concat([d["High"]-d["Low"], (d["High"]-d["Close"].shift()).abs(), (d["Low"]-d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    return d


def _candle(row):
    body = abs(row.Close - row.Open)
    rng = max(row.High - row.Low, 1e-9)
    upper = row.High - max(row.Open, row.Close)
    lower = min(row.Open, row.Close) - row.Low
    return body, rng, upper, lower, row.Close > row.Open, row.Close < row.Open


def detect_patterns(d):
    p = []
    if len(d) < 5:
        return p
    r = d.iloc[-1]
    body, rng, upper, lower, bull, bear = _candle(r)
    ratio = body / rng
    if ratio <= 0.12: p.append("十字星")
    elif ratio <= 0.25 and bull: p.append("小陽線")
    elif ratio <= 0.25 and bear: p.append("小陰線")
    if lower >= max(body * 2, rng * .45) and upper <= rng * .2: p.append("錘頭／錘子線")
    if upper >= max(body * 2, rng * .45) and lower <= rng * .2: p.append("倒錘子線")
    if bull and ratio >= .75 and upper <= rng*.08 and lower <= rng*.08: p.append("光頭大陽線")
    if bear and ratio >= .75 and upper <= rng*.08 and lower <= rng*.08: p.append("光頭大陰線")
    if len(d) >= 3:
        a,b,c=d.iloc[-3],d.iloc[-2],d.iloc[-1]
        ab,ar,au,al,abull,abear=_candle(a); bb,br,bu,bl,bbull,bbear=_candle(b); cb,cr,cu,cl,cbull,cbear=_candle(c)
        if abull and bbull and cbull and a.Close < b.Close < c.Close: p.append("紅三兵")
        if abear and bb <= ab*.6 and cbull and c.Close > (a.Open+a.Close)/2: p.append("早晨之星")
        if abear and cbull and c.Close > a.Open and c.Open <= a.Close: p.append("反攻")
        if abear and cbull and c.Close > (a.Open+a.Close)/2: p.append("曙光出現")
        if al > ab*1.5 and bl > bb*1.5: p.append("雙針探底")
        if abear and cbull and c.Open <= a.Close and c.Close >= a.Open: p.append("多頭吞噬")
        if abull and cbear and c.Open >= a.Close and c.Close <= a.Open: p.append("空頭吞噬")
    if len(d) >= 60:
        if r.MA5 > r.MA10 > r.MA20 > r.MA60: p.append("五線順上")
        if r.Close > r.MA5 > r.MA10 > r.MA20: p.append("均線多頭")
        if r.Close < r.MA5 < r.MA10 < r.MA20: p.append("均線空頭")
        if d.MA5.iloc[-2] <= d.MA20.iloc[-2] and d.MA5.iloc[-1] > d.MA20.iloc[-1]: p.append("均線黃金交叉")
        if d.MA5.iloc[-2] >= d.MA20.iloc[-2] and d.MA5.iloc[-1] < d.MA20.iloc[-1]: p.append("均線死亡交叉")
        if r.Close > r.MA5 and r.Close > r.MA10 and r.Close > r.MA20 and d.Close.iloc[-2] < d.MA20.iloc[-2]: p.append("一陽穿三線")
        if r.Close > r.BBUpper and r.Volume > r.V5: p.append("布林突破")
        prev_high=d.High.iloc[-21:-1].max()
        if r.Close > prev_high: p.append("突破前高")
        support=d.Low.iloc[-21:-1].min()
        if r.Low <= support*1.02 and r.Close > r.Open and r.Close > support: p.append("回踩支撐確認")
        if pd.notna(r.V5) and r.Volume > r.V5*1.5: p.append("量增")
        if pd.notna(r.V5) and r.Volume > r.V5*2: p.append("爆量")
    if len(d) >= 20:
        recent=d.Close.iloc[-10:]; older=d.Close.iloc[-20:-10]
        if recent.mean()>older.mean() and recent.iloc[-1]>recent.iloc[0]: p.append("短線上升")
        if recent.iloc[-1] < recent.max()*.97 and recent.iloc[-1] > older.mean(): p.append("高檔整理")
    return list(dict.fromkeys(p))


def bearish_patterns(d):
    p=[]
    if len(d)<5: return p
    r=d.iloc[-1]; body,rng,upper,lower,bull,bear=_candle(r)
    if upper>=rng*.6 and body/rng<=.35: p.append("高檔長上影")
    if bear and body/rng>=.7: p.append("長黑K")
    if len(d)>=3:
        a,b,c=d.iloc[-3],d.iloc[-2],d.iloc[-1]
        if a.Close>a.Open and b.Close>b.Open and c.Close<c.Open and c.Close < (a.Open+a.Close)/2: p.append("烏雲蓋頂")
        if a.Close>a.Open and c.Close<c.Open and c.Open>=a.Close and c.Close<=a.Open: p.append("空頭吞噬")
    if len(d)>=60:
        if r.Close<r.MA5<r.MA10<r.MA20: p.append("均線空頭")
        if d.MA5.iloc[-2]>=d.MA20.iloc[-2] and d.MA5.iloc[-1]<d.MA20.iloc[-1]: p.append("均線死亡交叉")
    return list(dict.fromkeys(p))


def score(d,bullish,bearish,institution_total=np.nan,institution_5d=np.nan,institution_20d=np.nan,revenue_yoy=np.nan):
    last=d.iloc[-1]; technical=0; reasons=[]; risks=[]
    bw={"紅三兵":7,"早晨之星":7,"多頭吞噬":7,"錘頭／錘子線":5,"雙針探底":5,"一陽穿三線":7,"五線順上":7,"均線多頭":5,"均線黃金交叉":6,"布林突破":7,"突破前高":8,"回踩支撐確認":7,"量增":4,"爆量":2,"短線上升":4,"曙光出現":5,"反攻":5,"光頭大陽線":5}
    sw={"烏雲蓋頂":-8,"空頭吞噬":-9,"長黑K":-6,"均線空頭":-8,"均線死亡交叉":-8,"高檔長上影":-5}
    for x in bullish:
        if x in bw: technical += bw[x]; reasons.append(x)
    for x in bearish:
        if x in sw: technical += sw[x]; risks.append(x)
    if pd.notna(last.MA20): technical += 5 if last.Close>last.MA20 else -5
    rsi=last.RSI
    if pd.notna(rsi):
        if 45<=rsi<=68: technical+=4
        elif 68<rsi<75: technical+=1
        elif rsi>=75: technical-=5; risks.append("RSI過熱")
        elif rsi<30: risks.append("RSI超賣，等待反轉確認")
    chip=0
    for val,weight in [(institution_total,8),(institution_5d,6),(institution_20d,4)]:
        if pd.notna(val): chip += weight if val>0 else -weight if val<0 else 0
    fundamental=0
    if pd.notna(revenue_yoy): fundamental += 6 if revenue_yoy>10 else 3 if revenue_yoy>0 else -5
    total=int(np.clip(50+technical+chip+fundamental,0,100))
    close=float(last.Close)
    ma20=float(last.MA20) if pd.notna(last.MA20) else close
    atr=float(last.ATR14) if pd.notna(last.ATR14) and float(last.ATR14)>0 else max(close*0.02, 0.01)
    breakout=any(x in bullish for x in ["突破前高","布林突破"])
    overheat=pd.notna(rsi) and rsi>=75
    if total>=85 and not overheat:
        signal="🟢 買進條件成立"; action="趨勢、籌碼與基本面同時偏多；以風險控管方式執行"
    elif total>=75 and breakout and not overheat:
        signal="🔵 突破確認"; action="突破型態成立，但應確認量能與停損位置"
    elif total>=68:
        signal="🟡 等待"; action="條件尚未完整，等待拉回支撐或突破確認"
    else:
        signal="🔴 不買"; action="目前訊號偏弱，不以單一指標逆勢進場"
    if overheat:
        signal="🟠 不追高"; action="短線過熱，等待拉回或重新形成低風險進場點"
    elif total<55:
        signal="🔴 不買"; action="多項條件偏弱"
    stop=max(0.01, close-1.5*atr)
    target1=close+1.5*atr
    target2=close+3.0*atr
    rr=(target1-close)/(close-stop) if close>stop else np.nan
    return {"分數":total,"訊號":signal,"動作":action,"理由":reasons[:8],"風險":risks[:8],"技術分":technical,"籌碼分":chip,"基本面分":fundamental,"ATR14":atr,"進場參考":close,"停損參考":stop,"目標1":target1,"目標2":target2,"風險報酬":rr,"突破確認":breakout}
