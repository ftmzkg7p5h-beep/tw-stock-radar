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
    """雙軌評分：趨勢確認 + 早期轉強。

    早期雷達刻意不追求「已經突破才給高分」，而是尋找：
    低/中低位階、跌深止穩、均線剛轉上、動能改善、量能回溫、法人轉向，
    且尚未過熱的組合。這是「提前觀察」模型，不是預測保證。
    """
    last=d.iloc[-1]
    prev=d.iloc[-2] if len(d)>=2 else last
    technical=0; reasons=[]; risks=[]
    bw={"紅三兵":7,"早晨之星":7,"多頭吞噬":7,"錘頭／錘子線":5,"雙針探底":5,"一陽穿三線":7,"五線順上":7,"均線多頭":5,"均線黃金交叉":6,"布林突破":7,"突破前高":8,"回踩支撐確認":7,"量增":4,"爆量":2,"短線上升":4,"曙光出現":5,"反攻":5,"光頭大陽線":5}
    sw={"烏雲蓋頂":-8,"空頭吞噬":-9,"長黑K":-6,"均線空頭":-8,"均線死亡交叉":-8,"高檔長上影":-5}
    for x in bullish:
        if x in bw: technical += bw[x]; reasons.append(x)
    for x in bearish:
        if x in sw: technical += sw[x]; risks.append(x)

    close=float(last.Close)
    ma5=float(last.MA5) if pd.notna(last.MA5) else close
    ma10=float(last.MA10) if pd.notna(last.MA10) else close
    ma20=float(last.MA20) if pd.notna(last.MA20) else close
    ma60=float(last.MA60) if pd.notna(last.MA60) else close
    atr=float(last.ATR14) if pd.notna(last.ATR14) and float(last.ATR14)>0 else max(close*0.02,0.01)
    rsi=float(last.RSI) if pd.notna(last.RSI) else np.nan

    if pd.notna(last.MA20): technical += 5 if close>ma20 else -5
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

    # ---------- Early Trend Radar v4.1 ----------
    early=50
    early_reasons=[]; early_risks=[]; setup_flags=[]
    dist20=(close/ma20-1)*100 if ma20 else 0
    ret20=float(last.Return20) if pd.notna(last.Return20) else np.nan

    # A. 價格位置：不要求跌到最低，但要求沒有離 MA20 太遠。
    if -3 <= dist20 <= 4:
        early += 11; early_reasons.append("股價貼近MA20，位置未過熱"); setup_flags.append("位置")
    elif 4 < dist20 <= 7:
        early += 5; early_reasons.append("仍在MA20上方合理距離")
    elif 7 < dist20 <= 10:
        early += 1
    elif dist20 > 12:
        early -= 14; early_risks.append("股價離MA20過遠")
    elif dist20 < -10:
        early -= 7; early_risks.append("仍低於MA20過多")

    # B. 60日位置：尋找「低/中低位開始回升」，避免把60日高檔當早期。
    if len(d)>=60:
        lo60=float(d.Low.iloc[-60:].min()); hi60=float(d.High.iloc[-60:].max())
        pos60=(close-lo60)/(hi60-lo60)*100 if hi60>lo60 else 50
        if 20 <= pos60 <= 60:
            early += 9; early_reasons.append("60日位階偏低至中段"); setup_flags.append("低位")
        elif 60 < pos60 <= 75:
            early += 3
        elif pos60 > 85:
            early -= 8; early_risks.append("接近60日高檔")
    else:
        pos60=np.nan

    # C. 均線剛轉上，而不是只獎勵已經多頭排列。
    if len(d)>=6 and pd.notna(d.MA5.iloc[-6]) and pd.notna(d.MA20.iloc[-6]):
        slope5=(ma5/float(d.MA5.iloc[-6])-1)*100
        slope20=(ma20/float(d.MA20.iloc[-6])-1)*100
        if slope5>0 and slope20>=0:
            early += 10; early_reasons.append("MA5/MA20開始轉上"); setup_flags.append("均線")
        elif slope5>0:
            early += 6; early_reasons.append("MA5開始轉上")
        elif slope5<0 and slope20<0:
            early -= 8; early_risks.append("短中期均線仍向下")
    else:
        slope5=slope20=0

    # D. RSI抓「從低中位往上」，不是等到75才追。
    rsi_prev=float(prev.RSI) if pd.notna(prev.RSI) else np.nan
    if pd.notna(rsi):
        if 42<=rsi<=60 and (pd.isna(rsi_prev) or rsi>=rsi_prev):
            early += 10; early_reasons.append("RSI低中位向上"); setup_flags.append("動能")
        elif 35<=rsi<42 and (pd.isna(rsi_prev) or rsi>rsi_prev):
            early += 5; early_reasons.append("RSI由弱轉升")
        elif rsi>=72:
            early -= 11; early_risks.append("RSI偏熱")

    # E. MACD柱體收斂，抓轉折前段。
    hist=float(last.MACDHist) if pd.notna(last.MACDHist) else np.nan
    prev_hist=float(prev.MACDHist) if pd.notna(prev.MACDHist) else np.nan
    if pd.notna(hist) and pd.notna(prev_hist):
        if hist>prev_hist and hist<=0:
            early += 10; early_reasons.append("MACD空方動能收斂"); setup_flags.append("動能")
        elif hist>0 and hist>=prev_hist:
            early += 5; early_reasons.append("MACD動能轉強")
        elif hist<prev_hist:
            early -= 4; early_risks.append("MACD動能轉弱")

    # F. 量能只獎勵溫和回溫，爆量反而降低早期分。
    v5=float(last.V5) if pd.notna(last.V5) else np.nan
    vr=float(last.Volume)/v5 if pd.notna(v5) and v5>0 else np.nan
    if pd.notna(vr):
        if 1.05<=vr<=1.8:
            early += 7; early_reasons.append("量能溫和回溫"); setup_flags.append("量能")
        elif vr>2.2:
            early -= 6; early_risks.append("末端爆量，避免追價")

    # G. 前高：距離不遠但尚未突破，最符合「突破前」觀察。
    if len(d)>=21:
        prev_high=float(d.High.iloc[-21:-1].max())
        gap=(prev_high-close)/prev_high*100 if prev_high else np.nan
        if pd.notna(gap) and 0<gap<=7:
            early += 9; early_reasons.append("接近前高但尚未突破"); setup_flags.append("前高")
        elif pd.notna(gap) and gap<0:
            early -= 2
    else:
        gap=np.nan

    # H. 20日報酬：越接近零/小幅上漲越符合早期；大漲直接扣分。
    if pd.notna(ret20):
        if -3<=ret20<=8:
            early += 8; early_reasons.append("20日漲幅仍早期"); setup_flags.append("漲幅")
        elif 8<ret20<=12:
            early += 3
        elif ret20>15:
            early -= 10; early_risks.append("20日漲幅偏大")
        elif ret20<-15:
            early -= 6; early_risks.append("中期仍弱")

    # I. 法人轉向：短期轉正、中期仍未完全轉正，是很重要的早期訊號。
    if pd.notna(institution_5d):
        if institution_5d>0 and (pd.isna(institution_20d) or institution_20d<=0):
            early += 10; early_reasons.append("法人由弱轉強"); setup_flags.append("法人")
        elif institution_5d>0:
            early += 5; early_reasons.append("法人近5日偏多")
        elif institution_5d<0:
            early -= 3

    if pd.notna(revenue_yoy):
        if revenue_yoy>10: early += 5; early_reasons.append("營收成長")
        elif revenue_yoy>0: early += 2

    early=int(np.clip(early,0,100))
    setup_count=len(set(setup_flags))
    breakout=any(x in bullish for x in ["突破前高","布林突破"])
    overheat=(pd.notna(rsi) and rsi>=72) or dist20>10 or (pd.notna(ret20) and ret20>15) or (pd.notna(pos60) and pos60>88)

    # 早期訊號不再被「總分」卡死；但至少要有多個獨立早期條件成立。
    early_ready=(early>=68 and setup_count>=3 and not breakout and not overheat)
    if overheat:
        signal="🟠 不追高"; action="趨勢可能仍強，但位置已偏高；等待拉回到低風險區，不追最後一段"
    elif early_ready:
        signal="🟢 早期佈局"; action="尚未明顯過熱，已有多個領先條件同步改善；列入提前觀察區"
    elif total>=85:
        signal="🟢 買進條件成立"; action="趨勢、籌碼與基本面同時偏多；仍需依停損執行"
    elif total>=75 and breakout:
        signal="🔵 突破確認"; action="突破型態成立；確認量能與停損後再處理"
    elif total>=68:
        signal="🟡 等待"; action="條件尚未完整，等待拉回支撐或突破確認"
    else:
        signal="🔴 不買"; action="目前訊號偏弱，不以單一指標逆勢進場"

    if total<50 and not early_ready and not overheat:
        signal="🔴 不買"; action="目前趨勢與早期轉強條件都不足"

    stop=max(0.01, close-1.5*atr)
    target1=close+1.5*atr
    target2=close+3.0*atr
    rr=(target1-close)/(close-stop) if close>stop else np.nan
    entry_zone_low=max(0.01, min(close, ma20)-0.75*atr)
    entry_zone_high=min(close+0.5*atr, ma20+1.0*atr)
    return {
        "分數":total,"早期趨勢分":early,"早期條件數":setup_count,"訊號":signal,"動作":action,
        "理由":reasons[:8],"風險":risks[:8],"早期理由":early_reasons[:10],"早期風險":early_risks[:8],
        "技術分":technical,"籌碼分":chip,"基本面分":fundamental,
        "ATR14":atr,"進場參考":close,"進場區下緣":entry_zone_low,"進場區上緣":entry_zone_high,
        "停損參考":stop,"目標1":target1,"目標2":target2,"風險報酬":rr,"突破確認":breakout,
        "位置距MA20%":dist20,"20日漲幅%":ret20,"60日位階%":pos60,"量能比5日均量":vr,
    }
