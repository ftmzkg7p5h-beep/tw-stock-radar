"""
台股雷達 PRO V18 - K線型態辨識引擎

目的：把使用者提供的 K 線圖卡轉成可重複執行的 OHLCV 規則。
注意：型態是研究訊號，不是獲利保證；複合型態採近似規則，避免把主觀圖形硬說成精準數學定義。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class Pattern:
    name: str
    group: str
    bullish: int  # -1 bearish, 0 neutral, +1 bullish
    weight: int
    note: str


def _safe(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _metrics(r: pd.Series) -> Dict[str, float]:
    o, h, l, c = map(_safe, [r.get("Open"), r.get("High"), r.get("Low"), r.get("Close")])
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "o": o, "h": h, "l": l, "c": c,
        "rng": rng, "body": body, "upper": upper, "lower": lower,
        "body_pct": body / rng, "close_pos": (c - l) / rng,
    }


def _near_limit(df: pd.DataFrame, idx: int, direction: int) -> bool:
    """台股一般現貨漲跌停的近似辨識；特殊價格/除權息情況仍可能不同。"""
    if idx <= 0:
        return False
    prev = _safe(df.iloc[idx - 1].get("Close"))
    cur = _safe(df.iloc[idx].get("Close"))
    if not np.isfinite(prev) or not np.isfinite(cur) or prev == 0:
        return False
    ret = cur / prev - 1
    return ret >= 0.095 if direction > 0 else ret <= -0.095


def _add(out: List[Pattern], name: str, group: str, bullish: int, weight: int, note: str):
    out.append(Pattern(name, group, bullish, weight, note))


def single_candle_patterns(df: pd.DataFrame) -> List[Pattern]:
    out: List[Pattern] = []
    if len(df) < 2:
        return out
    i = len(df) - 1
    m = _metrics(df.iloc[-1])
    body, rng, upper, lower = m["body"], m["rng"], m["upper"], m["lower"]
    o, c = m["o"], m["c"]
    bull = c >= o

    # 使用者最新圖卡：九種常見單根型態
    if body / rng <= 0.20:
        _add(out, "小陽線" if bull else "小陰線", "單根K", 1 if bull else -1, 2,
             "實體偏小，多空力量接近平衡")

    if 0.20 < body / rng <= 0.65 and upper / rng > 0.08 and lower / rng > 0.08:
        _add(out, "中陽上下影" if bull else "中陰上下影", "單根K", 1 if bull else -1, 2,
             "中等實體且上下皆有影線")

    if bull and _near_limit(df, i, +1) and m["close_pos"] > 0.82 and lower / rng > 0.08:
        _add(out, "下影漲停板", "單根K", 1, 6, "接近漲停且收在高位，下方有明顯下影")

    if lower >= max(body * 2.0, rng * 0.35) and upper <= rng * 0.20 and m["close_pos"] >= 0.55:
        _add(out, "錘頭／錘子線", "單根K", 1, 5, "長下影、短上影，若出現在跌勢末端較有意義")

    if bull and body / rng >= 0.80 and upper / rng <= 0.08 and lower / rng <= 0.08:
        _add(out, "光頭大陽線", "單根K", 1, 5, "多方實體大且上下影極短")

    if body / rng <= 0.08 and upper / rng > 0.25 and lower / rng > 0.25:
        _add(out, "十字星", "單根K", 0, 1, "開收接近，市場可能進入平衡或變盤")

    if upper >= max(body * 2.0, rng * 0.35) and lower <= rng * 0.20 and m["close_pos"] <= 0.55:
        _add(out, "倒錘子線" if bull else "上影線反轉型", "單根K", 1 if bull else -1, 4,
             "長上影、短下影；位置不同，意義可能不同")

    if (not bull) and _near_limit(df, i, -1) and m["close_pos"] < 0.18 and upper / rng > 0.08:
        _add(out, "上影跌停板", "單根K", -1, 6, "接近跌停且收在低位，上方有明顯上影")

    if (not bull) and body / rng >= 0.80 and upper / rng <= 0.08 and lower / rng <= 0.08:
        _add(out, "光頭大陰線", "單根K", -1, 5, "空方實體大且上下影極短")

    # 常見別名/補充
    if lower >= body * 2 and upper <= body * 0.8 and m["close_pos"] > 0.6:
        _add(out, "金針探底", "反轉", 1, 5, "長下影拒絕低檔")
    if upper >= body * 2 and lower <= body * 0.8 and m["close_pos"] < 0.4:
        _add(out, "射擊之星", "反轉", -1, 5, "長上影拒絕高檔")
    return out


def multi_candle_patterns(df: pd.DataFrame) -> List[Pattern]:
    out: List[Pattern] = []
    n = len(df)
    if n < 3:
        return out
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    ma, mb, mc = map(_metrics, [a, b, c])

    # 紅三兵 / 三連陽
    if all(_safe(x.Close) > _safe(x.Open) for x in [a, b, c]) and _safe(a.Close) < _safe(b.Close) < _safe(c.Close):
        _add(out, "紅三兵", "多K", 1, 8, "連三根陽線且收盤逐步墊高")

    # 雙針探底
    if ma["lower"] > ma["body"] * 1.5 and mb["lower"] > mb["body"] * 1.5 and _safe(c.Close) >= _safe(b.Close):
        _add(out, "雙針探底", "反轉", 1, 7, "連續兩日出現明顯下影，低檔承接增加")

    # 早晨之星近似
    if ma["c"] < ma["o"] and mb["body_pct"] < 0.35 and mc["c"] > mc["o"] and mc["c"] > (ma["o"] + ma["c"]) / 2:
        _add(out, "早晨之星", "反轉", 1, 8, "跌勢後出現小實體與強勢反包回升")

    # 曙光出現 / 穿刺
    if ma["c"] < ma["o"] and mc["c"] > mc["o"] and mc["c"] > (ma["o"] + ma["c"]) / 2:
        _add(out, "曙光出現", "反轉", 1, 6, "第二根陽線深入前一根陰線實體")

    # 淡友反攻 / 烏雲蓋頂
    if ma["c"] > ma["o"] and mc["c"] < mc["o"] and mc["c"] < (ma["o"] + ma["c"]) / 2:
        _add(out, "淡友反攻", "反轉", -1, 6, "高位出現陰線並深入前陽線")

    if n >= 5:
        x = df.iloc[-5:]
        closes = pd.to_numeric(x["Close"], errors="coerce")
        opens = pd.to_numeric(x["Open"], errors="coerce")
        if (closes > opens).sum() >= 4 and closes.iloc[-1] > closes.iloc[0]:
            _add(out, "小步上揚", "趨勢", 1, 5, "短線多數交易日收紅並抬高")
        if (closes < opens).sum() >= 4 and closes.iloc[-1] < closes.iloc[0]:
            _add(out, "連續走弱", "趨勢", -1, 5, "短線多數交易日收黑並下移")

    # 漲停雙響炮：強勢K → 小幅整理 → 再次大陽/接近漲停
    if n >= 5:
        r = df.iloc[-5:]
        ret1 = _safe(r.iloc[-4].Close) / max(_safe(r.iloc[-5].Close), 1e-9) - 1
        ret2 = _safe(r.iloc[-1].Close) / max(_safe(r.iloc[-2].Close), 1e-9) - 1
        if ret1 > 0.05 and ret2 > 0.05:
            _add(out, "漲停雙響炮", "突破", 1, 8, "前後皆有明顯強陽，中間出現整理")

    # 上升三法 / 一石二鳥等採較寬鬆的趨勢近似
    if n >= 5:
        r = df.iloc[-5:]
        if _safe(r.iloc[0].Close) < _safe(r.iloc[-1].Close) and (pd.to_numeric(r.Close).diff().dropna() > 0).sum() >= 3:
            _add(out, "上升三法", "趨勢", 1, 5, "主趨勢向上，中間短暫整理後續強")

    return out


def trend_patterns(df: pd.DataFrame) -> List[Pattern]:
    out: List[Pattern] = []
    n = len(df)
    if n < 60:
        return out
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df.get("Volume", pd.Series(index=df.index)), errors="coerce")
    ma5 = close.rolling(5).mean(); ma10 = close.rolling(10).mean(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    last = df.iloc[-1]
    c = _safe(last.Close)

    # 均線
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
        _add(out, "五線順上", "趨勢", 1, 8, "短中期均線呈多頭排列")
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
        _add(out, "均線多頭", "趨勢", 1, 5, "5/10/20日均線多頭排列")
    if ma5.iloc[-2] <= ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
        _add(out, "一陽穿三線", "突破", 1, 8, "短均線由下向上穿越中期均線")
    if ma5.iloc[-2] <= ma10.iloc[-2] and ma5.iloc[-1] > ma10.iloc[-1]:
        _add(out, "黃金交叉", "突破", 1, 5, "5日均線上穿10日均線")

    # 前高突破
    prev_high = pd.to_numeric(df.High, errors="coerce").iloc[-21:-1].max()
    if np.isfinite(prev_high) and c > prev_high:
        _add(out, "突破前高", "突破", 1, 8, "收盤突破近20個交易日高點")

    # 缺口向上：今日低點高於前一日高點
    if _safe(df.iloc[-1].Low) > _safe(df.iloc[-2].High) * 1.002:
        _add(out, "突破缺口", "突破", 1, 7, "向上跳空且缺口尚未回補")

    # 量增
    if len(vol.dropna()) >= 20:
        v20 = vol.rolling(20).mean().iloc[-1]
        if np.isfinite(v20) and vol.iloc[-1] > v20 * 1.5 and c > _safe(last.Open):
            _add(out, "量價突破", "量價", 1, 6, "上漲伴隨成交量放大")
        if np.isfinite(v20) and vol.iloc[-1] > v20 * 2:
            _add(out, "爆量", "量價", 0, 2, "成交量超過20日均量約兩倍")

    # 回踩支撐確認：近10日曾突破20日高，之後回踩10/20MA又收回
    if n >= 30:
        recent = close.iloc[-10:]
        support = ma20.iloc[-1]
        if np.isfinite(support) and recent.min() <= support * 1.02 and c > support and c > _safe(df.iloc[-2].Close):
            _add(out, "回踩支撐確認", "支撐", 1, 7, "回測20日均線後重新轉強")

    # 五日線上 / 上升通道爬坡型：20日斜率為正且最近多數收盤在20MA上方
    if n >= 30:
        ma20_slope = ma20.iloc[-1] / max(ma20.iloc[-11], 1e-9) - 1
        above = (close.iloc[-10:] > ma20.iloc[-10:]).mean()
        if ma20_slope > 0.03 and above >= 0.7:
            _add(out, "上升通道爬坡型", "趨勢", 1, 6, "20日均線上彎且大多數收盤位於其上")

    # 三重底：最近60日的三個局部低點接近
    if n >= 60:
        lows = pd.to_numeric(df.Low, errors="coerce").iloc[-60:].values
        q = np.nanpercentile(lows, 25)
        near = lows[lows <= q * 1.04]
        if len(near) >= 3:
            _add(out, "三重底", "底部", 1, 5, "近60日低點多次在相近區域獲得支撐")

    # 魚躍龍門：盤整後突破
    if n >= 30:
        range30 = (pd.to_numeric(df.High).iloc[-30:-5].max() - pd.to_numeric(df.Low).iloc[-30:-5].min()) / max(close.iloc[-5], 1e-9)
        if range30 < 0.18 and c > pd.to_numeric(df.High).iloc[-30:-1].max():
            _add(out, "魚躍龍門", "突破", 1, 7, "前期波動收斂後向上突破")

    # 老鴨頭近似：多頭排列→短線回檔→再突破
    if n >= 40:
        prior_bull = ma20.iloc[-15] > ma60.iloc[-15]
        pullback = close.iloc[-8:-2].min() < close.iloc[-15:-8].max() * 0.98
        reclaim = c > close.iloc[-15:-1].max() * 0.995
        if prior_bull and pullback and reclaim:
            _add(out, "老鴨頭", "趨勢", 1, 6, "多頭結構中的回檔後再度轉強")

    # 美人肩 / 九九豔陽天：以斜率與連續創高作近似
    if n >= 20:
        if close.iloc[-1] > close.iloc[-5] > close.iloc[-10] and ma20.iloc[-1] > ma20.iloc[-10]:
            _add(out, "九九豔陽天", "趨勢", 1, 6, "短中期價格與20MA同步上行")
        if ma20.iloc[-1] > ma20.iloc[-6] and abs(close.iloc[-1] / max(ma20.iloc[-1],1e-9)-1) < 0.06:
            _add(out, "美人肩", "整理", 1, 3, "上升趨勢中靠近均線整理")

    return out


def detect_patterns(df: pd.DataFrame) -> List[Pattern]:
    if df is None or len(df) < 3:
        return []
    return single_candle_patterns(df) + multi_candle_patterns(df) + trend_patterns(df)


def pattern_names(df: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(p.name for p in detect_patterns(df)))


def pattern_summary(df: pd.DataFrame) -> Dict[str, object]:
    ps = detect_patterns(df)
    bull = sum(p.weight for p in ps if p.bullish > 0)
    bear = sum(p.weight for p in ps if p.bullish < 0)
    neutral = sum(p.weight for p in ps if p.bullish == 0)
    return {
        "patterns": [p.name for p in ps],
        "bull_points": bull,
        "bear_points": bear,
        "neutral_points": neutral,
        "details": [p.__dict__ for p in ps],
    }


def decision(df: pd.DataFrame, institutional_total=np.nan, institutional_streak=0) -> Dict[str, object]:
    """產生『可買 / 再等等 / 不建議』研究訊號。"""
    if df is None or len(df) < 60:
        return {"label":"🟡 再等等", "score":0, "reasons":["歷史資料不足"], "risks":["無法完成完整型態確認"]}

    d = df.copy()
    close = pd.to_numeric(d.Close, errors="coerce")
    ma5 = close.rolling(5).mean(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    rsi = np.nan
    delta = close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    if len(close):
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - 100/(1+rs)).iloc[-1]) if pd.notna((100 - 100/(1+rs)).iloc[-1]) else np.nan

    s = pattern_summary(d)
    score = 50
    reasons: List[str] = []
    risks: List[str] = []

    score += min(22, s["bull_points"])
    score -= min(22, s["bear_points"])

    if close.iloc[-1] > ma20.iloc[-1]:
        score += 8; reasons.append("收盤站上20日均線")
    else:
        score -= 8; risks.append("收盤位於20日均線下方")

    if ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
        score += 7; reasons.append("5/20/60日均線偏多")
    elif ma5.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
        score -= 7; risks.append("均線偏空排列")

    if pd.notna(institutional_total):
        if institutional_total > 0:
            score += 8; reasons.append("三大法人合計買超")
        elif institutional_total < 0:
            score -= 8; risks.append("三大法人合計賣超")
    else:
        risks.append("法人資料未取得，不以0張代替")

    if institutional_streak >= 3:
        score += 5; reasons.append(f"法人連續買超{institutional_streak}日")

    if pd.notna(rsi):
        if rsi >= 80:
            score -= 8; risks.append(f"RSI {rsi:.1f}，短線過熱")
        elif rsi >= 75:
            score -= 4; risks.append(f"RSI {rsi:.1f}，偏熱")
        elif 50 <= rsi < 70:
            score += 3; reasons.append(f"RSI {rsi:.1f}，多方仍有空間")
        elif rsi < 35:
            risks.append(f"RSI {rsi:.1f}，弱勢區")

    score = int(max(0, min(100, score)))
    bullish_names = set(s["patterns"]) & {
        "金針探底","紅三兵","早晨之星","曙光出現","錘頭／錘子線",
        "五線順上","均線多頭","一陽穿三線","突破前高","突破缺口",
        "量價突破","回踩支撐確認","老鴨頭","九九豔陽天","漲停雙響炮",
        "上升三法","魚躍龍門"
    }
    bearish_names = set(s["patterns"]) & {"上影跌停板","光頭大陰線","射擊之星","淡友反攻","連續走弱"}

    # 需要『型態 + 趨勢 + 籌碼』至少同時成立，避免單根K就給買進訊號。
    institutional_ok = pd.notna(institutional_total) and institutional_total > 0
    trend_ok = close.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-1] > ma20.iloc[-1]
    hot = pd.notna(rsi) and rsi >= 78

    if score >= 78 and bullish_names and trend_ok and institutional_ok and not hot:
        label = "🟢 可買條件成立"
    elif score >= 58 and not bearish_names:
        label = "🟡 再等等"
        if bullish_names:
            risks.append("有偏多型態，但尚缺少趨勢/法人/價格確認")
    else:
        label = "🔴 不建議"
        if bearish_names:
            risks.append("偵測到偏空或反轉風險型態")

    if not reasons:
        reasons.append("尚未形成足夠的多方共振")
    if not risks:
        risks.append("仍需觀察下一交易日是否確認型態")

    return {
        "label": label,
        "score": score,
        "reasons": list(dict.fromkeys(reasons))[:6],
        "risks": list(dict.fromkeys(risks))[:6],
        "patterns": s["patterns"],
        "rsi": rsi,
    }
