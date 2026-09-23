# 台股雷達 PRO V3｜自動選股完整版本

這版不使用任何型態圖片，K線型態全部由 OHLC 數據自動辨識。

## 功能
- 全市場自動選股
- K線型態自動辨識
- MA / RSI / MACD / 布林通道 / 成交量
- 法人買超、賣超與 5 日 / 20 日統計
- 營收 YoY / MoM
- 公開財報查詢
- 個股詳細分析
- 自動給出：**可以 / 再等等 / 不可以**

## 判斷邏輯
系統依技術、籌碼、基本面規則計分，再轉成三種訊號。這是規則化研究工具，不是獲利保證。

## Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```
