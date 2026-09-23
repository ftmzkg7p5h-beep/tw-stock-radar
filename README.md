# 台股雷達 PRO V1｜從零開始

這是一個全新、獨立的 Streamlit 台股研究工具，不依賴任何舊專案。

功能：
1. 輸入股票代號查詢
2. 自動抓取日 K
3. 自動辨識常見 K 線型態
4. MA / RSI / MACD / 布林通道 / 成交量
5. 三大法人買賣超與連買/連賣
6. 月營收 YoY / MoM
7. 技術 + 籌碼 + 營收綜合評分
8. 輸出「偏多研究 / 再等等 / 偏弱研究」，並說明原因
9. K 線圖上直接標記辨識到的型態

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器開啟 Streamlit 顯示的網址。

## GitHub + Streamlit Cloud

1. 建立新的 GitHub Repository，例如 `tw-stock-radar`
2. 上傳：
   - app.py
   - kline_engine.py
   - data_sources.py
   - requirements.txt
   - README.md
3. 到 Streamlit Community Cloud 建立 App
4. Repository 選你的專案
5. Main file 選 `app.py`
6. Deploy

## 注意

資料來源會受到公開 API、交易日、網路及網站限制影響。
「偏多研究 / 再等等 / 偏弱研究」是規則化研究訊號，不是保證獲利或個人化投資建議。
