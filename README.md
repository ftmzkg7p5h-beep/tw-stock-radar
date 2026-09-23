# 台股雷達 PRO - final

1. Streamlit Main file path 設為 `app.py`
2. 將整個資料夾內容放入 GitHub 專案根目錄
3. `daily-radar.yml` 放在 `.github/workflows/`
4. 首頁不自動掃描大量股票；輸入股票後才分析
5. 法人歷史改為並行取得，避免逐檔重複 API
6. `kline_engine.py` 已補上 `PATTERN_DESCRIPTIONS`
