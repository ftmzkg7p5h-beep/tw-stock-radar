# 台股雷達 PRO

## 執行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 這版主要修正
- 第一壓力與賣出價分離
- 停損、第一壓力、第二目標、延伸目標分開
- 以 ATR + MA + 近期高低點動態計算，不寫死 19.12 / 20.10
- 持倉直接用「成本 × 股數」計算損益
- 個股分析 / 持倉總覽 / 持倉編輯 / 設定
- Yahoo Finance 資料快取 5 分鐘
- 可直接輸入 2330、3576 等台股代號
