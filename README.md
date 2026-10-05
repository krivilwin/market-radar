# Market Radar V1

Private market opportunity scanner for US stocks and ETFs.

## Features
- Swing and Investment modes
- RSI, MACD, SMA20/50/200, momentum and volume analysis
- Opportunity Score (0-100)
- BUY / WATCH / WAIT / AVOID signals
- Suggested entry zone, target and invalidation/stop level
- Paper trading portfolio starting at £10,000
- Twelve Data integration

## Run locally
1. Install Python 3.11+
2. `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml`:
   ```toml
   TWELVE_DATA_API_KEY = "YOUR_KEY"
   ```
4. `streamlit run app.py`

This is a research/paper-trading tool, not personalised financial advice.
