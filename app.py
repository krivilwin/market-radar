import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Market Radar", page_icon="📡", layout="wide")

DEFAULT_UNIVERSE = {
    "Stocks": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AMD","AVGO","NFLX","COST",
               "JPM","V","MA","LLY","UNH","WMT","HD","NKE","CRM","ORCL"],
    "ETFs": ["SPY","QQQ","VTI","VOO","IWM","DIA","XLK","XLF","XLE","XLV"]
}
DATA_FILE = Path("paper_portfolio.json")
MAX_SCAN = 6

MY_PORTFOLIO = [
    {"symbol": "ASYS", "shares": 5.8, "avg_price": 14.88, "currency": "USD"},
    {"symbol": "VWRP", "shares": 0.47846889, "avg_price": 146.30, "currency": "GBP"},
]

def api_key():
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        return ""

@st.cache_data(ttl=21600, show_spinner=False)
def history(symbol, interval="1day", outputsize=260):
    key = api_key()
    if not key:
        return None, "API key is not configured."
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "apikey": key, "format": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
    except Exception as e:
        return None, str(e)
    if "values" not in data:
        message = data.get("message", "No market data returned.")
        if "credit" in message.lower() or "limit" in message.lower():
            message = "Twelve Data free-plan limit reached. Wait about one minute, then try again."
        return None, message
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").dropna(subset=["close"]).reset_index(drop=True)
    return df, None

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def enrich(df):
    d = df.copy()
    d["sma20"] = d.close.rolling(20).mean()
    d["sma50"] = d.close.rolling(50).mean()
    d["sma200"] = d.close.rolling(200).mean()
    d["rsi14"] = rsi(d.close)
    ema12 = d.close.ewm(span=12, adjust=False).mean()
    ema26 = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["vol20"] = d.volume.rolling(20).mean()
    d["ret20"] = d.close.pct_change(20) * 100
    d["ret60"] = d.close.pct_change(60) * 100
    d["high52"] = d.high.rolling(252, min_periods=100).max()
    d["low20"] = d.low.rolling(20).min()
    d["atr"] = pd.concat([
        d.high-d.low,
        (d.high-d.close.shift()).abs(),
        (d.low-d.close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    return d

def score_symbol(df, mode="Swing"):
    d = enrich(df)
    x = d.iloc[-1]
    p = float(x.close)
    score = 50
    reasons = []
    if pd.notna(x.sma200):
        if p > x.sma200: score += 10; reasons.append("Price above 200-day trend")
        else: score -= 10; reasons.append("Price below 200-day trend")
    if pd.notna(x.sma50):
        if p > x.sma50: score += 7; reasons.append("Above 50-day average")
        else: score -= 5
    if pd.notna(x.rsi14):
        if 40 <= x.rsi14 <= 60: score += 8; reasons.append("RSI in healthy zone")
        elif 30 <= x.rsi14 < 40: score += 10; reasons.append("RSI shows a controlled pullback")
        elif x.rsi14 < 30: score += 5; reasons.append("Oversold: possible rebound, higher risk")
        elif x.rsi14 > 75: score -= 12; reasons.append("Overbought / extended")
        elif x.rsi14 > 68: score -= 6
    if pd.notna(x.macd) and pd.notna(x.macd_signal):
        if x.macd > x.macd_signal: score += 8; reasons.append("MACD momentum positive")
        else: score -= 4
    if pd.notna(x.ret20):
        if 0 < x.ret20 < 12: score += 6
        elif x.ret20 > 20: score -= 6; reasons.append("Recent move may be extended")
    if pd.notna(x.volume) and pd.notna(x.vol20) and x.vol20 > 0 and x.volume > x.vol20*1.25:
        score += 5; reasons.append("Volume above 20-day average")
    if mode == "Investment" and pd.notna(x.sma200) and p > x.sma200:
        score += 5
    score = int(max(0, min(100, round(score))))
    if score >= 85: signal = "🟢 STRONG SETUP"
    elif score >= 75: signal = "🟢 POTENTIAL BUY"
    elif score >= 65: signal = "🟡 WATCH"
    elif score >= 50: signal = "⚪ WAIT"
    else: signal = "🔴 AVOID"
    atr = float(x.atr) if pd.notna(x.atr) and x.atr > 0 else p*0.025
    entry_low = max(0, p - 0.5*atr)
    entry_high = p + 0.25*atr
    stop = max(0, p - (2.0 if mode=="Swing" else 2.5)*atr)
    target = p + (3.0 if mode=="Swing" else 4.0)*atr
    risk = p-stop
    rr = (target-p)/risk if risk > 0 else np.nan
    return {"price":p,"score":score,"signal":signal,
            "rsi":float(x.rsi14) if pd.notna(x.rsi14) else np.nan,
            "entry_low":entry_low,"entry_high":entry_high,"stop":stop,
            "target":target,"rr":rr,"reasons":reasons[:5],"data":d}

def load_portfolio():
    if DATA_FILE.exists():
        try: return json.loads(DATA_FILE.read_text())
        except Exception: pass
    return {"starting_cash":10000.0,"cash":10000.0,"positions":[],"closed":[]}

def save_portfolio(p):
    DATA_FILE.write_text(json.dumps(p, indent=2))

if "scan_offset" not in st.session_state:
    st.session_state.scan_offset = 0
if "scan_results" not in st.session_state:
    st.session_state.scan_results = {}
if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = {}
if "scan_universe_key" not in st.session_state:
    st.session_state.scan_universe_key = ""

st.title("📡 Market Radar V2")
st.caption("Opportunity scanner + paper trading. Research tool only — signals are probabilistic, not guarantees.")

mode = st.sidebar.radio("Strategy", ["Swing", "Investment"])
st.sidebar.markdown("**Data:** Twelve Data")
if api_key(): st.sidebar.success("API key configured")
else: st.sidebar.warning("Add TWELVE_DATA_API_KEY to Streamlit secrets.")
st.sidebar.info("Free-plan mode: up to 6 tickers per scan. Data is cached for 6 hours.")

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Opportunities","🔎 Analyse","💼 My Portfolio","💷 Paper Portfolio"])

with tab1:
    st.subheader(f"{mode} opportunities")
    universe_type = st.selectbox("Universe", ["Stocks","ETFs"])
    default_symbols = ",".join(DEFAULT_UNIVERSE[universe_type])
    symbols_text = st.text_area("Symbols to scan", default_symbols, height=90)
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols_text.split(",") if s.strip()))

    universe_key = f"{mode}|{universe_type}|" + ",".join(symbols)
    if universe_key != st.session_state.scan_universe_key:
        st.session_state.scan_universe_key = universe_key
        st.session_state.scan_offset = 0
        st.session_state.scan_results = {}
        st.session_state.scan_errors = {}

    if symbols:
        total_batches = int(np.ceil(len(symbols) / MAX_SCAN))

        # Never move backwards automatically. Offset points to the next batch to scan.
        st.session_state.scan_offset = max(
            0, min(st.session_state.scan_offset, max(0, len(symbols) - 1))
        )

        scanned_count = sum(s in st.session_state.scan_results for s in symbols)

        if scanned_count >= len(symbols):
            st.success(f"✅ Scan complete — {scanned_count}/{len(symbols)} tickers scanned.")
            if st.button("Start new scan"):
                st.session_state.scan_offset = 0
                st.session_state.scan_results = {}
                st.session_state.scan_errors = {}
                st.rerun()
        else:
            # Find the first not-yet-scanned ticker and start its batch there.
            remaining_indices = [
                i for i, s in enumerate(symbols)
                if s not in st.session_state.scan_results
            ]
            if remaining_indices:
                st.session_state.scan_offset = (remaining_indices[0] // MAX_SCAN) * MAX_SCAN

            batch_no = st.session_state.scan_offset // MAX_SCAN + 1
            batch = symbols[
                st.session_state.scan_offset:
                st.session_state.scan_offset + MAX_SCAN
            ]

            st.info(
                f"Free-plan protection: max {MAX_SCAN} tickers per scan. "
                f"Next: Batch {batch_no} of {total_batches}. Cached for 6 hours."
            )
            st.caption(f"Next batch: {', '.join(batch)}")
            st.caption(f"Progress: {scanned_count}/{len(symbols)} scanned")

            button_label = (
                "Scan first batch" if scanned_count == 0
                else f"Scan next batch ({batch_no}/{total_batches})"
            )

            if st.button(button_label, type="primary"):
                st.write("Scanning: **" + ", ".join(batch) + "**")
                bar = st.progress(0)
                successful = 0

                for i, symbol in enumerate(batch):
                    df, err = history(symbol)
                    if df is not None and len(df) >= 60:
                        a = score_symbol(df, mode)
                        st.session_state.scan_results[symbol] = {
                            "Symbol": symbol,
                            "Score": a["score"],
                            "Signal": a["signal"],
                            "Price": round(a["price"], 2),
                            "RSI": round(a["rsi"], 1),
                            "Entry low": round(a["entry_low"], 2),
                            "Entry high": round(a["entry_high"], 2),
                            "Target": round(a["target"], 2),
                            "Stop": round(a["stop"], 2),
                            "R/R": round(a["rr"], 2),
                        }
                        st.session_state.scan_errors.pop(symbol, None)
                        successful += 1
                    else:
                        msg = err or "Not enough market history returned."
                        st.session_state.scan_errors[symbol] = msg
                    bar.progress((i + 1) / max(1, len(batch)))

                # Advance only when every ticker in this batch succeeded.
                batch_done = all(
                    s in st.session_state.scan_results for s in batch
                )
                if batch_done:
                    next_offset = st.session_state.scan_offset + MAX_SCAN
                    st.session_state.scan_offset = min(
                        next_offset, max(0, len(symbols) - 1)
                    )
                    st.session_state.scan_errors = {
                        k: v for k, v in st.session_state.scan_errors.items()
                        if k not in batch
                    }
                    st.rerun()
                else:
                    st.warning(
                        f"{successful}/{len(batch)} tickers completed. "
                        "The batch was not advanced so failed tickers can be retried."
                    )

        all_rows = [
            st.session_state.scan_results[s]
            for s in symbols
            if s in st.session_state.scan_results
        ]

        if all_rows:
            st.divider()
            st.markdown(
                f"## 🏆 Overall ranking — {len(all_rows)}/{len(symbols)} scanned"
            )
            overall = pd.DataFrame(all_rows).sort_values(
                ["Score", "R/R"], ascending=False
            )
            st.dataframe(overall, use_container_width=True, hide_index=True)
            top = overall[overall.Score >= 75]
            if not top.empty:
                st.success(
                    f"{len(top)} setup(s) currently score 75 or higher "
                    "across all scanned batches."
                )

        if st.session_state.scan_errors:
            first_symbol = next(iter(st.session_state.scan_errors))
            st.warning(
                f"{first_symbol}: {st.session_state.scan_errors[first_symbol]}"
            )
    else:
        st.warning("Add at least one ticker.")

with tab2:
    symbol = st.text_input("Ticker", "NKE").strip().upper()
    if st.button("Analyse ticker"):
        df,err = history(symbol)
        if df is None:
            st.error(err)
        else:
            a = score_symbol(df, mode)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Price",f"${a['price']:.2f}")
            c2.metric("Opportunity score",f"{a['score']}/100")
            c3.metric("RSI (14)",f"{a['rsi']:.1f}")
            c4.metric("Risk / reward",f"{a['rr']:.2f}")
            st.subheader(a["signal"])
            st.write(f"**Entry zone:** ${a['entry_low']:.2f}–${a['entry_high']:.2f}  |  **Target:** ${a['target']:.2f}  |  **Invalidation/stop:** ${a['stop']:.2f}")
            if a["reasons"]: st.write("**Why:** " + " · ".join(a["reasons"]))
            d = a["data"].tail(180)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d.datetime,open=d.open,high=d.high,low=d.low,close=d.close,name=symbol))
            fig.add_trace(go.Scatter(x=d.datetime,y=d.sma50,name="SMA 50"))
            fig.add_trace(go.Scatter(x=d.datetime,y=d.sma200,name="SMA 200"))
            fig.update_layout(height=520,xaxis_rangeslider_visible=False)
            st.plotly_chart(fig,use_container_width=True)

with tab3:
    st.subheader("💼 My Portfolio")
    st.caption("Your real positions. Market data may be delayed and is cached for up to 6 hours.")

    rows = []
    for pos in MY_PORTFOLIO:
        df, err = history(pos["symbol"])
        if df is not None and len(df) >= 60:
            a = score_symbol(df, mode)
            current = a["price"]
            ret = ((current / pos["avg_price"]) - 1) * 100
            if a["score"] >= 75: action = "🟢 HOLD / STRONG"
            elif a["score"] >= 65: action = "🟡 HOLD / WATCH"
            elif a["score"] >= 50: action = "⚪ WATCH CLOSELY"
            else: action = "🔴 REVIEW POSITION"
            rows.append({"Symbol":pos["symbol"],"Shares":pos["shares"],"Avg price":pos["avg_price"],
                         "Current":round(current,2),"Return %":round(ret,2),"Score":a["score"],
                         "Signal":a["signal"],"RSI":round(a["rsi"],1),"Action":action,
                         "Currency":pos["currency"]})
        else:
            rows.append({"Symbol":pos["symbol"],"Shares":pos["shares"],"Avg price":pos["avg_price"],
                         "Current":np.nan,"Return %":np.nan,"Score":np.nan,
                         "Signal":err or "No data","RSI":np.nan,"Action":"⚪ DATA UNAVAILABLE",
                         "Currency":pos["currency"]})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for row in rows:
        st.markdown(f"### {row['Symbol']} — {row['Action']}")
        c1,c2,c3 = st.columns(3)
        prefix = "$" if row["Currency"] == "USD" else "£"
        c1.metric("Current", f"{prefix}{row['Current']:.2f}" if pd.notna(row["Current"]) else "N/A")
        c2.metric("Return", f"{row['Return %']:+.2f}%" if pd.notna(row["Return %"]) else "N/A")
        c3.metric("Radar score", f"{int(row['Score'])}/100" if pd.notna(row["Score"]) else "N/A")
        st.caption(f"Average price: {prefix}{row['Avg price']:.2f} • Shares: {row['Shares']} • {row['Signal']}")
        st.divider()
    st.info("ASYS is tracked in USD and VWRP in GBP. Return % does not include FX effects, fees or taxes.")

with tab4:
    p = load_portfolio()
    c1,c2 = st.columns(2)
    c1.metric("Paper cash",f"£{p['cash']:,.2f}")
    c2.metric("Open positions",len(p["positions"]))
    st.caption("Paper trades are recorded manually. Automatic mark-to-market and exit monitoring can be added later.")
    with st.form("paperbuy"):
        s = st.text_input("Symbol").upper()
        price = st.number_input("Entry price (USD)",min_value=0.01,value=100.0)
        amount = st.number_input("Paper amount (£)",min_value=1.0,value=200.0)
        submitted = st.form_submit_button("Paper BUY")
        if submitted:
            if amount > p["cash"]: st.error("Not enough paper cash.")
            elif not s: st.error("Enter a symbol.")
            else:
                p["positions"].append({"symbol":s,"entry_price":price,"amount_gbp":amount,"strategy":mode})
                p["cash"] -= amount
                save_portfolio(p)
                st.success(f"Paper position added: {s} — £{amount:.2f}")
                st.rerun()
    if p["positions"]: st.dataframe(pd.DataFrame(p["positions"]),use_container_width=True,hide_index=True)
    else: st.write("No paper positions yet.")

st.divider()
st.caption("Market Radar V2 • Free-plan optimized • Paper/research use. Entry, target and stop levels are model rules, not financial advice.")
