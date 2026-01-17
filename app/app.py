import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier

# ---------------- CONFIG ---------------- #
FORECAST_WEEKS = 26   # ~6 months

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(
    page_title="Indian Stock Forecast",
    layout="wide"
)

st.title("📈 Indian Stock Forecast & Valuation System")
st.caption("Hybrid model: XGBoost (direction) + SARIMA (price range)")

# ---------------- HELPERS ---------------- #
def standardize_ohlcv(df):
    """
    Force Yahoo Finance columns into standard OHLCV names
    """
    col_map = {}
    for c in df.columns:
        c_str = str(c).lower()
        if "open" in c_str:
            col_map[c] = "Open"
        elif "high" in c_str:
            col_map[c] = "High"
        elif "low" in c_str:
            col_map[c] = "Low"
        elif "close" in c_str and "adj" not in c_str:
            col_map[c] = "Close"
        elif "volume" in c_str:
            col_map[c] = "Volume"

    df = df.rename(columns=col_map)
    return df

def normalize_columns(df):
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert everything to string safely
    df.columns = [str(c) for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def fetch_daily(stock):
    df = yf.download(stock + ".NS", period="10y", auto_adjust=False)
    if df.empty:
        raise ValueError("No data found")
    df.reset_index(inplace=True)
    

    return df


def daily_to_weekly(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    weekly = weekly.asfreq("W-FRI")
    return weekly

@st.cache_data(show_spinner=False)
def xgb_direction(weekly):
    df = weekly.copy()

    df["ret1"] = df["Close"].pct_change(1)
    df["ret2"] = df["Close"].pct_change(2)
    df["ma12"] = df["Close"].rolling(12).mean()
    df["ma26"] = df["Close"].rolling(26).mean()
    df["vol"] = df["Close"].pct_change().rolling(12).std()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    X = df[["ret1", "ret2", "ma12", "ma26", "vol"]]
    y = df["target"]

    model = XGBClassifier(
        n_estimators=40,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss"
    )

    model.fit(X[:-1], y[:-1])

    prob_up = model.predict_proba(X.tail(1))[0][1]
    return prob_up

@st.cache_data(show_spinner=False)
def sarima_range(close):
    close = close[-156:]  # last 3 years only (speed + relevance)

    model = SARIMAX(
        close,
        order=(1,1,1),
        seasonal_order=(0,1,1,52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    res = model.fit(disp=False)
    fc = res.get_forecast(FORECAST_WEEKS)
    ci = fc.conf_int()

    return (
        close.iloc[-1],
        ci.iloc[-1, 0],
        ci.iloc[-1, 1]
    )

def get_fundamentals(stock):
    t = yf.Ticker(stock + ".NS")
    info = t.info or {}

    # Conservative fallback scoring
    fund_score = 0.5 if info else 0.4
    val_score = 0.5 if info else 0.4

    return fund_score, val_score

def final_decision(direction, fund, val, trend):
    final_score = (
        0.4 * direction +
        0.3 * fund +
        0.2 * val +
        0.1 * trend
    )

    if final_score >= 0.65:
        return "BUY", "🟢 High", final_score
    if final_score >= 0.45:
        return "HOLD", "🟡 Medium", final_score
    return "AVOID", "🔴 Low", final_score

# ---------------- UI ---------------- #

stock = st.text_input(
    "Enter NSE Stock Symbol (example: ITC, TCS, SYMPHONY)",
    placeholder="SYMPHONY"
)

if st.button("Analyze Stock") and stock:
    try:
        with st.spinner("Fetching data & running models (may take ~15 seconds)..."):
            daily = fetch_daily(stock.upper())
            weekly = daily_to_weekly(daily)

            direction_score = xgb_direction(weekly)
            current, low, high = sarima_range(weekly["Close"])
            fund_score, val_score = get_fundamentals(stock)

            trend_score = max((high - current) / current, 0)
            signal, confidence, final_score = final_decision(
                direction_score, fund_score, val_score, trend_score
            )

        # ---------------- RESULTS ---------------- #

        st.subheader(f"📊 Analysis for {stock.upper()}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Direction Confidence", f"{direction_score*100:.1f}%")
        col2.metric("Final Signal", signal)
        col3.metric("Confidence Level", confidence)

        st.markdown(f"""
        **Current Price:** ₹{current:.2f}  
        **Expected 6-Month Range:** ₹{int(low)} – ₹{int(high)}  
        **Final Weighted Score:** {final_score:.2f}
        """)

        # ---------------- PLOT ---------------- #

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(weekly.index, weekly["Close"], label="Weekly Close")
        ax.axhline(low, linestyle="--", color="red", label="6M Lower Range")
        ax.axhline(high, linestyle="--", color="green", label="6M Upper Range")
        ax.set_title(f"{stock.upper()} – Weekly Price & 6-Month Range")
        ax.legend()
        st.pyplot(fig)

        # ---------------- EXPLANATION ---------------- #

        st.subheader("🧠 Why this signal?")
        st.write(
            f"""
            • Direction model estimates **{direction_score*100:.1f}%** probability of upside  
            • Fundamentals score used conservatively due to data variability  
            • Valuation applied as a stabilizer  
            • SARIMA provides probabilistic price range (not exact price)
            """
        )

        st.caption("⚠️ This is a probabilistic model, not investment advice.")

    except Exception as e:
        st.error(f"❌ Unable to analyze stock: {e}")
