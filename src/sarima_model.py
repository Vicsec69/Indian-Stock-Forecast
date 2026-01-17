import pandas as pd
import os
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROCESSED_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\processed"
OUT_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\sarima_range"
os.makedirs(OUT_DIR, exist_ok=True)

FORECAST_WEEKS = 26  # ~6 months


def sarima_range(stock):
    """
    Runs SARIMA and returns forecast range + trend metrics
    """
    path = os.path.join(PROCESSED_DIR, f"{stock}_weekly.csv")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    if "Close" not in df.columns or len(df) < 60:
        raise ValueError("Insufficient data for SARIMA")

    close = df["Close"]

    model = SARIMAX(
        close,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)
    forecast = results.get_forecast(FORECAST_WEEKS)

    mean = forecast.predicted_mean
    ci = forecast.conf_int()

    current_price = close.iloc[-1]
    future_price = mean.iloc[-1]

    expected_return = (future_price - current_price) / current_price

    # Trend score: squash to [-1, 1]
    trend_score = float(np.tanh(expected_return * 5))

    return {
        "Stock": stock,
        "Current_Price": round(current_price, 2),
        "Lower_6M": round(ci.iloc[-1, 0], 2),
        "Upper_6M": round(ci.iloc[-1, 1], 2),
        "Mean_6M": round(future_price, 2), 
        "Expected_Return_6M": round(expected_return, 4),
        "SARIMA_Trend_Score": round(trend_score, 4)
    }

def run():
    rows = []

    for f in os.listdir(PROCESSED_DIR):
        if not f.endswith("_weekly.csv"):
            continue

        stock = f.replace("_weekly.csv", "")

        try:
            print(f"📈 SARIMA processing: {stock}")
            rows.append(sarima_range(stock))
        except Exception as e:
            print(f"❌ SARIMA failed for {stock}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sarima_6m_range.csv"),
            index=False
        )
        print("✅ SARIMA output saved")
    else:
        print("⚠️ No SARIMA results generated")


if __name__ == "__main__":
    run()
