import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- IMPORT PIPELINE MODULES ----------------
from data_fetcher import run_pipeline, normalize_symbol
from data_preprocessor import convert_daily_to_weekly
from sarima_model import sarima_range
from xgboost_direction import run_xgboost
from lstm_price_model import run_lstm
from fundamentals_model import run_fundamentals
from seasonality_model import run_seasonality
from decision_engine import decision_engine


# ---------------- VISUALIZATION ----------------
def plot_results(stock, weekly, sarima_out, xgb_out):
    plt.figure(figsize=(12, 6))

    # Weekly close
    plt.plot(
        weekly.index,
        weekly["Close"],
        label="Weekly Close",
        linewidth=2,
        color="blue"
    )

    # SARIMA upper & lower bounds
    plt.axhline(
        sarima_out["Upper_6M"],
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="SARIMA Upper (6M)"
    )

    plt.axhline(
        sarima_out["Lower_6M"],
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="SARIMA Lower (6M)"
    )

    # ✅ SARIMA mean / average forecast
    plt.axhline(
        sarima_out["Mean_6M"],
        color="green",
        linestyle="-",
        linewidth=2.5,
        label="SARIMA Mean Forecast (6M)"
    )

    plt.title(f"{stock} – Weekly Price & 6M SARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # XGBoost direction probability
    plt.figure(figsize=(4, 4))
    plt.bar(
        ["Down", "Up"],
        [
            1 - xgb_out["XGB_Up_Probability"],
            xgb_out["XGB_Up_Probability"]
        ]
    )
    plt.title("XGBoost Direction Probability")
    plt.ylim(0, 1)
    plt.show()


# ---------------- MAIN PIPELINE ----------------
def main():
    print("\n📈 INDIAN STOCK FORECAST SYSTEM\n")

    stock_input = input(
        "Enter NSE stock symbol (example: RELIANCE): "
    ).strip()

    stock = normalize_symbol(stock_input)   # e.g. SYMPHONY.NS
    clean_stock = stock.replace(".NS", "")  # e.g. SYMPHONY


    print(f"\n▶ Running forecast for {stock}...\n")

    # -------- 1. Fetch daily data --------
    success = run_pipeline([stock])

    if not success:
        print(f"❌ Data fetch failed for {stock}. Exiting pipeline.")
        return

    # -------- 2. Convert daily → weekly --------
    weekly = convert_daily_to_weekly(clean_stock)


    # -------- 3. SARIMA --------
    print("🔹 Running SARIMA...")
    sarima_out = sarima_range(stock)

    # -------- 4. XGBoost --------
    print("🔹 Running XGBoost...")
    xgb_out = run_xgboost(stock)

    # -------- 5. LSTM --------
    print("🔹 Running LSTM...")
    lstm_out = run_lstm(stock)

    # -------- 6. Fundamentals --------
    print("🔹 Running Fundamentals...")
    fund_out = run_fundamentals(stock)

    # -------- 7. Seasonality --------
    print("🔹 Running Seasonality...")
    season_out = run_seasonality(stock)

    # -------- 8. Decision Engine --------
    final = decision_engine(
        stock=stock,
        sarima_out=sarima_out,
        xgb_out=xgb_out,
        lstm_out=lstm_out,
        fund_out=fund_out,
        season_out=season_out
    )

    # -------- 9. RESULTS --------
    print("\n📊 FINAL RESULT\n")
    print(f"Stock: {stock}")
    print(f"SARIMA Trend Score      : {sarima_out['SARIMA_Trend_Score']}")
    print(f"LSTM Return Score       : {lstm_out['LSTM_Return_Score']}")
    print(f"XGBoost Up Probability  : {xgb_out['XGB_Up_Probability']}")
    print(f"Fundamental Score       : {fund_out['Fundamental_Combined_Score']}")
    print(f"Seasonality Score       : {season_out['Seasonality_Score']}")
    print("\n-----------------------------")
    print(f"FINAL DECISION : {final['Decision']}")
    print(f"CONFIDENCE     : {final['Confidence_%']}%")
    print("-----------------------------\n")

    # -------- 10. Visualizations --------
    plot_results(stock, weekly, sarima_out, xgb_out)


if __name__ == "__main__":
    main()
