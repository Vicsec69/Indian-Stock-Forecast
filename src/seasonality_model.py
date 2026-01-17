import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

# ---------------- PATHS ----------------
PROCESSED_DATA_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\processed"
OUTPUT_PLOT_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\seasonality_plots"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)


def run_seasonality(stock, save_plot=True):
    """
    Compute yearly seasonality strength for ONE stock.
    Returns normalized Seasonality_Score in [-1, 1].
    """

    file_path = os.path.join(
        PROCESSED_DATA_DIR, f"{stock}_weekly.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weekly data not found for {stock}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    close = df["Close"]

    # Need at least 2 years of weekly data
    if len(close) < 104:
        return {
            "Stock": stock,
            "Yearly_ACF": 0.0,
            "Seasonality_Score": 0.0
        }

    # -------- Seasonal Decomposition (52 weeks = yearly) --------
    decomposition = seasonal_decompose(
        close, model="additive", period=52
    )

    # -------- Autocorrelation at yearly lag --------
    acf_vals = acf(close, nlags=52)
    yearly_acf = float(acf_vals[52])

    # -------- Normalize to [-1, 1] --------
    seasonality_score = float(
        np.clip(yearly_acf, -0.6, 0.6) / 0.6
    )

    # -------- Save plot (optional) --------
    if save_plot:
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        plot_path = os.path.join(
            OUTPUT_PLOT_DIR, f"{stock}_seasonality.png"
        )
        plt.savefig(plot_path)
        plt.close()

    return {
        "Stock": stock,
        "Yearly_ACF": round(yearly_acf, 3),
        "Seasonality_Score": round(seasonality_score, 3)
    }


# ---------------- OPTIONAL BATCH MODE ----------------
def run_seasonality_batch():
    """
    Optional research mode: ranks all stocks by seasonality strength
    """
    results = []

    for file in os.listdir(PROCESSED_DATA_DIR):
        if not file.endswith("_weekly.csv"):
            continue

        stock = file.replace("_weekly.csv", "")

        try:
            out = run_seasonality(stock, save_plot=False)
            results.append((stock, out["Yearly_ACF"]))
            print(f"{stock}: Yearly ACF = {out['Yearly_ACF']:.2f}")
        except Exception as e:
            print(f"❌ {stock} failed: {e}")

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    df = pd.DataFrame(results, columns=["Stock", "Yearly_ACF"])
    df.to_csv(
        r"C:\Users\shiva\Indian Stock Forecast\data\seasonality_ranking.csv",
        index=False
    )

    print("\nTop Seasonal Stocks:")
    print(df.head(10))


if __name__ == "__main__":
    run_seasonality_batch()
