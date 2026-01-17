import pandas as pd
import os

# Paths (Windows-safe)
RAW_DATA_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\raw"
PROCESSED_DATA_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\processed"


def convert_daily_to_weekly(stock):
    """
    Convert ONE stock's daily data to weekly data (Friday close)
    """
    input_file = f"{stock}.NS_daily.csv"
    input_path = os.path.join(RAW_DATA_DIR, input_file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Daily data not found for {stock}")

    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")

    weekly_df = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(
        PROCESSED_DATA_DIR, f"{stock}.NS_weekly.csv"
    )
    weekly_df.to_csv(output_path)

    return weekly_df


# ---------------- OPTIONAL BATCH MODE ----------------
def run_weekly_pipeline():
    """
    Batch processing (kept for backward compatibility)
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith("_daily.csv")]

    print(f"Found {len(files)} daily files")

    for file in files:
        try:
            stock = file.replace(".NS_daily.csv", "")
            convert_daily_to_weekly(stock)
            print(f"✅ Weekly data created: {stock}")

        except Exception as e:
            print(f"❌ Failed for {file}: {e}")


if __name__ == "__main__":
    run_weekly_pipeline()
