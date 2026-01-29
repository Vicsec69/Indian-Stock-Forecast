import yfinance as yf
import os
import time

# Absolute path to data/raw folder (Windows-safe)
DATA_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\raw"

# Default stocks (used if user gives no input)
DEFAULT_STOCK_LIST = [
    "TCS"
]

def normalize_symbol(symbol: str) -> str:
    """
    Ensure NSE format (.NS)
    """
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    return symbol


def get_user_stock_list():
    """
    Takes user input from terminal.
    Returns list of stock symbols.
    """
    user_input = input(
        "\nEnter stock symbols (comma separated) or press Enter to use default:\n"
        "Example: RELIANCE,TCS,INFY\n> "
    ).strip()

    if not user_input:
        print("⚠️ No input given. Using default stock list.")
        return DEFAULT_STOCK_LIST

    stocks = [normalize_symbol(s) for s in user_input.split(",")]
    return stocks


def fetch_stock_data(symbol, period="10y"):
    """
    Fetch daily stock data from Yahoo Finance for NSE stocks
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    return df


def save_to_csv(df, symbol):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{symbol}_daily.csv"
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    print(f"✅ Saved: {path}")


def run_pipeline(stock_list=None):
    """
    Fetch daily data for given stocks.
    Returns True if ALL stocks succeed, else False.
    """
    if stock_list is None:
        stock_list = get_user_stock_list()

    all_success = True

    for stock in stock_list:
        try:
            print(f"\n📥 Fetching data for {stock}...")
            data = fetch_stock_data(stock)
            save_to_csv(data, stock)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Failed for {stock}: {e}")
            all_success = False

    return all_success



if __name__ == "__main__":
    run_pipeline()
