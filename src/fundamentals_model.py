import yfinance as yf
import pandas as pd
import numpy as np


# ---------------- SAFE HELPERS ---------------- #
def safe_get(df, row):
    try:
        return df.loc[row].iloc[0]
    except Exception:
        return np.nan


# ---------------- FUNDAMENTALS ---------------- #
def get_fundamentals(stock):
    """
    Fetch fundamental and valuation metrics for ONE stock
    """
    symbol = stock + ".NS"
    ticker = yf.Ticker(symbol)

    financials = ticker.financials
    balance = ticker.balance_sheet
    info = ticker.info or {}

    revenue = safe_get(financials, "Total Revenue")
    profit = safe_get(financials, "Net Income")
    equity = safe_get(balance, "Total Stockholder Equity")
    debt = safe_get(balance, "Total Debt")

    # Growth
    revenue_growth = (
        financials.loc["Total Revenue"].pct_change().mean()
        if financials is not None and "Total Revenue" in financials.index
        else np.nan
    )

    profit_growth = (
        financials.loc["Net Income"].pct_change().mean()
        if financials is not None and "Net Income" in financials.index
        else np.nan
    )

    roe = profit / equity if equity not in [0, np.nan] else np.nan
    debt_equity = debt / equity if equity not in [0, np.nan] else np.nan

    operating_margin = info.get("operatingMargins", np.nan)
    pe = info.get("trailingPE", np.nan)
    pb = info.get("priceToBook", np.nan)

    return {
        "Revenue_Growth": revenue_growth,
        "Profit_Growth": profit_growth,
        "ROE": roe,
        "Debt_Equity": debt_equity,
        "Operating_Margin": operating_margin,
        "PE": pe,
        "PB": pb
    }


# ---------------- SCORING ---------------- #
def fundamental_score(d):
    score = 0
    if pd.notna(d["Revenue_Growth"]) and d["Revenue_Growth"] > 0.08:
        score += 1
    if pd.notna(d["Profit_Growth"]) and d["Profit_Growth"] > 0.08:
        score += 1
    if pd.notna(d["ROE"]) and d["ROE"] > 0.15:
        score += 1
    if pd.notna(d["Debt_Equity"]) and d["Debt_Equity"] < 0.7:
        score += 1
    if pd.notna(d["Operating_Margin"]) and d["Operating_Margin"] > 0.15:
        score += 1
    return score / 5


def valuation_score(d):
    score = 0
    if pd.notna(d["PE"]) and d["PE"] < 25:
        score += 1
    if pd.notna(d["PB"]) and d["PB"] < 4:
        score += 1
    return score / 2


# ---------------- MAIN CALL ---------------- #
def run_fundamentals(stock):
    """
    Final callable used by main.py
    """
    data = get_fundamentals(stock)

    fund_score = fundamental_score(data)
    val_score = valuation_score(data)

    # Normalize to [-1, 1]
    combined = (fund_score * 0.6 + val_score * 0.4)
    combined_score = (combined - 0.5) * 2

    return {
        "Stock": stock,
        "Fundamental_Score": round(fund_score, 3),
        "Valuation_Score": round(val_score, 3),
        "Fundamental_Combined_Score": round(combined_score, 3)
    }
