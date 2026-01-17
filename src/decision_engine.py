def decision_engine(
    stock: str,
    sarima_out: dict,
    xgb_out: dict,
    lstm_out: dict,
    fund_out: dict,
    season_out: dict
):
    """
    Combines SARIMA, XGBoost, LSTM, Fundamentals, and Seasonality
    to produce final BUY / SELL / HOLD decision with confidence
    """

    # ---------------- WEIGHTS ----------------
    W_SARIMA = 0.22   # trend & range
    W_XGB    = 0.28   # direction probability
    W_LSTM   = 0.22   # price magnitude
    W_FUND   = 0.18   # fundamentals & valuation
    W_SEASON = 0.10   # seasonality (supporting only)

    # ---------------- SCORES ----------------
    sarima_score = sarima_out["SARIMA_Trend_Score"]              # [-1, 1]
    xgb_score    = xgb_out["XGB_Direction_Score"]                # [-1, 1]
    lstm_score   = lstm_out["LSTM_Return_Score"]                  # [-1, 1]
    fund_score   = fund_out["Fundamental_Combined_Score"]        # [-1, 1]
    season_score = season_out["Seasonality_Score"]               # [-1, 1]

    # ---------------- FINAL SCORE ----------------
    final_score = (
        W_SARIMA * sarima_score +
        W_XGB    * xgb_score +
        W_LSTM   * lstm_score +
        W_FUND   * fund_score +
        W_SEASON * season_score
    )

    # Confidence as absolute strength
    confidence = round(min(abs(final_score), 1.0) * 100, 2)

    # ---------------- DECISION ----------------
    if final_score >= 0.35:
        decision = "BUY"
    elif final_score <= -0.35:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "Stock": stock,
        "Final_Score": round(final_score, 3),
        "Confidence_%": confidence,
        "Decision": decision
    }
