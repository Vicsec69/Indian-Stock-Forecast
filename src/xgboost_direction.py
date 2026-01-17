import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

PROCESSED_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\processed"
OUT_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\direction"
os.makedirs(OUT_DIR, exist_ok=True)


def create_features(df):
    """
    Feature engineering for direction prediction
    """
    df = df.copy()

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_2"] = df["Close"].pct_change(2)
    df["ma_12"] = df["Close"].rolling(12).mean()
    df["ma_26"] = df["Close"].rolling(26).mean()
    df["vol"] = df["Close"].pct_change().rolling(12).std()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()


def run_xgboost(stock):
    """
    Train XGBoost and return direction probability
    """
    path = os.path.join(PROCESSED_DIR, f"{stock}_weekly.csv")
    df = pd.read_csv(path)

    if "Close" not in df.columns or len(df) < 50:
        raise ValueError("Insufficient data for XGBoost")

    df = create_features(df)

    X = df[["ret_1", "ret_2", "ma_12", "ma_26", "vol"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    prob_up = float(model.predict_proba(X.tail(1))[0][1])

    return {
        "Stock": stock,
        "XGB_Up_Probability": round(prob_up, 3),
        "XGB_Direction_Score": round((prob_up - 0.5) * 2, 3)  # maps to [-1, 1]
    }


def run():
    """
    Batch run (optional – keeps your old functionality)
    """
    rows = []

    for f in os.listdir(PROCESSED_DIR):
        if not f.endswith("_weekly.csv"):
            continue

        stock = f.replace("_weekly.csv", "")

        try:
            print(f"📊 XGBoost processing: {stock}")
            rows.append(run_xgboost(stock))
        except Exception as e:
            print(f"❌ XGBoost failed for {stock}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "direction_scores.csv"),
            index=False
        )
        print("✅ XGBoost direction scores saved")


if __name__ == "__main__":
    run()
