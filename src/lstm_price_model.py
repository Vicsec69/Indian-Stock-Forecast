import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- PATHS ----------------
PROCESSED_DIR = r"C:\Users\shiva\Indian Stock Forecast\data\processed"

# ---------------- PARAMETERS ----------------
LOOKBACK = 20        # weeks used as input
EPOCHS = 30
BATCH_SIZE = 16


def prepare_lstm_data(series, lookback=LOOKBACK):
    """
    Convert price series into LSTM sequences
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def run_lstm(stock):
    """
    Train LSTM on weekly close prices
    and predict next-week return
    """
    path = os.path.join(PROCESSED_DIR, f"{stock}_weekly.csv")
    df = pd.read_csv(path)

    if "Close" not in df.columns or len(df) < LOOKBACK + 10:
        raise ValueError("Insufficient data for LSTM")

    prices = df["Close"].values.reshape(-1, 1)

    # ---------------- SCALING ----------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    # ---------------- SEQUENCES ----------------
    X, y = prepare_lstm_data(scaled)

    # Train / validation split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((-1, LOOKBACK, 1))
    X_val = X_val.reshape((-1, LOOKBACK, 1))

    # ---------------- MODEL ----------------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )

    # ---------------- PREDICTION ----------------
    last_seq = scaled[-LOOKBACK:].reshape((1, LOOKBACK, 1))
    next_price_scaled = model.predict(last_seq, verbose=0)[0][0]

    next_price = scaler.inverse_transform([[next_price_scaled]])[0][0]
    current_price = prices[-1][0]

    expected_return = (next_price - current_price) / current_price

    # Map return to [-1, 1] score
    return_score = float(np.tanh(expected_return * 5))

    return {
        "Stock": stock,
        "LSTM_Next_Price": round(float(next_price), 2),
        "LSTM_Expected_Return": round(float(expected_return), 4),
        "LSTM_Return_Score": round(return_score, 4)
    }
