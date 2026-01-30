import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LOOK_BACK = 60

def preprocess_data(df):
    close_prices = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, LOOK_BACK, 1)
    y = np.array(y)

    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler, scaled

if __name__ == "__main__":
    df = pd.read_csv(RAW_DIR / "aapl.csv", index_col=0)

    X_train, y_train, X_test, y_test, scaler, scaled = preprocess_data(df)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "scaled.npy", scaled)

    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")

    print("[OK] Données prétraitées et sauvegardées")
