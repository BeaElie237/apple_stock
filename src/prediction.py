import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

DAYS = 21

if __name__ == "__main__":
    model = load_model(MODEL_DIR / "best_model.h5")
    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    scaled = np.load(DATA_DIR / "scaled.npy")

    last_seq = scaled[-60:, 0]
    preds = []

    for _ in range(DAYS):
        p = model.predict(last_seq.reshape(1, -1, 1))[0, 0]
        preds.append(p)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    df = pd.DataFrame({"day": range(1, DAYS + 1), "prediction": preds})
    df.to_csv("predictions.csv", index=False)

    print("[OK] Prédictions générées")
