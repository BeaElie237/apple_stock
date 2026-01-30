import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def evaluate(model, X_test, y_test, scaler):
    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    return mean_squared_error(y_test, pred)

if __name__ == "__main__":
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    scaler = joblib.load(DATA_DIR / "scaler.pkl")

    cnn = load_model(MODEL_DIR / "cnn_model.h5")
    gru = load_model(MODEL_DIR / "gru_model.h5")

    cnn_mse = evaluate(cnn, X_test, y_test, scaler)
    gru_mse = evaluate(gru, X_test, y_test, scaler)

    best_model = "gru_model.h5" if gru_mse < cnn_mse else "cnn_model.h5"

    Path(MODEL_DIR / "best_model.h5").write_bytes(
        Path(MODEL_DIR / best_model).read_bytes()
    )

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"cnn_mse": cnn_mse, "gru_mse": gru_mse}, f, indent=2)

    print(f"[OK] Meilleur modèle : {best_model}")
