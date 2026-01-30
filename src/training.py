import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

if __name__ == "__main__":
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    cnn = build_cnn((X_train.shape[1], 1))
    cnn.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    cnn.save(MODEL_DIR / "cnn_model.h5")

    gru = build_gru((X_train.shape[1], 1))
    gru.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    gru.save(MODEL_DIR / "gru_model.h5")

    print("[OK] Modèles CNN et GRU entraînés")
