# Apple Stock Price Prediction — CNN & GRU Models

Machine learning project for **AAPL (Apple Inc.)** stock price forecasting using deep learning models — a 1D Convolutional Neural Network (CNN) and a Gated Recurrent Unit (GRU) — with automated data collection, preprocessing, evaluation, and email alerting.

---

## Architecture Overview

```
Yahoo Finance
     │
     ▼
DataCollector (yfinance)
     │
     ▼
DataCleaner (validation + normalization)
     │
     ▼
Preprocessing (MinMaxScaler · windowing · train/test split)
     │
     ├──────────────────────┐
     ▼                      ▼
CNN Model              GRU Model
(Conv1D × 2)           (GRU × 2 + Dropout)
     │                      │
     └──────────┬───────────┘
                ▼
           Evaluator (MSE · MAE · RMSE)
                │
                ▼
     Prediction Dashboard (Streamlit)
                │
                ▼
     Email Alerts (on threshold breach)
```

---

## Features

- **Automated Data Collection** — Downloads AAPL history from Yahoo Finance with metadata tracking
- **CNN Architecture** — Two-stage Conv1D model for pattern extraction from price sequences
- **GRU Architecture** — Recurrent model with dropout for temporal dependency learning
- **Full Evaluation Suite** — MSE, MAE, and RMSE metrics on held-out test data
- **Streamlit Dashboard** — Interactive visualization of predictions vs. actual prices
- **Email Notifications** — Automated alerts when predicted price crosses configured thresholds
- **Modular Design** — Clean separation between data, preprocessing, models, and evaluation

---

## Project Structure

```
apple_stock/
├── src/
│   ├── data/
│   │   ├── collector.py         # yfinance download + metadata JSON
│   │   └── cleaner.py           # Data validation & normalization
│   ├── preprocessing.py         # MinMaxScaler, windowing, train/test split
│   ├── models/
│   │   ├── cnn_model.py         # CNN1DModel class (Conv1D + MaxPool + Dense)
│   │   └── trainer.py           # Training loop with callbacks
│   ├── training.py              # build_cnn() and build_gru() factory functions
│   ├── prediction.py            # End-to-end prediction pipeline
│   ├── prediction/
│   │   └── predictor.py         # Predictor utility class
│   ├── evaluation.py            # Metrics computation (MSE, MAE, RMSE)
│   ├── ingestion.py             # Full ingestion workflow orchestrator
│   └── notification/
│       └── email_sender.py      # SMTP email alert sender
├── requirements.txt
└── README.md
```

---

## Model Architectures

### CNN Model (`cnn_model.py`)

```
Input (sequence_length, features)
    │
    ├── Conv1D (64 filters, kernel=3, ReLU)
    ├── MaxPooling1D (pool=2)
    ├── Dropout (0.2)
    │
    ├── Conv1D (128 filters, kernel=3, ReLU)
    ├── MaxPooling1D (pool=2)
    ├── Dropout (0.2)
    │
    ├── Flatten
    ├── Dense (50, ReLU)
    └── Dense (1) ── predicted price
```

**Optimizer:** Adam | **Loss:** MSE

### GRU Model (`training.py`)

```
Input (sequence_length, features)
    │
    ├── GRU (50 units, return_sequences=True)
    ├── Dropout (0.2)
    ├── GRU (50 units)
    └── Dense (1) ── predicted price
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Collection | `yfinance`, `pandas` |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Classical ML | scikit-learn |
| Data Processing | NumPy, pandas |
| Visualization | Matplotlib, Pillow |
| Dashboard | Streamlit |
| Notifications | smtplib (SMTP) |
| Language | Python 3.10+ |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect AAPL data

```python
from src.data.collector import DataCollector

collector = DataCollector(ticker="AAPL")
collector.download_stock_data(period="4y")
```

Data is saved to `data/raw/AAPL_YYYYMMDD.csv` with a companion metadata JSON file.

### 3. Run the full pipeline

```bash
python src/ingestion.py
```

This executes the complete workflow: collect → clean → preprocess → train → evaluate → save model.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## Preprocessing

1. **Cleaning** — Drop null rows, validate OHLCV columns, remove outliers
2. **Scaling** — MinMaxScaler normalizes all features to [0, 1]
3. **Windowing** — Sliding window of `n` days → predict next close price
4. **Split** — 80% training / 20% test (chronological, no shuffle)

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error (primary metric) |

---

## Email Alerts

Configure SMTP credentials in `.env` to receive alerts when the predicted price diverges from the actual price by more than a configured percentage threshold.

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your_app_password
ALERT_THRESHOLD=0.05
```
