# =========================================================
# predict.py
# Loads saved ARIMA/XGBoost models + recent price history,
# builds the live feature row, and returns predictions.
# =========================================================

import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    "Modal_Price", "temp_mean", "rainfall_mm", "humidity",
    "price_lag_1d", "price_lag_7d", "price_lag_14d", "price_lag_30d",
    "price_7d_avg", "price_30d_avg",
    "rainfall_mm_7d_avg", "rainfall_mm_30d_avg",
    "temp_mean_7d_avg", "temp_mean_30d_avg",
    "volatility_7d", "price_change_1d",
    "month", "day_of_year",
]

HORIZONS = [7, 14, 30]

# -------------------------------------------------
# Cache loaded models + history in memory so we don't
# reload from disk on every single request
# -------------------------------------------------
_cache = {}


def _load_history(commodity):
    """Load the last ~60 days feature-complete history for a commodity."""
    key = f"history_{commodity}"
    if key not in _cache:
        path = os.path.join(BASE_DIR, "data", f"{commodity.lower()}_history.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Export recent history from your notebook first "
                f"(see save_history.py instructions)."
            )
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        df = df.set_index("date")
        _cache[key] = df
    return _cache[key]


def _load_xgb_model(commodity, horizon):
    key = f"xgb_{commodity}_{horizon}"
    if key not in _cache:
        path = os.path.join(BASE_DIR, "models", f"{commodity.lower()}_xgb_{horizon}d.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        _cache[key] = joblib.load(path)
    return _cache[key]


def _load_arima_model(commodity):
    key = f"arima_{commodity}"
    if key not in _cache:
        path = os.path.join(BASE_DIR, "models", f"{commodity.lower()}_arima.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        _cache[key] = joblib.load(path)
    return _cache[key]


def _build_latest_feature_row(history_df):
    """
    Build the single most-recent feature row needed for XGBoost prediction,
    using the same feature logic as the training notebook.
    """
    d = history_df.copy()

    d["price_lag_1d"] = d["Modal_Price"].shift(1)
    d["price_lag_7d"] = d["Modal_Price"].shift(7)
    d["price_lag_14d"] = d["Modal_Price"].shift(14)
    d["price_lag_30d"] = d["Modal_Price"].shift(30)

    d["price_7d_avg"] = d["Modal_Price"].rolling(7).mean()
    d["price_30d_avg"] = d["Modal_Price"].rolling(30).mean()

    if "rainfall_mm" in d.columns:
        d["rainfall_mm_7d_avg"] = d["rainfall_mm"].rolling(7).mean()
        d["rainfall_mm_30d_avg"] = d["rainfall_mm"].rolling(30).mean()
    if "temp_mean" in d.columns:
        d["temp_mean_7d_avg"] = d["temp_mean"].rolling(7).mean()
        d["temp_mean_30d_avg"] = d["temp_mean"].rolling(30).mean()

    d["volatility_7d"] = d["Modal_Price"].rolling(7).std()
    d["price_change_1d"] = d["Modal_Price"].pct_change(1)
    d["month"] = d.index.month
    d["day_of_year"] = d.index.dayofyear

    latest = d.iloc[[-1]]  # most recent row, as a 1-row DataFrame
    cols = [c for c in FEATURE_COLS if c in latest.columns]
    return latest[cols]


def predict_xgb(commodity, horizon):
    """Predict price `horizon` days ahead for a commodity using XGBoost."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")

    history_df = _load_history(commodity)
    model = _load_xgb_model(commodity, horizon)
    feature_row = _build_latest_feature_row(history_df)

    if feature_row.isna().any(axis=1).item():
        raise ValueError(
            "Not enough recent history to build all features "
            "(need at least 30 days of continuous data)."
        )

    prediction = model.predict(feature_row)[0]
    return float(prediction)


def predict_arima(commodity, steps):
    """Predict price `steps` days ahead using the ARIMA baseline model."""
    model = _load_arima_model(commodity)
    forecast = model.predict(n_periods=steps)
    return float(forecast[-1])
