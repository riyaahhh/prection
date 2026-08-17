# =========================================================
# predict.py
# Loads saved ARIMA/XGBoost models + recent price history,
# builds the live feature row, and returns predictions.
# =========================================================

import pandas as pd
import numpy as np
import joblib
import os
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATION_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "mandi_sample.csv"
)

_location_cache = None
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
    Use the already-preprocessed latest feature row from the history CSV.

    The history CSVs are generated from the same preprocessing pipeline
    used during model training, so the engineered features should not be
    recalculated here.
    """
    d = history_df.copy()

    # Ensure calendar features are available.
    d["month"] = d.index.month
    d["day_of_year"] = d.index.dayofyear

    # Take the most recent fully preprocessed row.
    latest = d.iloc[[-1]]

    # Keep only features expected by the trained XGBoost model.
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

# =========================================================
# Forecast + Risk + SHAP Reasoning + Buffer Stock
# =========================================================

FEATURE_CATEGORIES = {
    "Price momentum": [
        "Modal_Price",
        "price_lag_1d",
        "price_lag_7d",
        "price_lag_14d",
        "price_lag_30d",
        "price_7d_avg",
        "price_30d_avg",
    ],

    "Supply pressure (proxy)": [
        "volatility_7d",
        "price_change_1d",
    ],

    "Rainfall": [
        "rainfall_mm",
        "rainfall_mm_7d_avg",
        "rainfall_mm_30d_avg",
    ],

    "Temperature & humidity": [
        "temp_mean",
        "temp_mean_7d_avg",
        "temp_mean_30d_avg",
        "humidity",
    ],

    "Seasonality": [
        "month",
        "day_of_year",
    ],
}


CATEGORY_NOTES = {
    "Supply pressure (proxy)": (
        "Estimated from price volatility, since live market arrivals data "
        "wasn't available — price already reacts to supply shocks."
    )
}


UNAVAILABLE_CATEGORIES = [
    "Transport risk"
]


def explain_prediction(model, X_row, current_price, predicted_price):
    """
    Generate SHAP-based reasoning exactly like the notebook.

    SHAP values are grouped into human-readable categories:
    Price momentum, Rainfall, Seasonality, etc.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_row)[0]

    feature_shap = dict(
        zip(X_row.columns, shap_values)
    )

    total_abs_impact = float(
        np.sum(np.abs(shap_values))
    )

    category_results = []

    for category, features in FEATURE_CATEGORIES.items():

        category_shap_sum = sum(
            abs(feature_shap.get(feature, 0))
            for feature in features
        )

        if total_abs_impact > 0:
            impact_pct = (
                category_shap_sum
                / total_abs_impact
                * 100
            )
        else:
            impact_pct = 0

        entry = {
            "category": category,
            "impact_pct": round(
                float(impact_pct),
                1
            )
        }

        if category in CATEGORY_NOTES:
            entry["note"] = CATEGORY_NOTES[category]

        category_results.append(entry)

    # Transport risk has no available data/proxy
    for category in UNAVAILABLE_CATEGORIES:

        category_results.append({
            "category": category,
            "impact_pct": None,
            "note": (
                "Not available — no data source or reliable proxy "
                "for this in the current dataset"
            )
        })

    # Highest-impact factors first.
    # Unavailable factors remain at the bottom.
    category_results.sort(
        key=lambda x: (
            x["impact_pct"] is None,
            -(x["impact_pct"] or 0)
        )
    )

    # Calculate actual price movement
    pct_change = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    move_word = "rise" if pct_change > 0 else "fall"

    top_category = next(
        (
            c["category"]
            for c in category_results
            if c["impact_pct"] is not None
        ),
        "recent trends"
    )

    net_summary = (
        f"Price is forecast to {move_word} "
        f"{abs(pct_change):.1f}% "
        f"(₹{current_price:.0f} → ₹{predicted_price:.0f}), "
        f"most strongly influenced by "
        f"{top_category.lower()}."
    )

    return {
        "net_summary": net_summary,
        "factor_breakdown": category_results
    }


def assess_risk(
    current_price,
    forecast_7d,
    forecast_14d,
    forecast_30d,
    volatility
):
    """
    Same rule-based risk logic as the notebook.
    """

    pct_change_7d = (
        (forecast_7d - current_price)
        / current_price
    ) * 100

    pct_change_30d = (
        (forecast_30d - current_price)
        / current_price
    ) * 100

    high_risk = (
        (
            abs(pct_change_7d) > 15
            or abs(pct_change_30d) > 25
        )
        and volatility > 50
    )

    watch = (
        abs(pct_change_7d) > 8
        or abs(pct_change_30d) > 15
    )

    if high_risk:
        level = "High Risk"
    elif watch:
        level = "Watch"
    else:
        level = "Stable"

    return (
        level,
        pct_change_7d,
        pct_change_30d
    )


def buffer_stock_recommendation(
    risk_level,
    pct_change_7d,
    commodity
):
    """
    Same illustrative buffer-stock logic as the notebook.
    """

    ASSUMED_DAILY_ARRIVALS = {
        "Onion": 500,
        "Potato": 600
    }

    baseline_arrivals = ASSUMED_DAILY_ARRIVALS.get(
        commodity,
        500
    )

    INTERVENTION_FACTOR = 0.5

    if (
        risk_level == "High Risk"
        and pct_change_7d > 0
    ):

        qty = round(
            baseline_arrivals
            * (pct_change_7d / 100)
            * INTERVENTION_FACTOR
        )

        return (
            f"Recommend releasing approximately "
            f"{qty} quintals of buffer stock over the next "
            f"7 days to help offset the forecasted "
            f"{pct_change_7d:.1f}% price rise "
            f"(estimate based on an assumed average arrival "
            f"volume of {baseline_arrivals} quintals/day — "
            f"illustrative, not live government data)."
        )

    elif (
        risk_level == "High Risk"
        and pct_change_7d < 0
    ):

        qty = round(
            baseline_arrivals
            * (abs(pct_change_7d) / 100)
            * INTERVENTION_FACTOR
        )

        return (
            f"Recommend procuring approximately "
            f"{qty} quintals to support farmer prices "
            f"given the forecasted "
            f"{abs(pct_change_7d):.1f}% price drop "
            f"(estimate based on an assumed average arrival "
            f"volume of {baseline_arrivals} quintals/day — "
            f"illustrative, not live government data)."
        )

    elif risk_level == "Watch":

        return (
            "Monitor closely over the next week — "
            "no stock intervention recommended yet."
        )

    else:

        return (
            "Market is stable. "
            "No buffer stock intervention needed."
        )


def get_full_forecast(commodity):
    """
    Return the complete forecast output.

    Uses the SAME:
    - XGBoost models
    - SHAP explanation
    - risk rules
    - buffer-stock logic

    from the Colab notebook.
    """

    commodity = commodity.lower()

    if commodity not in ["onion", "potato"]:
        raise ValueError(
            "commodity must be one of ['onion', 'potato']"
        )

    history_df = _load_history(commodity)

    # -------------------------------------------------
    # Use the latest complete feature row
    # -------------------------------------------------

    X_row = _build_latest_feature_row(history_df)

    if X_row.empty:
        raise ValueError(
            "No valid feature row available for prediction."
        )

    if X_row.isna().any(axis=1).item():
        raise ValueError(
            "Latest feature row contains missing values."
        )

    latest_date = X_row.index[0]

    current_price = float(
        X_row["Modal_Price"].iloc[0]
    )

    volatility = float(
        X_row["volatility_7d"].iloc[0]
    )

    # -------------------------------------------------
    # Generate 7 / 14 / 30 day predictions
    # -------------------------------------------------

    forecasts = {}
    explanations = {}

    for horizon in [7, 14, 30]:

        model = _load_xgb_model(
            commodity,
            horizon
        )

        prediction = float(
            model.predict(X_row)[0]
        )

        forecasts[horizon] = round(
            prediction,
            2
        )

        explanations[horizon] = explain_prediction(
            model,
            X_row,
            current_price,
            forecasts[horizon]
        )

    # -------------------------------------------------
    # Risk
    # -------------------------------------------------

    (
        risk_level,
        pct_change_7d,
        pct_change_30d
    ) = assess_risk(
        current_price,
        forecasts[7],
        forecasts[14],
        forecasts[30],
        volatility
    )

    # -------------------------------------------------
    # Buffer stock recommendation
    # -------------------------------------------------

    buffer_action = buffer_stock_recommendation(
        risk_level,
        pct_change_7d,
        commodity.capitalize()
    )

    # -------------------------------------------------
    # FINAL JSON
    # -------------------------------------------------

    return {
        "commodity": commodity.capitalize(),

        "as_of_date": str(
            latest_date.date()
        ),

        "current_price": round(
            current_price,
            2
        ),

        "forecast_7d": forecasts[7],

        "forecast_14d": forecasts[14],

        "forecast_30d": forecasts[30],

        "pct_change_7d": round(
            pct_change_7d,
            2
        ),

        "pct_change_30d": round(
            pct_change_30d,
            2
        ),

        "risk_level": risk_level,

        "explanation_7d": explanations[7],

        "explanation_30d": explanations[30],

        "buffer_stock_recommendation": buffer_action
    }
# =========================================================
# LOCATION SUPPORT
# =========================================================

def _load_locations():
    """
    Load valid State -> District -> Market combinations
    from the original mandi CSV.
    """
    global _location_cache

    if _location_cache is None:
        if not os.path.exists(LOCATION_FILE):
            raise FileNotFoundError(
                f"Missing location dataset: {LOCATION_FILE}"
            )

        df = pd.read_csv(LOCATION_FILE)

        required = ["STATE", "district", "Market Name", "Commodity"]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Location dataset is missing columns: {missing}"
            )

        df = df.dropna(
            subset=[
                "STATE",
                "district",
                "Market Name",
                "Commodity"
            ]
        )

        # Normalize strings
        for col in [
            "STATE",
            "district",
            "Market Name",
            "Commodity"
        ]:
            df[col] = df[col].astype(str).str.strip()

        _location_cache = df

    return _location_cache


def get_locations(commodity=None):
    """
    Return valid locations available in the Mandi CSV.

    If commodity is supplied, only locations for that
    commodity are returned.
    """

    df = _load_locations()

    if commodity:
        commodity = commodity.lower()

        df = df[
            df["Commodity"].str.lower() == commodity
        ]

    locations = []

    for _, row in (
        df[
            ["STATE", "district", "Market Name"]
        ]
        .drop_duplicates()
        .iterrows()
    ):
        locations.append({
            "state": row["STATE"],
            "district": row["district"],
            "market": row["Market Name"]
        })

    return locations


def validate_location(
    commodity,
    state,
    district,
    market
):
    """
    Check whether the selected location exists in the
    original Mandi CSV for the selected commodity.
    """

    df = _load_locations()

    filtered = df[
        (df["Commodity"].str.lower() == commodity.lower())
        &
        (df["STATE"].str.lower() == state.lower())
        &
        (df["district"].str.lower() == district.lower())
        &
        (df["Market Name"].str.lower() == market.lower())
    ]

    return not filtered.empty


def get_location_forecast(
    commodity,
    state,
    district,
    market
):
    """
    Return the existing complete forecast together with
    the validated selected location.

    IMPORTANT:
    The current XGBoost models are trained on the exported
    historical series and are NOT retrained per market.
    Therefore this endpoint does not claim to produce a
    separately trained market-specific model.
    """

    commodity = commodity.lower()

    if commodity not in ["onion", "potato"]:
        raise ValueError(
            "commodity must be one of ['onion', 'potato']"
        )

    valid = validate_location(
        commodity,
        state,
        district,
        market
    )

    if not valid:
        raise ValueError(
            f"Location not available for {commodity}: "
            f"{state} → {district} → {market}"
        )

    # Use your already-working complete forecast.
    result = get_full_forecast(commodity)

    # Attach the selected location.
    result["location"] = {
        "state": state,
        "district": district,
        "market": market
    }

    return result