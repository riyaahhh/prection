# =========================================================
# app.py
# FastAPI wrapper exposing the price forecasting model
# Run locally with: uvicorn app:app --reload
# =========================================================

from fastapi import FastAPI, HTTPException
from predict import predict_xgb, predict_arima, HORIZONS

app = FastAPI(title="KrishiDrishti Price Forecast API")

VALID_COMMODITIES = ["onion", "potato"]


@app.get("/")
def root():
    return {"status": "ok", "message": "KrishiDrishti forecasting API is running"}


@app.get("/predict/{commodity}/{horizon}d")
def predict(commodity: str, horizon: int):
    commodity = commodity.lower()

    if commodity not in VALID_COMMODITIES:
        raise HTTPException(status_code=400, detail=f"commodity must be one of {VALID_COMMODITIES}")

    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {HORIZONS}")

    try:
        prediction = predict_xgb(commodity, horizon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "commodity": commodity.capitalize(),
        "horizon": f"{horizon}d",
        "predicted_price": round(prediction, 2),
        "model": "xgboost",
    }


@app.get("/predict/{commodity}/arima")
def predict_baseline(commodity: str, steps: int = 7):
    commodity = commodity.lower()

    if commodity not in VALID_COMMODITIES:
        raise HTTPException(status_code=400, detail=f"commodity must be one of {VALID_COMMODITIES}")

    try:
        prediction = predict_arima(commodity, steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "commodity": commodity.capitalize(),
        "steps": steps,
        "predicted_price": round(prediction, 2),
        "model": "arima",
    }
