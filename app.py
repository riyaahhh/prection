# =========================================================
# app.py
# FastAPI wrapper exposing the price forecasting model
# Run locally with: uvicorn app:app --reload
# =========================================================

from fastapi import FastAPI, HTTPException
from predict import (
    predict_xgb,
    predict_arima,
    get_full_forecast,
    get_locations,
    get_location_forecast,
    HORIZONS
)

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
@app.get("/forecast/{commodity}")
def forecast(commodity: str):
    commodity = commodity.lower()

    if commodity not in VALID_COMMODITIES:
        raise HTTPException(
            status_code=400,
            detail=f"commodity must be one of {VALID_COMMODITIES}"
        )

    try:
        return get_full_forecast(commodity)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/locations")
def locations(commodity: str | None = None):
    """
    Return locations available in the Mandi CSV.

    Optional:
    ?commodity=onion
    ?commodity=potato
    """

    if commodity is not None:
        commodity = commodity.lower()

        if commodity not in VALID_COMMODITIES:
            raise HTTPException(
                status_code=400,
                detail=f"commodity must be one of {VALID_COMMODITIES}"
            )

    try:
        return {
            "commodity": (
                commodity.capitalize()
                if commodity
                else "all"
            ),
            "locations": get_locations(commodity)
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/reasoning/{commodity}")
def reasoning(commodity: str):
    commodity = commodity.lower()

    if commodity not in VALID_COMMODITIES:
        raise HTTPException(
            status_code=400,
            detail=f"commodity must be one of {VALID_COMMODITIES}"
        )

    try:
        result = get_full_forecast(commodity)

        return {
            "commodity": result["commodity"],
            "explanation_7d": result["explanation_7d"],
            "explanation_30d": result["explanation_30d"],
            "buffer_stock_recommendation": result[
                "buffer_stock_recommendation"
            ]
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )    

@app.get("/forecast/{commodity}/location")
def location_forecast(
    commodity: str,
    state: str,
    district: str,
    market: str
):
    commodity = commodity.lower()

    if commodity not in VALID_COMMODITIES:
        raise HTTPException(
            status_code=400,
            detail=f"commodity must be one of {VALID_COMMODITIES}"
        )

    try:
        return get_location_forecast(
            commodity=commodity,
            state=state,
            district=district,
            market=market
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )