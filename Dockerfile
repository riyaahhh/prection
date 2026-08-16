# =========================================================
# Dockerfile for KrishiDrishti Forecasting API
# Build:  docker build -t krishi-forecast-api .
# Run:    docker run -p 8000:8000 krishi-forecast-api
# =========================================================

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by pmdarima/xgboost build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, models, and data
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
