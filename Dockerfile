
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

# Render assigns its own port via the PORT env var at runtime.
# Default to 8000 for local docker run, but respect $PORT when set.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}