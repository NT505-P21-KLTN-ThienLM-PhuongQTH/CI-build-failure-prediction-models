FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/* && rm -rf /app/wheels

COPY . .

CMD ["python", "consumer.py", "--mode", "consume"]