FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY config/ config/
COPY models/ models/
COPY src/ src/
COPY scripts/ scripts/
COPY data.zip .

RUN chmod +x scripts/run_xMalconv

ENV PYTHONPATH="/app:/app/models/MalConv2-main"

EXPOSE 8501

ENTRYPOINT ["scripts/run_xMalconv"]
