FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install official Google Antigravity CLI (agy)
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash -s -- --dir /usr/local/bin \
    && agy --version

# Cache runtime dependencies separately so code-only rebuilds finish in ~2s
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" \
    "hatchling" \
    "fastapi>=0.115.0" \
    "psycopg[binary,pool]>=3.2.0" \
    "pydantic>=2.9.0" \
    "pydantic-settings>=2.6.0" \
    "uvicorn[standard]>=0.32.0"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agy_gateway ./agy_gateway

RUN pip install --no-cache-dir --no-deps --no-build-isolation .

EXPOSE 8787
CMD ["agy-gateway", "serve", "--host", "0.0.0.0", "--port", "8787"]
