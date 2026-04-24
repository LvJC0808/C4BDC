# Use Docker Hub directly. If CN network is slow, configure daemon registry-mirrors
# in /etc/docker/daemon.json (e.g. docker.m.daocloud.io) or HTTP_PROXY in
# /etc/systemd/system/docker.service.d/http-proxy.conf
FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library (GitHub mirror is more reliable from CN than sourceforge)
RUN (wget --tries=3 --timeout=60 https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz \
     || wget --tries=3 --timeout=60 http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz) && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv (requires Docker daemon network access to ghcr.io)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock requirements-submission.txt ./

# Install dependencies using the EXACT frozen versions from the golden 5090 .venv
# (pyproject.toml/uv.lock kept for local dev; requirements-submission.txt is SSOT for image).
ENV UV_HTTP_TIMEOUT=600
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r requirements-submission.txt

# Copy the application code
COPY . .

# Keep a read-only backup of bundled data so that, if the judges mount their own
# data/ (with only train.csv/test.csv) over /app/data, init.sh can re-materialize
# auxiliary CSVs (industry_map, hs300_history, csi300_index, stock_basic, trade_calendar)
# and merge train.csv + test.csv into stock_data.csv.
RUN cp -r /app/data /app/data_bundled

# Ensure runtime directories exist
RUN mkdir -p /app/model /app/output /app/temp

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Keep container running idle; execute train/predict manually via docker exec.
CMD ["sleep", "infinity"]
