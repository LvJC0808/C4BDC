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
COPY pyproject.toml uv.lock ./

# Install dependencies (torch removed from uv.lock — see pyproject.toml)
# Use Aliyun PyPI mirror + 10 min timeout for large wheels under CN network.
ENV UV_HTTP_TIMEOUT=600
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN uv sync --frozen

# Copy the application code
COPY . .

# Ensure runtime directories exist
RUN mkdir -p /app/model /app/output /app/temp

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Keep container running idle; execute train/predict manually via docker exec.
CMD ["sleep", "infinity"]
