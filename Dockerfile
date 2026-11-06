# ==============================================================================
# Stage 1: Build stage
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for deterministic wheel build and dependency resolution
RUN pip install --no-cache-dir uv==0.5.29

# Copy project files necessary for building the wheel
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/

# Build standalone wheel and install into isolated virtual environment with service extras
RUN uv build --wheel && \
    uv venv /opt/venv && \
    WHEEL=$(ls dist/*.whl) && \
    uv pip install --python /opt/venv "${WHEEL}[service]"

# ==============================================================================
# Stage 2: Runtime stage
# ==============================================================================
FROM python:3.12-slim AS runtime

# Security hardening: Create non-root system user and group
RUN groupadd -g 10001 reversallab && \
    useradd -u 10001 -g reversallab -s /bin/false -m -d /home/reversallab reversallab

# Copy isolated virtual environment containing installed wheel from builder
COPY --from=builder --chown=reversallab:reversallab /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER reversallab:reversallab
WORKDIR /home/reversallab

EXPOSE 8000

# Health check using standard library urllib (no curl dependency required)
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production startup command using uvicorn module invocation
CMD ["python", "-m", "uvicorn", "reversallab.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
