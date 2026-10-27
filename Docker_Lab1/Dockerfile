# Multi-stage build: one Dockerfile, three images.
#   trainer   (default) - the original lab's training job
#   dashboard           - Streamlit app that reads the trainer's artifacts
#   tests               - dashboard image plus pytest, runs the test suite
# `docker build .` builds the last stage (trainer), so the original lab's
# commands still work; docker-compose.yml picks each stage with `target:`.

# ---------------------------------------------------------------- base
# Slim Python image instead of the full python:3.10 image used in the original lab
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="breast-cancer-trainer" \
      org.opencontainers.image.description="Docker Lab 1: train a breast cancer classifier in a container"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies before copying the code, so this layer stays cached
# when only the Python files change
COPY src/requirements.txt .
RUN pip install -r requirements.txt

# Non-root user that owns the output directory (also the mount point of the shared volume)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/output \
    && chown appuser:appuser /app/output

# Training settings; override any of them with `docker run -e NAME=value`
ENV MODEL_TYPE=auto \
    TEST_SIZE=0.2 \
    RANDOM_STATE=42 \
    CV_FOLDS=5 \
    OUTPUT_DIR=/app/output

# ---------------------------------------------------------------- dashboard
FROM base AS dashboard

COPY src/requirements-dashboard.txt .
RUN pip install -r requirements-dashboard.txt

COPY src/ .
USER appuser

# Listen on all interfaces inside the container; print only the localhost URL
# (no public-IP lookup), send no usage stats, and hide the "Deploy" button
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_SERVER_ADDRESS=localhost \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer

EXPOSE 8501

# Healthy = model artifacts present in the volume AND the Streamlit server responding
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "healthcheck.py"]

CMD ["streamlit", "run", "dashboard.py"]

# ---------------------------------------------------------------- tests
FROM dashboard AS tests

USER root
COPY tests/ ./tests/
RUN pip install -r tests/requirements.txt
USER appuser

# A one-off test run has no server to check
HEALTHCHECK NONE

CMD ["python", "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests"]

# ---------------------------------------------------------------- trainer (default)
FROM base AS trainer

COPY src/ .
USER appuser

# Train when the container starts; mount a volume on /app/output to keep the model
CMD ["python", "main.py"]
