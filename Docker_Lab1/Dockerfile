# Slim Python image instead of the full python:3.10 image used in the original lab
FROM python:3.11-slim

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

COPY src/ .

# Run as a non-root user that owns the output directory
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/output \
    && chown appuser:appuser /app/output
USER appuser

# Training settings; override any of them with `docker run -e NAME=value`
ENV MODEL_TYPE=auto \
    TEST_SIZE=0.2 \
    RANDOM_STATE=42 \
    CV_FOLDS=5 \
    OUTPUT_DIR=/app/output

# Train when the container starts; mount a volume on /app/output to keep the model
CMD ["python", "main.py"]
