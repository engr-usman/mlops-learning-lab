FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && chown appuser:appuser /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser tests/ ./tests/
USER appuser
CMD ["python", "-m", "src.component_cli", "--help"]
