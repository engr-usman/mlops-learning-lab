# One image for every service in docker-compose.yml: the MLflow server, the
# training job and the API. Only the command differs between them.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install dependencies before copying the code, so editing a .py file
# doesn't reinstall every package on the next build.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py config.yaml ./

# Run as a normal user, not root. /mlflow holds the MLflow server's data.
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/logs /mlflow \
    && chown -R appuser /app /mlflow
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
