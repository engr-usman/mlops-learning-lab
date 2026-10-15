# Image for the Kubeflow pipeline steps and the in-cluster MLflow server.
# Same as ../Dockerfile, but it stays root: the KFP v2 launcher writes step outputs
# under top-level folders such as /minio, which a non-root user can't create.
#
# Build from the project root, for the EC2 node's CPU (amd64 for most instance types):
#   docker buildx build --platform linux/amd64 -f kubeflow/Dockerfile \
#       -t docker.io/<your-dockerhub-user>/bank-marketing-kfp:v1 --push .
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py config.yaml ./
RUN mkdir -p /app/data /app/logs /mlflow
