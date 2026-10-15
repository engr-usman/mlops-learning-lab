# Bank Marketing: End-to-End MLOps Project

This project predicts whether a bank customer will **subscribe to a term deposit** (`yes` / `no`) after a marketing phone call.

It was built step by step, the same way a real ML project grows:

1. **Explore the data by hand** in a Jupyter notebook
2. **Turn the notebook into clean scripts**: config, logging, validation, training, evaluation and an API
3. **Track experiments with MLflow** (every training run and model version is recorded)
4. **Package everything with Docker**, so it runs the same on any machine
5. **Automate it with a Kubeflow pipeline** on a Kubernetes cluster running on AWS EC2

Each section below explains **what** was done, **why**, and the **exact commands** to repeat it.

---

## Table of contents

- [Project structure](#project-structure)
- [Step 1: Get the dataset](#step-1-get-the-dataset)
- [Step 2: Set up the Python environment](#step-2-set-up-the-python-environment)
- [Step 3: Explore the data manually (notebook)](#step-3-explore-the-data-manually-notebook)
- [Step 4: Turn the notebook into scripts](#step-4-turn-the-notebook-into-scripts)
- [Step 5: Train, evaluate and serve locally](#step-5-train-evaluate-and-serve-locally)
- [Step 6: Docker](#step-6-docker)
- [Step 7: Kubeflow pipeline on AWS EC2](#step-7-kubeflow-pipeline-on-aws-ec2)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## Project structure

```
lecture-03a/
├── data/
│   ├── bank-full.csv        # original dataset (semicolon-separated)
│   ├── data.csv             # same data, comma-separated (used by all scripts)
│   └── test.csv             # held-out test split, written by train.py
├── notebook/Untitled.ipynb  # first manual exploration
├── config.yaml              # ALL settings: paths, schema, model params, thresholds
├── config_loader.py         # reads config.yaml + points MLflow to the right place
├── logger.py                # shared logging (console + logs/<name>.log)
├── ingest.py                # builds data.csv from bank-full.csv (local file or URL)
├── data_validation.py       # checks data.csv against the rules in config.yaml
├── train.py                 # trains the model, logs + registers it in MLflow
├── evaluate.py              # tests a model version, promotes it to "champion"
├── api.py                   # FastAPI app that serves the champion model
├── requirements.txt         # Python libraries (pinned versions)
├── Dockerfile               # image for MLflow server, trainer and API
├── docker-compose.yml       # runs MLflow + API (+ trainer) together
├── .dockerignore            # keeps data/logs/mlruns out of the image
├── kubeflow/
│   ├── Dockerfile           # image for the Kubeflow pipeline steps
│   ├── mlflow.yaml          # MLflow server running inside Kubernetes
│   └── pipeline.py          # the Kubeflow pipeline definition
├── logs/                    # log files (created automatically)
├── mlflow.db, mlruns/       # local MLflow data (created automatically)
```

---

## Step 1: Get the dataset

**Dataset:** [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing): 45,211 rows, 16 input columns + 1 target column `y`.

1. Download `bank-full.csv` from the UCI page (it is inside the zip file) and put it in `data/`.
2. The original file uses `;` (semicolon) between values. Most tools expect `,` (comma), so convert it to `data/data.csv`:

```bash
python -c "import pandas as pd; pd.read_csv('data/bank-full.csv', sep=';').to_csv('data/data.csv', index=False)"
```

> Later, this step became the script `ingest.py`, which does the same thing and can also download the zip directly:
> `python ingest.py` (local file) or `python ingest.py --source https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`

**Important columns:**

| Column | Meaning |
|---|---|
| `age`, `job`, `marital`, `education` | Who the customer is |
| `balance`, `housing`, `loan`, `default` | Customer's financial situation |
| `contact`, `day`, `month`, `duration`, `campaign` | Details of the current marketing call |
| `pdays`, `previous`, `poutcome` | Previous marketing campaigns (`pdays = -1` means never contacted) |
| `y` | **Target**: did the customer subscribe? (`yes` / `no`) |

---

## Step 2: Set up the Python environment

A virtual environment (`.venv`) keeps this project's libraries separate from the rest of your computer.

```bash
cd lecture-03a
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter                # only needed for the notebook
```

---

## Step 3: Explore the data manually (notebook)

**Why:** before writing any "real" code, you need to understand the data: its size, data types, missing values and how balanced the target is.

```bash
jupyter notebook notebook/Untitled.ipynb
```

What was checked in the notebook:

| Code | What it tells us |
|---|---|
| `df.head()`, `df.shape`, `df.info()` | First rows, size (45211 × 17), column types |
| `df.isnull().sum()` | No real null values |
| `(df == 'unknown').sum()` | Missing values are written as the word **`unknown`** instead (in `job`, `education`, `contact`, `poutcome`) |
| `df['y'].value_counts()` | **Imbalanced target**: only ~12% of customers say `yes` |
| `select_dtypes(...)` | Split columns into numeric and categorical |
| `value_counts()` per category | All allowed values of each categorical column |
| `train_test_split(..., test_size=0.20, random_state=42, stratify=y)` | 80% train / 20% test, with the same yes/no ratio in both |

**Key findings that shaped the later code:**

- **`duration` leaks the answer.** It is the length of the call, which is only known *after* the call is finished. A real model can't use it before calling the customer, so it is **dropped** from training.
- **The target is imbalanced**, so accuracy alone is misleading (a model that always says "no" is 88% accurate!). We use `class_weight: balanced` and also look at **recall** and **ROC AUC**.
- The allowed values and ranges found here became the **validation rules** in `config.yaml`.

---

## Step 4: Turn the notebook into scripts

A notebook is good for exploring, but hard to rerun, test or automate. So the work was split into small scripts, each with one job. All settings live in **one** file (`config.yaml`), so you never hard-code numbers or paths inside the Python code.

### 4.1 `config.yaml`: all settings in one place

Contains:
- `paths`: where data, logs and MLflow files live
- `numeric_features` with allowed `[min, max]` ranges, and `categorical_features` with allowed values
- `validation`: minimum rows, maximum duplicate fraction
- `split`: test size and random seed
- `training`: features to drop (`duration`) and parameters for each model (`logistic_regression`, `random_forest`)
- `mlflow`: experiment name, registered model name, and the **promotion thresholds** (`roc_auc >= 0.75`, `recall >= 0.50`)

**Why:** change behaviour (e.g. try a different model) by editing the YAML, without touching the code.

### 4.2 `config_loader.py`: read the config

- `load_config()` reads `config.yaml` once and turns relative paths into full paths.
- `setup_mlflow()` connects to MLflow:
  - **locally**, it uses the `mlflow.db` SQLite file
  - **in Docker/Kubeflow**, it uses the server given in the `MLFLOW_TRACKING_URI` environment variable

### 4.3 `logger.py`: consistent logs

`get_logger("train")` prints messages to the console **and** saves them to `logs/train.log`. Every script uses it, so all logs look the same:

```
2026-10-11 16:25:10,416 | INFO     | api | Loaded models:/bank-marketing-classifier@champion (version 2)
```

**Why:** `print()` disappears; log files let you check later what happened and when.

### 4.4 `data_validation.py`: check the data before training

Checks: correct columns and order, enough rows, no nulls, numbers within range, only allowed category values, both target classes present, not too many duplicates. It also warns about `unknown` values.

```bash
python data_validation.py            # validates data/data.csv
python data_validation.py some.csv   # validates another file
```

It **exits with code 1 if any check fails**, so it can stop a pipeline before bad data reaches the model ("garbage in, garbage out").

### 4.5 `train.py`: train and register the model

What it does, in order:
1. Validates the data (stops if it's bad)
2. Drops `duration`, splits 80/20 (stratified) and saves the test split to `data/test.csv`
3. Builds a **scikit-learn Pipeline**: `StandardScaler` for numbers + `OneHotEncoder` for categories + the model
4. Trains it and logs parameters, metrics and the model to **MLflow**
5. **Registers** the model as a new version of `bank-marketing-classifier`

**Why the preprocessing is inside the pipeline:** the saved model then applies exactly the same scaling/encoding to new data at prediction time. No mismatch between training and serving.

### 4.6 `evaluate.py`: test the model and pick a "champion"

1. Loads a registered model version (latest by default)
2. Calculates accuracy, precision, recall, F1 and ROC AUC on `test.csv`
3. Logs the metrics, confusion matrix and classification report to MLflow
4. **Promotes** the version to the `champion` alias only if:
   - `roc_auc >= 0.75` **and** `recall >= 0.50`, **and**
   - its `roc_auc` is at least as good as the current champion's

**Why:** a new model should never replace a better one automatically. The API always serves whatever is `champion`.

### 4.7 `api.py`: serve predictions with FastAPI

On startup it loads `models:/bank-marketing-classifier@champion` from MLflow.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the API running and is a model loaded? |
| `GET /model-info` | Which model version is being served |
| `POST /predict` | Predict for one customer |
| `POST /predict/batch` | Predict for up to 1000 customers |

Input is validated automatically (e.g. `age` must be 18–100, `job` must be an allowed value), using the same rules from `config.yaml`.

---

## Step 5: Train, evaluate and serve locally

Run these from the project root with `.venv` activated:

```bash
# 1. Validate the data
python data_validation.py

# 2. Train (random_forest is the default from config.yaml)
python train.py
python train.py --model logistic_regression     # try another model

# 3. Evaluate the latest version and promote it if it's good enough
python evaluate.py
python evaluate.py --version 2 --no-promote     # evaluate a specific version, don't promote

# 4. Look at all runs and models in the MLflow UI → http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 5. Start the API (in another terminal) → docs at http://127.0.0.1:8000/docs
uvicorn api:app --reload
```

**Test the API:**

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 58, "job": "management", "marital": "married", "education": "tertiary",
       "default": "no", "balance": 2143, "housing": "yes", "loan": "no", "contact": "unknown",
       "day": 5, "month": "may", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"}'
```

Expected answer, something like:
```json
{"subscribed": "no", "probability_yes": 0.1599, "model_version": "2"}
```

---

## Step 6: Docker

**Why Docker:** "it works on my machine" is not enough. Docker puts the code, Python and all libraries into one **image**, so it runs the same everywhere: your laptop, a server, or Kubernetes.

### 6.1 The files

- **`Dockerfile`** builds one image (`python:3.14-slim`) used by all services. Only the start command is different. It installs `requirements.txt` first (so code changes don't reinstall everything) and runs as a normal user (`appuser`), not root, for safety.
- **`.dockerignore`** keeps `data/`, `logs/`, `mlruns/`, `mlflow.db`, `.venv` and the notebook **out** of the image. Data is mounted at runtime instead of baked in.
- **`docker-compose.yml`** runs three services:

| Service | What it does | Port |
|---|---|---|
| `mlflow` | MLflow tracking server + model registry (data kept in a Docker volume) | 5000 |
| `trainer` | Runs `train.py` then `evaluate.py` (only when you ask for it) | – |
| `api` | The FastAPI app, loads the champion from the `mlflow` service | 8000 |

### 6.2 Commands

```bash
# Build the image and start MLflow + API
docker compose up -d --build

# Train + evaluate inside Docker (registers a model and promotes a champion)
docker compose run --rm trainer

# Restart the API so it loads the new champion
docker compose restart api

# Check that everything is running
docker compose ps
docker compose logs -f api
```

### 6.3 Test it

- MLflow UI: http://localhost:5000 (you should see the `bank-marketing` experiment)
- API docs: http://localhost:8000/docs
- Same `curl` commands as in [Step 5](#step-5-train-evaluate-and-serve-locally), on port 8000

```bash
# Stop everything (add -v to also delete the MLflow data volume)
docker compose down
```

---

## Step 7: Kubeflow pipeline on AWS EC2

### 7.1 The big picture

**Kubeflow** is an ML platform that runs on **Kubernetes**. A **Kubeflow Pipeline** is a list of steps that run automatically, each one in its own container (pod), passing files to the next step.

Think of Kubeflow as a **factory** on your EC2 machine. To make a model it needs:

| What the factory needs | In this project | Setup step |
|---|---|---|
| Workers + tools | The Docker image with our code | 7.3 |
| A record book | MLflow server inside the cluster | 7.4 |
| Work instructions | The compiled pipeline YAML | 7.5 |
| "Start working!" | Upload + Create run in the UI | 7.6 |

Our pipeline:

```
ingest ──► validate ──► train ──► evaluate
```

| Step | Script it runs | What it does |
|---|---|---|
| `ingest` | `ingest.py` | Downloads the UCI zip, builds `data.csv` |
| `validate` | `data_validation.py` | Checks the data; **the run stops here** if the data is bad |
| `train` | `train.py` | Trains, registers a new model version in MLflow, outputs `test.csv` + the version number |
| `evaluate` | `evaluate.py` | Tests that version and promotes it to `champion` if it qualifies |

Kubeflow passes files between steps (`data.csv` → validate/train, `test.csv` + version → evaluate). The two small changes made to `train.py` for this, `--test-out` and `--version-out`, tell it where to write those outputs.

### 7.2 Prerequisites

- Kubeflow installed on an AWS EC2 instance, and `kubectl` working on that instance
- Dashboard reachable from your laptop through an SSH tunnel, for example:

  ```bash
  # on EC2: expose the Kubeflow gateway
  kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80

  # on your laptop: tunnel port 8080 to EC2
  ssh -i <key.pem> -L 8080:localhost:8080 ubuntu@<ec2-public-ip>
  ```

  Then open http://localhost:8080 and log in (default Kubeflow user: `user@example.com`).
- A **Docker Hub** account (free), where the image is stored so EC2 can download it
- The `kfp` library on your laptop: `pip install "kfp>=2,<3"`

### 7.3 Build and push the image

```bash
docker login
docker buildx build --platform linux/amd64 -f kubeflow/Dockerfile \
  -t docker.io/<your-dockerhub-user>/bank-marketing-kfp:v1 --push .
```

**Why:**
- The code is on your laptop, but Kubeflow runs on EC2. Pushing to Docker Hub lets EC2 **pull** it. Make the repository **public**.
- `--platform linux/amd64`: a Mac with an Apple chip builds ARM images by default, but most EC2 instances are x86 (amd64). Check with `uname -m` on EC2: `x86_64` → `linux/amd64`, `aarch64` → `linux/arm64`.
- `kubeflow/Dockerfile` is like the main `Dockerfile` but **runs as root**, because the Kubeflow v2 launcher writes step outputs to top-level folders (like `/minio`) that a normal user can't create.

### 7.4 Deploy MLflow inside the cluster

**Why:** pipeline steps are temporary pods that disappear when they finish. `train` and `evaluate` need a **shared, permanent place** to store models and metrics, and Kubeflow doesn't include MLflow.

`kubeflow/mlflow.yaml` creates:
- a namespace `mlflow` (with Istio sidecar injection **disabled**, so pipeline pods can call it with plain HTTP)
- a 5Gi **PersistentVolumeClaim** (disk), so runs and models survive restarts
- a **Deployment** running `mlflow server` with the same settings as in `docker-compose.yml`
- a **Service**, so pipeline pods can reach it at `http://mlflow.mlflow.svc.cluster.local:5000`

First set your image in `kubeflow/mlflow.yaml` (`image: docker.io/<your-dockerhub-user>/bank-marketing-kfp:v1`), then:

```bash
# copy the file to EC2
scp -i <key.pem> kubeflow/mlflow.yaml ubuntu@<ec2-public-ip>:~

# on EC2
kubectl apply -f mlflow.yaml
kubectl -n mlflow rollout status deploy/mlflow
kubectl -n mlflow get pods        # should show 1/1 Running
```

### 7.5 Compile the pipeline

**Why:** the pipeline is written in Python (`kubeflow/pipeline.py`), but Kubeflow only understands **YAML**. Compiling converts it.

```bash
source .venv/bin/activate
KFP_IMAGE=docker.io/<your-dockerhub-user>/bank-marketing-kfp:v1 python kubeflow/pipeline.py
# → writes kubeflow/bank_marketing_pipeline.yaml
```

`KFP_IMAGE` writes **your** image name into every step of the YAML.

### 7.6 Upload and run the pipeline

In the Kubeflow dashboard (http://localhost:8080):

1. **Pipelines → Pipelines → Upload pipeline**, choose `kubeflow/bank_marketing_pipeline.yaml`, name it `bank-marketing`.
2. Open the pipeline and click **Create run**.
3. Choose or create an **Experiment** (just a folder that groups related runs).
4. Set the run **parameters**:
   - `model`: `random_forest` (default) or `logistic_regression`
   - `source`: the dataset URL (the default UCI link works)
5. Click **Start**.
6. Go to **Runs**, open your run and watch the graph. Click any step to see its **Logs**.

### 7.7 See the results in MLflow

```bash
# on EC2
kubectl -n mlflow port-forward svc/mlflow 5000:5000

# on your laptop (second terminal)
ssh -i <key.pem> -L 5000:localhost:5000 ubuntu@<ec2-public-ip>
```

Open http://localhost:5000 → experiment `bank-marketing`. Each pipeline run adds two MLflow runs:
- `train-random_forest`: training params, metrics and the registered model
- `evaluate-vN`: test metrics for **model version N**

> The `vN` in `evaluate-vN` is the **MLflow model version**, not the Docker image tag. Every pipeline run creates a new version (v1, v2, v3, ...).

### 7.8 Running again / after code changes

| Situation | What to do |
|---|---|
| **No code change**, just want a new run | Kubeflow UI → open the pipeline → **Create run** |
| **Changed the code** | Rebuild with a new tag (`:v2`), recompile with `KFP_IMAGE=...:v2`, then in the UI open the pipeline → **Upload version** → **Create run** |

```bash
docker buildx build --platform linux/amd64 -f kubeflow/Dockerfile \
  -t docker.io/<your-dockerhub-user>/bank-marketing-kfp:v2 --push .
KFP_IMAGE=docker.io/<your-dockerhub-user>/bank-marketing-kfp:v2 python kubeflow/pipeline.py
```

> `kubectl apply -f mlflow.yaml` only restarts the **MLflow server**. It does **not** train anything. New runs appear in MLflow only when a **pipeline run** happens.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| Pod stuck in `ImagePullBackOff` | Docker Hub repo is private, or the image name/tag is wrong |
| Step fails with `exec format error` | Image built for the wrong CPU. Rebuild with the right `--platform` |
| `ingest` fails with a network error | The cluster can't reach the internet. Set the `source` parameter to a URL it can reach |
| `train`/`evaluate` can't connect to MLflow | `kubectl -n mlflow get pods` and `kubectl -n mlflow logs deploy/mlflow` |
| `validate` step fails | The data broke a rule. Its logs list exactly which check failed |
| Upload says the pipeline spec is invalid | Kubeflow Pipelines backend is older than the `kfp` SDK. Install an older `kfp` 2.x and recompile |
| No new runs in MLflow | You restarted MLflow but didn't start a pipeline run (see 7.8) |
| API says `Model not loaded` | No `champion` yet. Run train + evaluate first, or the thresholds weren't met |

---

## Glossary

| Term | Simple meaning |
|---|---|
| **Data leakage** | Using information during training that won't exist at prediction time (here: `duration`) |
| **Stratified split** | Train and test sets keep the same yes/no ratio |
| **ROC AUC** | How well the model ranks "yes" customers above "no" customers (0.5 = random, 1.0 = perfect) |
| **Recall** | Of all real "yes" customers, how many did the model find |
| **MLflow run** | One record of a training or evaluation: params, metrics, files |
| **Model registry / version** | A named list of saved models; each training adds a new version |
| **Champion alias** | A label pointing to the version the API should serve |
| **Docker image / container** | Image = packaged app; container = that package running |
| **Kubernetes pod** | The smallest running unit in Kubernetes (one or more containers) |
| **Namespace** | A separate "folder" inside the cluster (e.g. `mlflow`, `kubeflow-user-example-com`) |
| **PVC** | A disk for pods, whose data survives pod restarts |
| **Kubeflow pipeline / run** | Pipeline = the recipe of steps; run = one execution of it |
