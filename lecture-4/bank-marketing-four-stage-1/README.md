# Bank Marketing — four-stage Kubeflow Pipelines v2

Predicts whether a bank customer subscribes to a term deposit (`y` = yes/no) and runs
the whole ML workflow as four Kubeflow pipeline steps, with MLflow for tracking and the
model registry:

```
validate-dataset ──▶ train-candidates ──▶ evaluate-candidates ──▶ register-winner
 checks CSV rules     4 models, one        reloads models from      registers the best
 freezes config       stratified split     MLflow, quality gate,    model as a new
                      logs to MLflow       picks winner by F1       MLflow model version
```

Each step runs in its own pod from one Docker image and passes typed artifacts to the
next step. Nothing is shared in memory.

## Contents

1. [Project layout](#1-project-layout)
2. [Prerequisites](#2-prerequisites)
3. [First deployment](#3-first-deployment)
4. [Verify a successful run](#4-verify-a-successful-run)
5. [Release a new version (small change)](#5-release-a-new-version-small-change)
6. [Failure test cases](#6-failure-test-cases)
7. [Troubleshooting](#7-troubleshooting)
8. [Cleanup](#8-cleanup)

---

## 1. Project layout

| Path | Purpose |
|---|---|
| `kubeflow/Dockerfile` | Image used by all four steps (`src/`, `configs/`, `data/`) |
| `kubeflow/pipeline.py` | Pipeline definition. Compiles to `kubeflow/bank_marketing_pipeline.yaml` |
| `kubeflow/mlflow.yaml` | MLflow server, PVC and Service in namespace `mlflow-four-stage` |
| `configs/config.yaml` | Features, model hyperparameters, quality gate (`minimum_accuracy`) |
| `configs/config_strict.yaml` | Same config with `minimum_accuracy: 0.999` (failure test F5) |
| `data/balance.csv` | Training data (~10.5k rows) |
| `data/failure_cases/` | Broken CSVs for failure tests F3 and F4 |
| `src/` | Stage logic. Entry point: `python -m src.component_cli <stage>` |
| `tests/test_stages.py` | 33 unit tests |

Pipeline run parameters (set in **Create run**):

| Parameter | Default | Notes |
|---|---|---|
| `tracking_uri` | `http://mlflow.mlflow-four-stage.svc.cluster.local:5000` | In-cluster MLflow service |
| `dataset_path` | `/app/data/balance.csv` | Path **inside the image** |
| `config_path` | `/app/configs/config.yaml` | Path **inside the image** |

## 2. Prerequisites

- Kubeflow (with Pipelines) running on a kind cluster on EC2, dashboard reachable on `localhost:8080`.
- On the Mac: Docker with `buildx`, Python 3.10+ and a Docker Hub account.
- SSH access to the EC2 instance.

Set these once per terminal on the **Mac** and change the values to your own:

```bash
export DOCKER_USER=usm87                  # Docker Hub username
export TAG=v1                             # bump for every new release: v2, v3, ...
export IMAGE=docker.io/$DOCKER_USER/bank-marketing-kfp-four-stage:$TAG
export EC2=ubuntu@<ec2-ip>
export KEY=<path-to-key.pem>
```

Set this once per terminal on **EC2**:

```bash
export NS=kubeflow-user-example-com       # your Kubeflow profile namespace
```

## 3. First deployment

### Step 1 — Run the unit tests (Mac, optional but recommended)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m unittest tests.test_stages
```

Expected: `Ran 33 tests ... OK`.

### Step 2 — Build and push the image (Mac)

```bash
docker login
docker buildx build --platform linux/amd64 -f kubeflow/Dockerfile -t $IMAGE --push .
```

`--platform linux/amd64` is required on Apple Silicon because the EC2 nodes are amd64.

### Step 3 — Deploy MLflow (EC2, once)

```bash
scp -i $KEY kubeflow/mlflow.yaml $EC2:~          # Mac
```
```bash
kubectl apply -f mlflow.yaml                     # EC2
kubectl -n mlflow-four-stage rollout status deploy/mlflow
kubectl -n mlflow-four-stage get pods,svc,pvc
```

Expected: the pod is `1/1 Running`, the Service is `mlflow` on port 5000 and the PVC is `Bound`.

### Step 4 — Compile the pipeline (Mac)

```bash
python3 -m venv .venv-kfp
.venv-kfp/bin/pip install -r requirements-kfp.txt
KFP_IMAGE=$IMAGE .venv-kfp/bin/python kubeflow/pipeline.py
grep "image:" kubeflow/bank_marketing_pipeline.yaml   # all 4 lines must show $TAG
```

The image name is written into the YAML at compile time. Every new image tag needs a recompile.

### Step 5 — Open the dashboards

On EC2, run each command in its own terminal:

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
kubectl -n mlflow-four-stage port-forward svc/mlflow 5000:5000
```

On the Mac, open one SSH session that tunnels both ports:

```bash
ssh -i $KEY -L 8080:localhost:8080 -L 5000:localhost:5000 $EC2
```

- Kubeflow: http://localhost:8080
- MLflow: http://localhost:5000

### Step 6 — Upload and run

1. **Pipelines → Upload pipeline**. Name: `bank-marketing-four-stage`. File: `kubeflow/bank_marketing_pipeline.yaml`.
2. **Create run**. Experiment: `bank-marketing-four-stage-exp` (create it the first time).
3. Keep the default parameters and click **Start**.

Watch the pods from EC2:

```bash
kubectl get pods -n $NS -w
```

## 4. Verify a successful run

| # | Check | Where | Expected |
|---|---|---|---|
| S1 | Run status | KFP → Runs | ✅ green, all 4 steps succeeded |
| S2 | Step graph | KFP → run → Graph | validate → train → evaluate → register, with artifact boxes between them |
| S3 | Image used | `kubectl get pod <pod> -n $NS -o jsonpath='{.spec.containers[*].image}'` | Shows `$IMAGE` |
| S4 | Step logs | KFP → click a step → Logs, or `kubectl logs <pod> -n $NS -c main` | Ends with `COMPONENT \| stage=<stage> \| state=SUCCEEDED` |
| S5 | Metrics | KFP → evaluate-candidates → `evaluation_metrics` | Accuracy/precision/recall/F1 per model, `quality_gate_passed = 1` |
| S6 | MLflow runs | MLflow → `bank-marketing-pipeline` | One `four-stage-<run-id>` parent with 4 child runs, one per model |
| S7 | Parent tags | MLflow → parent run → Tags | `quality_gate=passed`, `registration_status=registered`, `selected_model=<winner>` |
| S8 | Model registry | MLflow → Models → `bank-marketing-pipeline-model` | A new version, sourced from the winning child run |
| S9 | Registration output | KFP → register-winner → `registration_result` | `"status": "registered"` and the version number |

The `four-stage-<id>` name in MLflow matches the KFP run ID, which links the two systems.

## 5. Release a new version (small change)

Example: change the decision tree depth.

1. Edit `configs/config.yaml`:
   ```yaml
   decision_tree:
     max_depth: 5        # was 3
   ```
2. Rebuild with a **new tag**. Never reuse a tag: the pipeline uses
   `imagePullPolicy: IfNotPresent`, so nodes keep running the cached image for an old tag.
   ```bash
   export TAG=v2
   export IMAGE=docker.io/$DOCKER_USER/bank-marketing-kfp-four-stage:$TAG
   docker buildx build --platform linux/amd64 -f kubeflow/Dockerfile -t $IMAGE --push .
   ```
3. Recompile:
   ```bash
   KFP_IMAGE=$IMAGE .venv-kfp/bin/python kubeflow/pipeline.py
   grep "image:" kubeflow/bank_marketing_pipeline.yaml    # must show v2
   ```
4. KFP → **Pipelines → bank-marketing-four-stage → Upload version**. Version name: `v2-dt-depth5`.
5. **Create run**. Select pipeline version `v2-dt-depth5`, keep the same experiment, then **Start**.
6. Verify:
   - S3 shows the `v2` image.
   - MLflow → new parent run → `decision_tree` child → Parameters shows `max_depth = 5`.
   - Select both parent runs → **Compare** to see the metric differences.
   - Models → `bank-marketing-pipeline-model` has a **new version** (every successful run registers one).
7. Optional: on the better version, MLflow → model version → **Add alias** → `champion`.

## 6. Failure test cases

These checks confirm that the pipeline **fails loudly at the right step** and never
registers a bad model. F2–F5 need **no rebuild**. You only change a run parameter in
**Create run**, because the broken inputs ship inside the image.

| # | Scenario | How to trigger | Expected result |
|---|---|---|---|
| F1 | Compile without an image | `.venv-kfp/bin/python kubeflow/pipeline.py` (no `KFP_IMAGE`) | Exits with `Set KFP_IMAGE, ...`. No YAML is written |
| F2 | Dataset file missing | `dataset_path = /app/data/does_not_exist.csv` | ❌ **validate-dataset** fails: `FileNotFoundError: Dataset not found`. Later steps never start. No MLflow run |
| F3 | Invalid category value | `dataset_path = /app/data/failure_cases/invalid_category.csv` | ❌ **validate-dataset** fails: `ValueError: Invalid values in job: ['astronaut']` |
| F4 | Required column missing | `dataset_path = /app/data/failure_cases/missing_column.csv` | ❌ **validate-dataset** fails: `ValueError: Missing columns: ['duration']` |
| F5 | Quality gate rejects all models | `config_path = /app/configs/config_strict.yaml` | ✅ All 4 steps green, but **no new model version**. `registration_result` → `"status": "not_registered"`, reason `No candidate passed the minimum accuracy gate`. MLflow parent tags: `quality_gate=failed`, `registration_status=not_registered`. Each child has `quality_gate_passed = 0` |
| F6 | Wrong MLflow URL | `tracking_uri = http://mlflow.wrong-ns.svc.cluster.local:5000` | ✅ validate succeeds. ❌ **train-candidates** fails with a connection/DNS error after MLflow's HTTP retries, which can take a few minutes. evaluate and register never start |
| F7 | MLflow server down | See commands below | ❌ **train-candidates** fails as in F6. After MLflow is restored, earlier runs and model versions are still there because they live on the PVC |
| F8 | Image tag does not exist | See commands below | ⏳ validate-dataset pod stays in `ImagePullBackOff` / `ErrImagePull`. The run never progresses |

**F5 is the most important test.** A run can be "green" in Kubeflow and still correctly
refuse to register. Always check `registration_result`, not just the run color.

### F7 — MLflow down (EC2)

```bash
kubectl -n mlflow-four-stage scale deploy/mlflow --replicas=0
# start a run with default parameters in the KFP UI, wait for train-candidates to fail
kubectl -n mlflow-four-stage scale deploy/mlflow --replicas=1
kubectl -n mlflow-four-stage rollout status deploy/mlflow
# MLflow UI: previous runs and model versions are still listed
```

### F8 — Image tag does not exist

```bash
KFP_IMAGE=docker.io/$DOCKER_USER/bank-marketing-kfp-four-stage:does-not-exist \
  .venv-kfp/bin/python kubeflow/pipeline.py                         # Mac
```

1. Upload the YAML as pipeline version `broken-image` and start a run.
2. On EC2, run `kubectl get pods -n $NS` and `kubectl describe pod <validate-pod> -n $NS`. The pod shows `ImagePullBackOff`, and the Events mention `not found`.
3. In KFP, **Terminate** the run.
4. Recompile with the real image before the next normal run:
   ```bash
   KFP_IMAGE=$IMAGE .venv-kfp/bin/python kubeflow/pipeline.py
   ```

### Where to read the error for a failed step

```bash
kubectl get pods -n $NS --sort-by=.metadata.creationTimestamp | tail -8
kubectl logs <failed-pod> -n $NS -c main --tail=50
```

You can also open KFP → run → click the red step → **Logs**. Look for
`COMPONENT | stage=<stage> | state=FAILED` and the Python error above it.

### Failure test checklist

| Test | Failed step | Error seen | New model version? | Pass? |
|---|---|---|---|---|
| F1 | — (compile) | `Set KFP_IMAGE` | No | ☐ |
| F2 | validate-dataset | `Dataset not found` | No | ☐ |
| F3 | validate-dataset | `Invalid values in job` | No | ☐ |
| F4 | validate-dataset | `Missing columns` | No | ☐ |
| F5 | none (all green) | `not_registered` | **No** | ☐ |
| F6 | train-candidates | connection error | No | ☐ |
| F7 | train-candidates | connection error, data kept | No | ☐ |
| F8 | validate-dataset (pod) | `ImagePullBackOff` | No | ☐ |

After the failure tests, run once more with default parameters and confirm S1–S9 still pass.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImagePullBackOff` on a normal run | Tag not pushed, typo, or private repo | Run `docker buildx imagetools inspect $IMAGE`. Make the repo public or add an `imagePullSecret` |
| `exec format error` | Image built for arm64 | Rebuild with `--platform linux/amd64` |
| Code change not picked up | Same tag reused, so the cached image ran | Push a new tag and recompile |
| Pods `Pending` | Not enough CPU/memory (train requests 1Gi, limit 3Gi) | `kubectl describe pod`, then use a larger instance |
| MLflow `403 Invalid Host header` | Host not in `--allowed-hosts` | Check `kubectl logs deploy/mlflow -n mlflow-four-stage` and update `mlflow.yaml` |
| evaluate fails loading a model | MLflow not serving artifacts | `mlflow.yaml` must keep `--artifacts-destination` and `--serve-artifacts` |
| MLflow pod `CrashLoopBackOff` | Volume or DB problem | `kubectl logs deploy/mlflow -n mlflow-four-stage --previous` |

## 8. Cleanup

```bash
# Remove MLflow. WARNING: deleting the namespace deletes the PVC, so all runs and models are lost.
kubectl delete -f mlflow.yaml
```

Old pipeline versions, runs and experiments can be deleted or archived from the KFP UI.

---

Background: [CODEX_HANDOFF.md](CODEX_HANDOFF.md) and [docs/](docs/) describe the original
local verification, architecture ([docs/02](docs/02_ARCHITECTURE_AND_DATA_FLOW.md)) and
operations ([docs/05](docs/05_DEBUGGING_AND_OPERATIONS.md)). `pipelines/four_stage.py` is the
earlier version of the pipeline. Use `kubeflow/pipeline.py` for deployment.
