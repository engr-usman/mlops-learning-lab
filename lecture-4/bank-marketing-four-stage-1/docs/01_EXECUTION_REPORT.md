# Execution report

Generated UTC: 2026-10-13T11:21:45.128907+00:00. Verdict: **PARTIAL**.

## Executive summary

A genuine four-stage KFP v2 definition was implemented and compiled. Validation,
training, evaluation and registration have independent file contracts and container
entrypoints. Exactly four tasks and their real artifact connections passed inspection.
The actual bank workflow succeeded in four separate local Python 3.12 processes using
new isolated MLflow SQLite/artifact storage. Thirty-three automated tests passed.

**No new Kubernetes/KFP run was submitted.** Docker and containerd sockets and the
cluster API are inaccessible under this session's sandbox. A read-only KFP health
probe also failed DNS resolution. The Docker build was attempted and failed before
building. No MinIO transfer or live metadata retrieval was verified.

The reported earlier lecture-05 success and shared model version 1 are user-provided
history, not live evidence gathered in this task. The version 1 below belongs to an
entirely separate local registry and must not be confused with that shared version.

## Environment

| Item | Observed / reported |
|---|---|
| Workstation | Ubuntu environment; bash; Asia/Karachi timezone |
| Application Python | 3.12.15, isolated `.venv/` |
| Application versions observed | MLflow 3.16.1; scikit-learn 1.9.1; pandas 3.0.6 |
| Compiler observed | Existing read-only Lecture 4 Python 3.12 environment; KFP 2.17.0 |
| Dependencies | Exact copied requirements installed offline; `pip check` passed |
| MicroK8s / Charmed Kubeflow | 1.35.x / 1.11 reported by user; current versions/health NOT VERIFIED |
| Intended namespace | `admin`; platform `kubeflow`; tracking `mlops` |
| Intended MLflow URI | `http://mlflow.mlops.svc.cluster.local:5000` |
| Intended MinIO service | `minio.kubeflow.svc.cluster.local:9000`; reachability NOT VERIFIED |

## Implementation and compilation

New directory: `lectures/lecture-04-kubeflow/bank-marketing-four-stage/`.
It did not exist before this task. Seven original source modules, `configs/config.yaml`,
`data/balance.csv` and `requirements.txt` were copied byte-for-byte. All changes are
inside the new directory. See [source changelog](07_SOURCE_CHANGELOG.md) for inventory.

Image reference: **`bank-marketing-kfp-four-stage:v1`**. Dockerfile targets Python 3.12,
packages only the active CSV and application, runs as UID 10001, and excludes verification
storage/environments. Image digest: **NOT VERIFIED / image not built here**.
Containerd import and Kubernetes image smoke test: **NOT VERIFIED**.

Compiled package: `pipelines/compiled/bank_marketing_four_stage_v1.yaml`.
SDK: kfp-2.17.0; IR schema: 2.1.0.
Spec verification: **PASS_SPEC_ONLY**. Tasks: `validate-dataset`, `train-candidates`,
`evaluate-candidates`, `register-winner`. Each uses the dedicated image, CPU/memory
requests and limits, disabled caching and explicit `IfNotPresent` pull policy.
The compiled file has a second YAML document for the Kubernetes platform extension.

## Execution evidence

Kubeflow run ID: **NOT AVAILABLE — not submitted**.
Argo workflow name/status: **NOT VERIFIED**.
Task status/duration in Kubernetes: **NOT VERIFIED**.

| Stage | Local verification | Seconds including process startup | Process PID | KFP task status |
|---|---|---:|---:|---|
| validate | PASS, local process | 5.025 | 19 | NOT VERIFIED |
| train | PASS, local process | 18.152 | 47 | NOT VERIFIED |
| evaluate | PASS, local process | 5.674 | 180 | NOT VERIFIED |
| register | PASS, local process | 4.522 | 207 | NOT VERIFIED |

Local run began UTC: 20261013T110612Z.
Local execution ID: `local-proof-20261013T110612Z`.
Local MLflow experiment ID: `1`.
Local parent run: `d777eafa6a244c399c07932525b62cf5`.
Local tracking URI: `sqlite:////home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/mlflow.db`.

| Candidate | Local MLflow child run ID | Actual logged model URI |
|---|---|---|
| logistic_regression | `4f8ece1f529a497fab6d2b3be4d5d52f` | `models:/m-e2e31285bd6a4be284bbbdcbf8f7b794` |
| decision_tree | `22f47d882a3747dbbee69b3ae487acfb` | `models:/m-c437b48b98914d898c2ed33ae080625a` |
| random_forest | `113aa2252316439090a04f8d4f2ab92a` | `models:/m-ba8956d36ef840d4ba66cb41b40d46a3` |
| gradient_boosting | `c6a6f85cf51a4a618070082c1011f9cf` | `models:/m-bcb30f66d7dd4edcbd3c77477bdf248c` |

| Candidate | Accuracy | Precision | Recall | F1 | Eligible |
|---|---:|---:|---:|---:|---|
| logistic_regression | 0.979206049 | 0.987500000 | 0.970699433 | 0.979027645 | True |
| decision_tree | 0.983459357 | 1.000000000 | 0.966918715 | 0.983181163 | True |
| random_forest | 0.982514178 | 0.992285439 | 0.972589792 | 0.982338902 | True |
| gradient_boosting | 0.982986767 | 0.992292871 | 0.973534972 | 0.982824427 | True |

Winner: **decision_tree**. Local registered model:
**bank-marketing-pipeline-model**, version **1**.
Its source run is `22f47d882a3747dbbee69b3ae487acfb` and source URI is `models:/m-c437b48b98914d898c2ed33ae080625a`.
Registered-model retrieval and sample prediction passed. Re-executing only registration
reused version 1; the isolated registry still contained exactly one version.

Dataset rows: 10,578; features: 16; train: 8,462; held-out: 2,116.
The active source CSV was renamed after validation in the test harness. Later processes
succeeded with that original input path absent, proving they used persisted stage outputs.
All candidates shared the same split and held-out hash. Training-only preprocessing and
split disjointness were tested. These scores use the same held-out data for selection;
they are not independent final-test estimates. `duration` remains a post-call feature.

## Artifact locations

Local verification root: `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z`.
Dataset/config/report outputs and actual model files are retained there. Concise copies
of JSON artifacts, held-out CSV and logs are under `docs/evidence/local-run/`.
See `docs/evidence/local-verification.json` for complete URIs, hashes, commands and results.

KFP/MinIO artifact URIs: **NOT VERIFIED — no live KFP run exists for this task**.
`docs/evidence/executor-contract-result.json` verifies only that generated artifact
metadata parses as the actual KFP ExecutorOutput protobuf; it does not verify upload.

## Failures and resolutions

- Snap wrappers refused execution under session confinement. Direct kubectl avoided
  the wrapper but still failed with `socket: operation not permitted`. No host repair
  was attempted; these errors do not prove the installation itself is unhealthy.
- Elevated Docker access was rejected by automatic approval review because sandbox
  escalation is disabled. The later normal Docker build also failed socket permissions.
- The first offline dependency attempts omitted usable ABI-compatible wheel tags and
  selected a Python 2 tag for a dual-tagged package. Correct cache tag selection enabled
  exact dependency installation. No upstream package version was changed.
- An initial compiler command used the wrong relative interpreter path. Correcting it
  to `../.venv/bin/python` compiled the spec. The existing compiler environment was not modified.
- No local four-stage training attempt failed. Additional rejection/retry/error scenarios
  were tested without unnecessary full retraining.

## Final verification matrix

| Requirement | Result |
|---|---|
| Existing lectures preserved | PASS: 68 protected file hashes unchanged |
| Exactly four typed KFP stages | PASS: compiled and inspected |
| Dataset/config artifact dependencies | PASS in spec and local processes; live KFP transfer NOT VERIFIED |
| Dataset validation | PASS on real CSV and invalid-input tests |
| Four models trained | PASS locally |
| Evaluation retrieves actual saved models | PASS in separate local process; another Pod NOT VERIFIED |
| Quality selection / no-winner handling | PASS in tests; happy-path locally |
| Registration and source linkage | PASS in isolated local registry |
| Sequential retry does not duplicate | PASS, mocked and actual local registry |
| Registered model retrievable | PASS locally; shared registry NOT VERIFIED |
| Dedicated image built/imported/startable | NOT VERIFIED; permission blocked |
| Real KFP run / MinIO storage | NOT VERIFIED; permission blocked |
| KFP API/Argo/MLMD consistency | NOT VERIFIED; permission blocked |
| Historical missing-context root cause | NOT DETERMINED |
| Tests and requested reports | PASS within documented scope |

There is no authorization request for destructive remediation. The remaining work needs
an execution environment with permitted Docker/cluster access and an authenticated KFP
endpoint, followed by image smoke testing, one identifiable live run and metadata review.
