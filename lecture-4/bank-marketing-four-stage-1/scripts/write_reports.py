"""Generate evidence-backed reports without substituting local runs for KFP runs."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
EVIDENCE = DOCS / 'evidence'
LOCAL = json.loads((EVIDENCE / 'local-verification.json').read_text())
SPEC = json.loads((EVIDENCE / 'spec-verification.json').read_text())
PRESERVATION = json.loads((EVIDENCE / 'source-preservation.json').read_text())
REG = LOCAL['registration']
STAMP = datetime.now(timezone.utc).isoformat()


def write(name, value):
    (DOCS / name).write_text(textwrap.dedent(value).strip() + '\n')


def metrics_table():
    lines = ['| Candidate | Accuracy | Precision | Recall | F1 | Eligible |',
             '|---|---:|---:|---:|---:|---|']
    for item in LOCAL['evaluation']['candidates']:
        m = item['metrics']
        lines.append(f"| {item['model_name']} | {m['accuracy']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['f1_score']:.9f} | {item['eligible']} |")
    return '\n'.join(lines)


run_table = '\n'.join(f"| {s['stage']} | PASS, local process | {s['seconds']:.3f} | {s['pid']} | NOT VERIFIED |" for s in LOCAL['stages'])
lineage_table = '\n'.join(f"| {c['model_name']} | `{c['run_id']}` | `{c['model_uri']}` |" for c in LOCAL['manifest']['candidates'])
write('01_EXECUTION_REPORT.md', f'''
# Execution report

Generated UTC: {STAMP}. Verdict: **PARTIAL**.

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
SDK: {SPEC['sdk_version']}; IR schema: {SPEC['schema_version']}.
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
{run_table}

Local run began UTC: {LOCAL['started_utc']}.
Local execution ID: `{REG['execution_id']}`.
Local MLflow experiment ID: `{LOCAL['experiment_id']}`.
Local parent run: `{REG['parent_run_id']}`.
Local tracking URI: `{LOCAL['tracking_uri']}`.

| Candidate | Local MLflow child run ID | Actual logged model URI |
|---|---|---|
{lineage_table}

{metrics_table()}

Winner: **{REG['selected_model']}**. Local registered model:
**{REG['registered_model_name']}**, version **{REG['registered_model_version']}**.
Its source run is `{REG['selected_run_id']}` and source URI is `{REG['selected_model_uri']}`.
Registered-model retrieval and sample prediction passed. Re-executing only registration
reused version 1; the isolated registry still contained exactly one version.

Dataset rows: 10,578; features: 16; train: 8,462; held-out: 2,116.
The active source CSV was renamed after validation in the test harness. Later processes
succeeded with that original input path absent, proving they used persisted stage outputs.
All candidates shared the same split and held-out hash. Training-only preprocessing and
split disjointness were tested. These scores use the same held-out data for selection;
they are not independent final-test estimates. `duration` remains a post-call feature.

## Artifact locations

Local verification root: `{LOCAL['root']}`.
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
| Existing lectures preserved | PASS: {PRESERVATION['protected_files']} protected file hashes unchanged |
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
''')

write('02_ARCHITECTURE_AND_DATA_FLOW.md', '''
# Architecture and data flow

## Four real stages

```mermaid
flowchart TD
    A[Image: original CSV and YAML] --> V[1 validate-dataset]
    V --> D[Dataset: validated CSV]
    V --> C[Artifact: configuration snapshot]
    V --> R[Artifact: validation report]
    D --> T[2 train-candidates]
    C --> T
    R --> T
    T --> M[Artifact: candidate manifest with MLflow URIs]
    T --> H[Dataset: held-out CSV with source row IDs]
    T --> S[Artifact: split metadata]
    M --> E[3 evaluate-candidates]
    H --> E
    S --> E
    C --> E
    E --> Q[Artifact: metrics, eligibility, winner and gate]
    E --> K[Metrics: candidate scores in KFP]
    Q --> G[4 register-winner]
    C --> G
    T --> ML[MLflow runs and fitted models]
    ML --> E
    G --> MR[MLflow registered version or explicit rejection]
    G --> O[Artifact: registration result]
```

KFP's launcher materializes input artifacts before invoking the process and uploads
output files after success. The backend's configured pipeline root determines their
object-store URIs; MinIO is the intended KFP store here. Application code receives local
artifact paths, not MinIO credentials. The image contains source data only for stage 1.
Training and evaluation do not independently reopen that original CSV.

## Contracts

| Stage | Inputs | Outputs | Functions |
|---|---|---|---|
| Validate | Source CSV path and YAML path in image | `Output[Dataset]` validated dataset; `Output[Artifact]` configuration and validation report | `validate_stage`, `load_config`, `check_config`, `validate_dataset`, splitter preflight |
| Train | `Input[Dataset]`; configuration/report artifacts; tracking URI; KFP execution ID | Candidate-manifest artifact; held-out Dataset; split-metadata artifact | `train_stage`, `split_dataset`, `train_model` |
| Evaluate | Manifest, held-out Dataset, split and configuration artifacts; tracking URI | Evaluation Artifact; `Output[Metrics]` | `evaluate_stage`, `load_model`, `evaluate_model`, `passes_quality_gate`, `select_best_model` |
| Register | Evaluation and configuration artifacts; tracking URI | Registration-result Artifact | `register_stage`, version search, `register_model`, registry retrieval |

The evaluation report contains candidate metrics/eligibility, selected candidate or
null, and an explicit gate object. These are kept together in one typed Artifact to
avoid independently inconsistent winner/gate parameters. Registration recomputes the
selection from that report and rejects contradictions before any registry creation.
The candidate manifest references models in MLflow; it is not itself a fitted model.

### File content schemas

- Validated CSV: all original columns with validated numeric values, original row order,
  no injected label encoding. Validation report records original and normalized CSV hashes,
  counts, feature order, label and class counts.
- Configuration: JSON snapshot of loaded YAML, hashed before downstream use. JSON preserves
  the model mapping's order and avoids a task independently reading a changed YAML file.
- Held-out CSV: ordered feature columns, target and `__source_row_id`. That reserved ID is
  excluded from model inputs; evaluation checks its order against recorded test row IDs.
- Split metadata: train/test IDs, sizes, seed, ratio, stratification, hash of each ordered
  ID list, held-out file hash and validation lineage. DataFrame index IDs are provenance,
  not customer identifiers and not predictors.
- Manifest: execution/experiment/parent IDs, four ordered model names, actual MLflow model
  URIs, child run IDs and shared held-out/config/split hashes.
- Evaluation: four metric dictionaries, confusion matrices, eligibility, winner, quality
  policy and lineage.
- Registration: explicit registered/not_registered status, version or null, actual source
  run/URI, metrics and retry key/reuse flag.

These hashes detect accidental content changes; they are not signed authenticity proofs.
CSV is simple and inspectable for this dataset. Parquet is an alternative for larger or
more complex typed data. No DataFrame or fitted Python object is passed between pods.

## Tracking and model lineage across processes

```mermaid
sequenceDiagram
    participant T as Train process
    participant E as Evaluate process
    participant R as Register process
    participant M as MLflow
    T->>M: Create parent execution run
    T->>M: Create four nested child runs and log models
    T->>M: End contexts; tag awaiting_evaluation
    T-->>E: Manifest with explicit run IDs and URIs
    E->>M: Resume parent and each child by run_id
    E->>M: Load actual model URI, log metrics to that child
    E->>M: Log comparison to parent; tag awaiting_registration
    E-->>R: Evaluation artifact
    R->>M: Resume parent; check retry key / register winner
    R->>M: Query version, load registered model, log result
    R->>M: End context; tag workflow_state=completed
```

An active MLflow context is process-local. The application never assumes that stage 2's
context remains active in stage 3. Context managers explicitly end each resumed run;
exceptions set the active run to FAILED. Between stages, MLflow may show FINISHED because
a context has ended, while `workflow_state` still says awaiting evaluation/registration.
The KFP run status is the orchestration authority. A failure before a resumed context
(for example an input hash mismatch) fails the KFP task but cannot automatically mark all
previous MLflow runs failed. No failure-handler infrastructure was introduced.

## Responsibilities

Kubernetes schedules task pods. KFP decides task readiness from artifact dependencies,
manages launcher execution, stores task state and publishes KFP/MLMD artifact lineage.
MinIO holds KFP artifact bytes through the launcher. MLflow owns experiment parameters,
candidate models, metric artifacts and registered versions. The existing MLflow server
uses its own configured artifact persistence; this implementation does not reconfigure
it or assume that its artifacts are in the KFP bucket.

A Python function call returns an in-memory object in one interpreter. A KFP component
specifies a container execution whose output must survive process and pod termination.
A control-only `.after()` creates an ordering constraint. Here artifact arguments create
both ordering and concrete producer/consumer data relationships. No `.after()` is needed.

Dataset artifacts describe examples and labels. Model artifacts contain fitted encoder,
scaler and classifier state. Training fits that state using only training rows; evaluation
uses it without refitting on held-out rows. Tracking records what happened; registry
creation gives an approved candidate a named version. Pipeline success means software
steps finished, whereas model quality is the gate decision. All-rejected runs complete
successfully with a clear non-registration result.

## Failure, retry, caching and resource decisions

All caching is disabled. Training, evaluation and registration write externally to MLflow;
replaying a cached output could hide those side effects. Validation is deterministic for
fixed image/data/config but is also uncached to ensure visible fresh checks and provenance.
Immutable external data references would be needed before safely enabling validation caching.

No automatic task retries are configured. Training retries could create new parent/candidate
runs. Evaluation retries resume the same IDs and overwrite named artifacts/log new metric
history; they do not create new candidates. Registration first searches every version page
for a matching SHA-256 key, winner URI and source run. The key includes registry name,
parent run and model URI. Tags are supplied at version creation, avoiding a separate
post-create tagging gap. Sequential retries are tested to reuse the version. Simultaneous
registrations can race between lookup and creation: this is best-effort idempotency,
not a distributed exactly-once guarantee. No existing version is deleted or mutated.

| Stage | CPU request / limit | Memory request / limit |
|---|---|---|
| Validate | 100m / 1 | 256Mi / 1Gi |
| Train | 500m / 2 | 1Gi / 3Gi |
| Evaluate | 250m / 1 | 512Mi / 2Gi |
| Register | 100m / 1 | 256Mi / 1Gi |

KFP launcher overhead and profile quota must also fit the workstation. The versioned
local image uses explicit IfNotPresent, but import must cover every schedulable node.
A private registry is an alternative for multiple nodes; public publishing is not authorized.

## Alternatives intentionally deferred

One task per candidate could parallelize training but adds split distribution, fan-in and
lineage complexity. Four stages preserve a single shared split with the requested four
visible boundaries. Models could be stored as KFP Model artifacts, but keeping actual
models in MLflow preserves the existing model logging/registry contract. Authentication,
production serving, monitoring, CV, a final untouched test set, distributed locking and
large-scale data storage are outside this implementation.

Primary references: [KFP typed artifacts](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/),
[KFP container components](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/container-components/),
[MLflow model logging](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).
''')

signatures = []
for relative in ['src/stages.py','src/component_cli.py','src/artifact_io.py','src/config.py',
                 'src/data_validation.py','src/training.py','src/evaluation.py','src/registration.py',
                 'src/pipeline.py','src/logging_config.py','pipelines/four_stage.py']:
    tree = ast.parse((ROOT / relative).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.FunctionDef):
                suffix = f' -> {ast.unparse(node.returns)}' if node.returns else ''
                header = f'def {node.name}({ast.unparse(node.args)}){suffix}'
            else:
                header = f'class {node.name}'
            signatures.append(f'| [{relative}:{node.lineno}](../{relative}) | `{header}` |')

roles = {
'mlflow':'Full experiment tracking/model registry client and server distribution inherited from Lecture 3',
'mlflow-skinny':'Core MLflow Python tracking/flavor clients', 'mlflow-tracing':'MLflow tracing support; not used directly by this pipeline',
'scikit-learn':'Preprocessing, split, four classifiers and classification metrics', 'numpy':'Numerical arrays used by pandas/scikit-learn',
'pandas':'CSV/DataFrame handling', 'scipy':'Numerical routines used by scikit-learn', 'PyYAML':'Original YAML configuration parsing',
'matplotlib':'Confusion-matrix plots', 'skops':'Safer scikit-learn model serialization; trusted tree internals from original code',
'joblib':'Scikit-learn serialization/parallel utilities', 'threadpoolctl':'Numerical thread-pool coordination',
'cloudpickle':'Python object serialization support', 'SQLAlchemy':'MLflow SQL tracking/registry backend', 'alembic':'MLflow schema support; new isolated local DB only',
'pyarrow':'Tabular serialization/data support', 'fastapi':'Inherited optional serving dependency; API not copied into this implementation',
'pydantic':'Typed validation used by inherited serving/server dependencies', 'pydantic_core':'Pydantic native validation engine',
'uvicorn':'ASGI server dependency; no new server launched', 'starlette':'ASGI web framework dependency',
'Flask':'MLflow web server framework', 'flask-cors':'MLflow HTTP CORS support', 'gunicorn':'MLflow web server worker manager',
'docker':'Python Docker SDK inherited; the build uses Docker CLI instead', 'GitPython':'MLflow Git provenance support',
'gitdb':'GitPython object database helper', 'smmap':'Git object memory mapping', 'requests':'HTTP client used by MLflow/services',
'urllib3':'HTTP connection/pooling support', 'certifi':'TLS certificate bundle', 'charset-normalizer':'HTTP text encoding detection',
'idna':'International domain encoding', 'cryptography':'Authentication/TLS cryptographic support', 'cffi':'Native-library interface',
'pycparser':'C parser for CFFI', 'google-auth':'Authentication dependency; no Google cloud integration configured',
'databricks-sdk':'MLflow integration dependency; no Databricks service configured', 'protobuf':'Structured MLflow/KFP message serialization',
'graphene':'MLflow GraphQL dependency', 'graphql-core':'GraphQL engine', 'graphql-relay':'GraphQL connection helpers',
'Jinja2':'Server template rendering', 'MarkupSafe':'Template escaping', 'Mako':'Migration/template support',
'Werkzeug':'Flask HTTP utilities', 'blinker':'Flask signal support', 'click':'CLI parsing in MLflow tooling',
'python-dotenv':'Environment-file support; no credentials read from .env here', 'python-dateutil':'Datetime utilities',
'packaging':'Version/requirement parsing', 'importlib_metadata':'Distribution metadata support', 'zipp':'Zip metadata support',
'sqlparse':'SQL text utilities', 'pillow':'Plot/image encoding', 'contourpy':'Matplotlib contours', 'cycler':'Matplotlib style cycles',
'fonttools':'Plot font processing', 'kiwisolver':'Plot layout solving', 'pyparsing':'Plot/text parsing',
'prettytable':'Tabular presentation dependency', 'narwhals':'Dataframe interoperability dependency', 'six':'Compatibility helpers',
'cachetools':'Authentication/client caches', 'huey':'Inherited background-task support', 'h11':'HTTP/1 protocol support',
'anyio':'Async concurrency support', 'aiohttp':'Async HTTP support', 'aiohappyeyeballs':'Async connection selection',
'aiosignal':'Async signal helpers', 'attrs':'Dependency data structures', 'frozenlist':'Async dependency collections',
'multidict':'HTTP multivalue mappings', 'propcache':'Async HTTP property caches', 'yarl':'Async URL handling',
'annotated-types':'Type constraint helpers', 'annotated-doc':'Documentation/type metadata', 'typing_extensions':'Typing compatibility',
'typing-inspection':'Runtime typing helpers', 'pyasn1':'ASN.1 encoding dependency', 'pyasn1_modules':'Authentication ASN.1 schemas',
'opentelemetry-api':'MLflow telemetry API', 'opentelemetry-sdk':'MLflow telemetry implementation',
'opentelemetry-proto':'Telemetry protobuf messages', 'opentelemetry-semantic-conventions':'Telemetry naming conventions',
'wcwidth':'Terminal display-width utilities',
}
dependency_table = '\n'.join(f"| `{line}` | {roles.get(line.split('==')[0], 'Inherited transitive dependency; not directly called by application code')} |" for line in (ROOT/'requirements.txt').read_text().splitlines())
write('03_CODE_WALKTHROUGH.md', '''
# Code walkthrough

All paths below are relative to `lectures/lecture-04-kubeflow/bank-marketing-four-stage/`.
The source modules copied from Lecture 3 are unchanged. New code wraps those functions
rather than reproducing classifier or quality-selection implementations.

## Follow one execution from the container boundary

`Dockerfile` chooses Python 3.12, installs the copied exact requirements, creates UID
10001, copies source/config/active data/tests, and sets a harmless help command by default.
KFP overrides CMD with `python -m src.component_cli STAGE ...`. Dependencies are baked in;
there is no pip install when a task starts. `/app` is the image working directory. The
launcher supplies artifact paths that need not resemble the laptop directory. Runtime
writes go to output paths and `/tmp`, not into protected source directories.

`src/component_cli.py:parser` uses argparse subcommands. Required flags declare each
stage's file contract. `--executor-input` is optional only for local-process verification.
`main` configures logging, records RUNNING/SUCCEEDED/FAILED and elapsed seconds, dispatches
the selected stage, then publishes artifact metadata. Exceptions are logged and re-raised,
so a nonzero exit remains visible to KFP. Outputs are not replaced with pretend success.

## src/artifact_io.py

`sha256(path)` reads one-megabyte blocks; `iter(..., b'')` stops at EOF without requiring a
whole dataset in RAM. `read_json` requires an object rather than accepting any JSON shape.
`write_json` creates parent directories, writes a temporary file with finite JSON values
and renames it into place after writing. This prevents consumers observing a partially
written JSON result. It is not a transactional commit across all stage output files.
`check_hash` fails if artifact bytes changed.

`publish_metadata` reads the launcher's executor input, preserves each output artifact's
identity/type/URI and adds its metadata to an ExecutorOutput JSON file. It does not upload
objects itself. KFP's launcher owns that upload. The generated JSON was tested against the
actual KFP protobuf, using synthetic URIs; that test is not MinIO integration evidence.

## src/stages.py — validation

```python
config = load_config(config_path)
check_config(config)
frame = validate_dataset(dataset_path, data['features'], data['target'],
                         data['numeric_features'], data['target_values'],
                         config['training']['test_size'])
```

`load_config` uses `yaml.safe_load`, which loads data without executing arbitrary YAML
objects. Its normal path is relative to the module's project root; an explicit absolute
container path is also valid. An environment tracking URI override is captured if present,
but later stages explicitly use their tracking URI argument.

`check_config` rejects duplicate/empty features, target leakage, reserved row IDs, unsupported
labels/models/selection policy and impossible scalar ranges. It calls the original
`create_pipeline` to reuse feature-partition validation without fitting a model.
`validate_stage` calls the original bank CSV checks, preflights the original splitter,
materializes the numeric-normalized CSV, writes a JSON configuration snapshot and records
source and normalized hashes. A source-byte hash and normalized-artifact hash can differ
because pandas reserializes CSV content; both are meaningful and recorded separately.

`src/data_validation.py` checks required columns, nonempty/missing data, fixed bank
categories, finite numeric conversion, allowed target values, minimum row counts and two
class feasibility. Literal `unknown`, negative balances and `pdays=-1` retain their original
meaning. It is a bank schema, not a universal validator. It does not silently impute values.

## src/stages.py — training

`train_stage` verifies validated-data/config hashes before reading the artifact. It rewires
only the in-memory configuration's data path to that artifact, because the unchanged
`train_model` hashes `config['data']['path']`. The original YAML file is not edited.

```python
split = split_dataset(frame, config)
X_train, X_test, y_train, y_test = split
for name in config['models']:
    result = train_model(frame, config, mlflow_run=child,
                         model_name=name, split=split)
```

The tuple is made once and passed unchanged to every candidate. pandas index values identify
source rows. Ordered train/test IDs and SHA-256 fingerprints are recorded, and overlapping
sets cause failure. The held-out CSV combines X_test, aligned y_test and a reserved row ID.
It contains no fitted object. Test rows are never passed into `.fit`.

`src/training.py:create_pipeline` builds a `ColumnTransformer`: categorical columns use
`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`; numeric columns use
`StandardScaler` for Logistic Regression and passthrough for trees. A scikit-learn Pipeline
puts that preprocessor before the classifier. `.fit(X_train, y_train)` learns both encoder/
scaler and classifier on training data. `.predict(X_test)` applies their already fitted
state. Saving the whole Pipeline preserves the prediction-time transformation contract.

Classifier parameters come from YAML. Logistic Regression max_iter is 1000; Decision Tree
uses gini/depth 3; Random Forest uses 100 trees; Gradient Boosting uses 100 estimators,
learning rate .1 and depth 3. Seed 42 controls split and estimator randomness. Feature
partition/order and YAML candidate order are preserved.

`train_model` logs split sizes, model parameters, a dataset hash and the fitted model in the
active child context. MLflow 3 returns actual `models:/m-...` logged-model URIs in the tested
environment. The manifest preserves those values instead of inventing `runs:/.../model`
paths. Trusted scikit-learn tree types remain explicitly supplied by the original code.

`train_stage` creates one parent and four nested children with `mlflow.parentRunId` lineage,
logs validation/config/split/manifest artifacts to the parent and tags workflow progress.
The run IDs are persisted so a later process can resume them. Context managers close all
runs; no object or context survives between stages.

## src/stages.py — evaluation

`evaluate_stage` checks held-out/config/split hashes and row order. It also checks candidate
order and each candidate's shared held-out hash. It selects only the configured feature
columns, excluding the target and provenance ID. It resumes the explicit parent and child
IDs and verifies every child's parent tag before logging metrics.

```python
model = mlflow.sklearn.load_model(candidate['model_uri'])
metrics = evaluate_model(model, X_test, y_test, config['data']['positive_label'])
eligible = passes_quality_gate(metrics, config)
```

The loaded model was trained in a different process. In Kubernetes it must be downloadable
from the tracking/artifact service by the evaluation pod. This was verified using local
MLflow storage but has not been verified over the cluster network.

`src/evaluation.py:evaluate_model` calculates sklearn accuracy/precision/recall/F1, with
`pos_label='yes'` and zero_division=0. Matrix row/column order is `[no, yes]`. It logs a PNG
and JSON to the active child. Evaluation also puts matrix counts into its output report.
The typed KFP Metrics output stores numeric candidate scores in artifact metadata.

`passes_quality_gate` uses accuracy >= threshold. `select_best_model` filters on the same
threshold then ranks tuples `(f1_score, accuracy)`; Python max retains the first exact tie,
preserving YAML order. No qualified candidate means selected_candidate is null and the
explicit gate is false. Parent artifacts/metrics contain the full comparison.

## src/stages.py — registration

`register_stage` checks configuration integrity and recomputes selection to detect winner/
gate contradictions. A false gate produces `not_registered` and does not call registration.
A true gate verifies source-run parent linkage, calculates a registration key, searches all
version pages, and either reuses a matching version or calls `mlflow.register_model` with
the actual winner URI and creation-time tags. It queries the version, checks source run ID
and loads the registered model before claiming completion. Failure raises an exception.

`find_registered_version` matches key, model URI and source child run. It is a sequential
retry safeguard, not a concurrency lock. The original `src/registration.py:register_model`
is preserved for the copied single-process workflow but not called by the new registration
stage because its interface has no retry-key strategy. The original gate function is reused.

## Preserved single-process code and logging

`src/pipeline.py` remains an executable reference. Its `pipeline_stage` context manager
tracks stage failure and safely attempts failure tags before re-raising. `run_pipeline`
configures console logging and prints a summary; `_run_pipeline` implements the original
single-process validate/train/evaluate/select/register workflow. The new KFP definition
does not invoke it. This is why four task boxes reflect genuine process boundaries.

`src/logging_config.py` uses ContextVar to add execution IDs to logs. A context variable is
process-local, so its value cannot be used as the cross-pod lineage contract. Persisted
MLflow/KFP IDs and artifact hashes provide that contract instead.

## pipelines/four_stage.py

Each `@dsl.container_component` returns a ContainerSpec. The annotated Input/Output
contracts are converted into typed artifact definitions. `.path` produces launcher-local
path placeholders, not a DataFrame. `PIPELINE_IMAGE` is read at compilation time, so changing
an image requires recompilation. `PIPELINE_JOB_ID_PLACEHOLDER` links the training parent to
the actual KFP execution identifier when deployed.

`bank_marketing_four_stage` passes producer outputs into consumer inputs. KFP sees those
edges and schedules consumers only after needed artifacts are ready. `configure` sets
resource requests/limits and disables caching. It uses the public `platform_config`
Kubernetes imagePullPolicy field directly rather than adding a kfp-kubernetes dependency;
the emitted extension and IfNotPresent value were inspected. Live backend compatibility
is still unverified. Compilation writes the versioned YAML with both IR and platform docs.

## Verification and operational scripts

- `scripts/local_verify.py`: creates NEW storage with exist_ok=False, executes four
  subprocesses, makes the original input path unavailable after validation, checks lineage,
  repeats registration only and retrieves the registered model. It is deliberately not a
  KFP or MinIO emulator.
- `scripts/verify_spec.py`: asserts task names, typed edges, images, resources, cache settings
  and pull policy against the actual compiled package.
- `scripts/collect_environment.py`: bounded read-only status/log commands plus the authorized
  unique-image Docker build attempt. No Secrets, database contents or resets are requested.
- `scripts/submit_run.py`: optional authenticated endpoint submission into admin, unique UTC
  run name, 30-minute monitoring timeout and persisted API result. It never cancels/deletes
  a timed-out run. It has not been executed against a live endpoint in this session.
- `scripts/write_reports.py`: generates these reports from retained evidence. It does not
  convert a local result into a Kubernetes success claim.
- `tests/test_stages.py`: validation failures, reproducibility/leakage, metrics/selection,
  registration decisions/source linkage/retries/errors and output metadata integrity.

## Source signature and line reference index

Line numbers correspond to the delivered source. Links point to full files so learners can
inspect surrounding code. The signatures below are extracted from Python ASTs.

| Path and line | Signature |
|---|---|
''' + '\n'.join(signatures) + '''

## Dependencies: direct use versus inherited support

`requirements.txt` is byte-identical to Lecture 3, preserving its pinned environment.
Some packages support MLflow servers or the optional API and are not directly needed by
these components. Removing them safely would require a separately tested runtime lockfile;
this task prioritizes compatibility. The compiler is separate and pins KFP 2.17.0,
kfp-pipeline-spec/server-api 2.17.0, protobuf 6.33.6 and PyYAML 6.0.3.

| Exact application requirement | Role |
|---|---|
''' + dependency_table + '''

Use `docs/evidence/application-environment.txt` and the successful offline-install log
for actual installed versions. Python package installation is not a Docker build test.
''')

write('04_NEW_DATASET_MIGRATION_GUIDE.md', '''
# New dataset migration guide

Start with a new directory and a local scikit-learn baseline. Reuse architecture only
after defining the new prediction question. Do not alter the preserved bank lectures.
This guide describes changes to a future copy; it does not claim a second dataset was run.

## 1. Inspect and define the problem

Read a small sample and the dataset documentation. Record source, permission to use it,
collection date, row meaning, units, label meaning and prediction-time assumptions.
Inspect shape, dtypes, unique values, duplicates, missing counts, target counts, numerical
quantiles and any group/time columns. Inspect delimiters and encoding before assuming CSV
parsing worked. Avoid putting private/raw customer records into reports unnecessarily.

```python
frame = pd.read_csv('data/new.csv')
print(frame.shape)
print(frame.dtypes)
print(frame.isna().sum())
print(frame['target'].value_counts(dropna=False))
print(frame.select_dtypes('number').describe())
```

Choose the target from the intended decision, not because it happens to be the last column.
A categorical answer is classification; a continuous amount is regression. Numeric labels
such as 0/1 can still represent classification. Record exactly what the positive label means.
Remove target-derived columns and information unavailable at the moment of prediction.
An ID can leak identity; it is not automatically a useful predictor.

## 2. Define a validation contract

List required columns, permitted extras, nullability, numeric units/ranges, category sets,
label values and minimum sample requirements. Distinguish invalid measurements from valid
sentinels. In the bank dataset, negative balance and pdays=-1 are valid. Another dataset
needs its own rules. Numeric-looking identifiers should not automatically be standardized
as continuous quantities.

Do not blindly discard outliers. Investigate whether they are errors or rare valid cases.
Measure imbalance and its implications for splitting and metrics. A schema should fail
clearly before training rather than convert invalid categories into silent success.
Use tests for missing columns, malformed types, invalid labels/categories, missing values,
empty files and infeasible splits. The bank validator rejects missing values; if the new
problem needs imputation, make that decision explicit.

## 3. Prevent leakage and choose splitting

Random stratification works for independent binary examples. Use chronological splits
for future prediction, group-aware splits for repeated customers/patients/devices, and
appropriate sampling for spatial or related examples. Do not put the same person's
related observations in both train and test simply to preserve class ratios.

Split before learning preprocessing, feature selection or imputation. Fit encoders,
scalers and imputers inside a sklearn Pipeline on training rows only. Map unseen categories
with a deliberate policy. Keep record IDs as artifact provenance, never accidental inputs.
Reserve an untouched final test set after candidate selection; the bank implementation uses
one held-out split for selection and would need extending for that stronger design.

All candidates must share the same evaluation rows and feature ordering. Persist ordered
row IDs, seed, split strategy, ratio, data/config hashes and label distributions. A seed
helps reproducibility but cannot compensate for changing data order, library versions or
nondeterministic algorithms.

## 4. Adapt preprocessing and algorithms

Numerical values may need standardization for linear/distance-based models; trees often
do not. Low-cardinality nominal categories may use one-hot encoding; ordered categories
need a justified ordering. Missing values can use train-fitted imputation. High-cardinality
encoding and feature engineering require careful leakage control, especially target
encoding. Do not fit transformations on the combined training/evaluation dataset.

Start with a DummyClassifier/DummyRegressor baseline. Choose algorithms suitable for sample
size, feature representation, interpretability and latency. Keep candidate settings in YAML.
Changing the task type requires changing estimator constructors, not merely editing targets.

## 5. Metrics and quality policy

For imbalanced binary classification consider precision, recall, F1, PR-AUC and a threshold
policy appropriate to the cost of false positives/negatives. Accuracy is useful context,
not universal approval. For multiclass classification specify macro/micro/weighted averaging
and all confusion-matrix labels. For regression use MAE/RMSE/R² as appropriate and rank
lower-is-better metrics correctly. Define the gate before inspecting test scores and compare
to the baseline. Operational latency, data validity and segment performance may also matter.

A no-winner result must remain explicit and must never trigger registration. A successful
pipeline should not be confused with a model that meets the problem's quality requirements.

## 6. YAML changes and bank-specific constraints

A future binary classification configuration might contain:

```yaml
project:
  name: wine-binary
  random_state: 42
data:
  path: data/wine_binary.csv
  target: wine_class
  target_values: ['yes', 'no']
  positive_label: 'yes'
  features: [alcohol, malic_acid, color_intensity]
  numeric_features: [alcohol, malic_acid, color_intensity]
  categorical_features: []
training:
  test_size: 0.2
evaluation:
  minimum_accuracy: 0.8
  selection_metric: f1_score
mlflow:
  tracking_uri: http://mlflow.mlops.svc.cluster.local:5000
  experiment_name: wine-binary-four-stage
  registered_model_name: wine-binary-model
```

Supply a complete `models` mapping as well. These feature names are illustrative; use
actual columns. The example threshold is an exercise choice, not a claim of appropriate
quality. Keep experiment/model names distinct from bank history.

The current implementation explicitly requires four bank classifiers, yes/no labels,
positive yes and F1 selection in `check_config`. Confusion matrices in copied evaluation
code use fixed `[no, yes]`. The copied validator has bank category defaults and exactly-two-
class rules. Update those intentionally in a new copy; do not weaken bank rules here.
If keeping another pair of labels, generalize matrix labels and validation in addition to
positive_label. The schema version should change when artifact semantics change.

## 7. Worked binary example using wine data

After installing the copied application requirements in the new project:

```python
from sklearn.datasets import load_wine
wine = load_wine(as_frame=True)
keep = wine.target.isin([0, 1])
frame = wine.data.loc[keep].copy()
frame['wine_class'] = wine.target.loc[keep].map({0: 'no', 1: 'yes'})
frame.to_csv('data/wine_binary.csv', index=False)
```

The question is class 1 versus class 0; original class 2 is deliberately excluded. This
is not a three-class classifier. Populate YAML features from all non-target columns,
set all predictors numeric and set categorical_features to an empty list. Replace
EXPECTED_CATEGORIES with the new dataset's contract, and replace bank fallback numeric
columns. The original preprocessing supports an empty categorical selection.

The four-stage contract can stay structurally identical:

```python
validated: Output[Dataset]       # Wine rows after wine-specific validation
configuration: Output[Artifact] # Wine configuration snapshot
heldout: Output[Dataset]         # Wine test rows, wine_class and source row IDs
manifest: Output[Artifact]       # Four wine model URIs and run IDs
evaluation: Output[Artifact]     # Positive=yes metrics and winner/gate
registration: Output[Artifact]   # Wine model name/version or non-registration
```

Change model/experiment/pipeline names and the image reference. Add tests using actual
wine rows, fit every candidate, check serialization and run tracking behavior. Keep
existing bank-specific tests as references outside the new active suite or adapt their
fixtures; do not hide failures by assuming only one smoke test is sufficient.

For regression, a corresponding interface could still pass Dataset/Artifact types, but
its evaluation content must be different:

```python
# Pseudocode, not the delivered bank implementation:
model = make_regressor(config).fit(X_train, y_train)
metrics = {'mae': mean_absolute_error(y_test, model.predict(X_test))}
eligible = metrics['mae'] <= config['evaluation']['maximum_mae']
winner = min(eligible_candidates, key=lambda item: item['metrics']['mae'], default=None)
```

Remove classifier-specific validation/stratification, replace algorithms and selection
policy, and change the quality-gate object/schema. Typed artifacts describe the storage
contract; they do not automatically make bank business logic generic.

## 8. Adapt MLflow and runtime integration

Log original/validated data hashes, config, split, candidate settings and preprocessing.
Create a parent run and child candidates. Persist actual model URIs returned by the model
logging call. Evaluation should resume child IDs and load those exact artifacts in a new
process. Registration should match source run/URI and use an idempotency key that identifies
the selected artifact, not merely the latest model name.

Test locally before scheduling: validate, train, evaluate and register with isolated new
MLflow storage. Remove the source input path after validation in the test harness to expose
hidden reloads. Verify loaded-model predictions match before/after serialization. Inspect
all-rejected and error cases with unit tests rather than full repeated pipelines.

Build a unique image such as `wine-kfp-four-stage:v1`; do not overwrite bank tags:

```bash
docker build -t wine-kfp-four-stage:v1 .
docker run --rm wine-kfp-four-stage:v1 python -m unittest discover -s tests
docker save wine-kfp-four-stage:v1 -o /tmp/wine-kfp-four-stage-v1.tar
microk8s ctr image import /tmp/wine-kfp-four-stage-v1.tar
PIPELINE_IMAGE=wine-kfp-four-stage:v1 .venv-kfp/bin/python pipelines/four_stage.py
```

A properly installed separate compiler environment is assumed in that last command.
Verify the compiled image and typed edges before upload. Submit into your permitted profile,
record run/workflow IDs, inspect all tasks, retrieve artifacts and compare KFP API state
with workflow state. Check registry source/version and make a sample prediction. Merely
seeing a green MLflow run cannot verify KFP's metadata or object-store transport.

## Reusable infrastructure versus business logic

Reusable: container process pattern, KFP artifact plumbing, JSON/hash helpers, explicit
tracking IDs, image import workflow, resource settings, evidence collection and operations
playbook. Dataset-specific: schema, parsing, target meaning, split policy, features,
preprocessing, candidate algorithms, metrics, quality gate and serving input contract.
Do not bring the bank FastAPI schema into an unrelated prediction service.

## Practical completion checklist

- [ ] Define task, target, units, positive class and inference-time information.
- [ ] Inspect source/license/schema, missing values, imbalance, outliers and duplicates.
- [ ] Choose and justify a split that prevents group/time leakage.
- [ ] Create and test a dataset-specific validator.
- [ ] Keep preprocessing inside train-fitted pipelines; compare a baseline.
- [ ] Update YAML, algorithms, metrics, gate and artifact schema together.
- [ ] Test actual model save/load and independent-process evaluation.
- [ ] Confirm common held-out IDs/hashes and explicit parent/child source lineage.
- [ ] Test failed gates and sequential registration retry safety.
- [ ] Build/import a new image; inspect four compiled task contracts.
- [ ] Run once in the permitted namespace; retain failures and successes.
- [ ] Retrieve KFP artifacts, model artifacts and the registered version.
- [ ] Compare API/workflow/metadata state and document limitations.
- [ ] Define serving/monitoring separately after training is correct.
''')

write('05_DEBUGGING_AND_OPERATIONS.md', '''
# Debugging and operations

Follow the failing layer: input contract → local process → tracking/model storage →
container runtime → scheduling → launcher/object store → KFP API/MLMD. Do not reset a
cluster because the UI reports a failure while a workflow succeeded.

## Local commands inside the new project

```bash
export MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/local_verify.py
# Existing Lecture 4 compiler is read-only and currently available:
../.venv/bin/python pipelines/four_stage.py
../.venv/bin/python scripts/verify_spec.py
```

The local verifier creates a fresh timestamped storage directory and refuses existing
storage. It trains the real dataset; do not rerun it unnecessarily just to test a gate.
Use unit tests for negative cases. Reports distinguish local from cluster evidence.

For a portable compiler environment, create `.venv-kfp` in this directory and install
`requirements-kfp.txt`. Use that interpreter for compilation/submission. Do not modify
Lecture 4's existing environment or import application dependencies into it.

## Build/import and smoke test when runtime access is permitted

```bash
docker build -t bank-marketing-kfp-four-stage:v1 .
docker run --rm bank-marketing-kfp-four-stage:v1 python -m src.component_cli --help
docker run --rm bank-marketing-kfp-four-stage:v1 python -m unittest discover -s tests
docker save bank-marketing-kfp-four-stage:v1 -o /tmp/bank-marketing-kfp-four-stage-v1.tar
microk8s ctr image import /tmp/bank-marketing-kfp-four-stage-v1.tar
microk8s ctr images ls | rg bank-marketing-kfp-four-stage
```

A uniquely named standalone smoke pod is authorized as a non-destructive execution:

```bash
microk8s kubectl run bank-four-stage-image-smoke-UNIQUE -n admin \
  --image=bank-marketing-kfp-four-stage:v1 --image-pull-policy=IfNotPresent \
  --restart=Never --command -- python -m src.component_cli --help
microk8s kubectl logs bank-four-stage-image-smoke-UNIQUE -n admin
microk8s kubectl get pod bank-four-stage-image-smoke-UNIQUE -n admin
```

Replace UNIQUE with a new UTC suffix. Retain the pod as evidence. Deleting existing
resources is outside current authorization; do not add an automatic cleanup command.
These runtime commands were not successful in this restricted session.

## Submit and inspect KFP

Use an existing authenticated KFP endpoint/profile. `KFP_HOST` identifies the endpoint;
optional `KFP_TOKEN`/`KFP_COOKIES` are read without printing/saving them. Do not paste
credentials into reports or commit them. Authentication mechanisms depend on the installed
Charmed Kubeflow configuration; this task did not change ingress or auth.

```bash
.venv-kfp/bin/python scripts/submit_run.py --namespace admin --timeout 1800
microk8s kubectl get workflows -n admin
microk8s kubectl get pods -n admin -o wide
microk8s kubectl describe pod POD_NAME -n admin
microk8s kubectl logs POD_NAME -n admin -c main --tail=200
microk8s kubectl get events -n admin --sort-by=.lastTimestamp
```

The submit script requires KFP_HOST or --host. Use actual pod/container names. In the UI,
select admin, open the named experiment and run, inspect all four task states/durations,
input/output artifacts, their URIs and task logs. Retain the API JSON produced by the script.
A timeout does not cancel the run; check its actual state before submitting another.

To compare an old run, read the KFP API run object for its ID and find the workflow using
that ID's run labels/name. Do not assume every workflow listed in admin belongs to this task.

## Artifact and MLflow inspection

Download an output artifact through the authenticated KFP UI or the configured authorized
artifact client. Verify its content hash against validation/split metadata. Compare its
reported URI with object-store persistence. Do not list/delete arbitrary MinIO objects or
extract bucket credentials just to troubleshoot.

Open shared MLflow with a local tunnel:

```bash
microk8s kubectl port-forward -n mlops svc/mlflow 5001:5000
```

Browse http://127.0.0.1:5001. Find the parent tagged with the KFP execution ID, inspect its
four child runs and confirm source URIs, data hashes, metrics and selected run/version.
Inside Kubernetes, the tracking URI is the mlflow.mlops service, not localhost.
A registry version must be queryable and its model loadable before declaring serving-ready
artifact retrieval. This does not automatically deploy an inference API.

## Common symptoms

| Symptom | Inspect | Safe next action |
|---|---|---|
| Validation fails | Schema, category/target, missing values, CSV delimiter | Correct dataset/config in a new version; preserve failure logs |
| Artifact hash mismatch | Wrong producer output, mutation, stale config | Compare expected/actual bytes and compiled image; do not bypass check |
| Missing manifest/held-out file | Producer exit, output paths, launcher upload | Inspect upstream logs and artifact URI |
| Model URI cannot load | URI from manifest, MLflow artifact access, dependency versions | Test download from evaluation runtime; keep actual URI |
| MLflow HTTP 403/timeouts | DNS, service endpoints, namespace network/mesh policy | Collect connectivity evidence; request approval for shared-policy changes |
| ImagePullBackOff | Imported image/tag, node location, registry reachability | Import unique image to eligible nodes or authorized private registry |
| Pending/OOMKilled | Quota, CPU/RAM, storage/events | Adjust new task resources when justified; do not reset cluster |
| KFP succeeded but model not registered | quality_gate and selected_candidate | Accept explicit rejection or revise justified training policy |
| Workflow succeeded but API/UI missing context | MLMD/context/API consistency | Follow metadata report; no database reset |
| Registration retry duplicates | Key/tags, source URI, parallel executions | Investigate race; serialize registration or design approved locking |
| Snap/socket errors only in Codex | Session confinement and access policy | Do not infer host health or alter shared services |

## Metadata investigation and recovery boundaries

Read current pod readiness/restarts and bounded logs from metadata-writer, MLMD gRPC,
MySQL and ml-pipeline. Compare old versus newly created run/context/artifact retrieval.
Inspect warnings about publish, context lookup and DAG-state updates with timestamps.
A subprocess exit 0 or publish-success log may precede a later metadata/API retrieval error.
See [metadata diagnosis](08_METADATA_DIAGNOSIS.md) for evidence limitations and hypotheses.

Safe under this task: read statuses/logs/events, query existing runs/models, create isolated
code/test artifacts, build a unique image, create named execution runs/pods, inspect output
artifacts through existing permissions. Do not print Secrets or raw credential-bearing
configuration into evidence.

Requires explicit approval: modifying/deleting existing Kubernetes resources, PVCs/databases/
MinIO objects, restarting/reconfiguring shared services, changing Charmed applications,
installing cluster-wide services, destructive Git actions or external image publishing.
If diagnosis indicates infrastructure repair, first document the specific resource,
configuration/version change, evidence, expected effect and preservation/rollback plan.
No such repair was applied in this task. Deleting a database is not a diagnostic experiment.
''')

write('06_LEARNING_PLAN.md', '''
# Learning plan — start from the evidence

The current implementation has a verified local workflow and compiled KFP spec. Begin
there. A live KFP/MinIO/metadata session remains an operational exercise until runtime access
is available. Every session follows **Observe → Explain → Inspect Code → Modify → Break
Safely → Debug → Restore → Reproduce**. Perform experiments only in this new directory or
another isolated copy. Keep original lectures, shared runs/models and infrastructure intact.

## Session 1 — reproduce and read the outcome

Observe the retained local logs, candidate scores and registration result. Explain what
PASS_LOCAL_ONLY means and why it differs from a cluster PASS. Inspect local_verify and its
four subprocess commands. Modify only the output location in a fresh experiment. Break a
copied input path safely; debug FileNotFoundError; restore it and reproduce validation.
Questions: Which evidence proves four processes ran? Which evidence is still absent?
Why is the local registry's version 1 not the earlier shared version 1?

## Session 2 — data contract and validation

Observe validation counts/hashes. Explain feature versus label, categories, numeric values,
missing values and stratification. Inspect YAML, validator and validate_stage. Modify one
copied CSV row to an invalid job or missing age. Debug the precise exception; restore and
rerun only validation. Explain why pdays=-1 and negative balance remain valid. What does a
source hash prove, and why can the normalized CSV hash differ?

## Session 3 — train-only preprocessing and a shared split

Observe split row IDs and counts. Explain pandas column selection/index alignment and
sklearn Pipeline/ColumnTransformer. Inspect split_dataset/create_pipeline/train_model.
Modify the seed in a disposable configuration and observe changed row-ID fingerprints.
Break the numeric/categorical partition; debug the early error; restore and rerun unit tests.
Explain which exact `.fit` call learns encoder/scaler state. Why is held-out duration still
a prediction-time concern? Why does one split tuple improve candidate comparability?

## Session 4 — evaluate saved artifacts

Observe child model URIs and confusion matrices. Explain loading a fitted object without
refitting, accuracy/precision/recall/F1 and positive yes. Inspect evaluate_stage/evaluate_model.
Modify a test-only prediction array; calculate the expected matrix by hand. Break an artifact
hash in a temporary file; debug integrity failure; restore. Reproduce metric tests. Explain
why a models:/m-ID cannot safely be replaced with an invented runs:/run/model path.

## Session 5 — quality gate and retry-safe registry

Observe winner and registration-retry evidence. Explain F1 ranking, accuracy ties, exact-tie
order and all-rejected success. Inspect select_best_model/register_stage/version search.
Modify test metrics so every candidate is rejected. Break parent linkage in a mocked run;
debug rejection before registration; restore tests. Explain why lookup/create can race and
what exactly was proven by the real sequential retry. Why isn't model registration deployment?

## Session 6 — image, launcher and typed artifacts

Observe the compiled YAML and Dockerfile. Explain image versus container/pod, Docker versus
containerd stores, parameter versus artifact, control versus data dependency. Inspect four
container components, `.path` placeholders and ExecutorOutput. Modify the compile-time image
reference in a disposable compilation. Break one expected edge in a temporary YAML and run
spec inspection; debug and restore. Recompile from Python rather than hand-maintaining YAML.
Identify which code uploads to MinIO: the launcher, not the CSV-reading Python function.

## Session 7 — live run and metadata when access permits

Observe a unique smoke pod, then one real four-stage run. Explain each task's scheduling,
logs, duration and artifact URI. Inspect KFP API data and workflow status together. Modify
only new-run parameters; avoid shared service experiments. Break a copied task's input
contract with unit/local tests rather than deleting storage. Debug safe failure boundaries,
restore the implementation and reproduce. Explain why a successful Argo workflow or MLflow
version does not prove a system.PipelineRun context can be retrieved.

## Session 8 — migrate a new dataset and production review

Observe the wine example/schema plan in the migration guide. Explain task type, target,
split strategy, preprocessing, baseline and quality policy. Inspect reusable versus bank-
specific functions. Modify a separate new project. Break one new schema/label assumption
safely, debug and restore. Reproduce local boundaries before deploying. Present a report
with actual evidence and limitations. Discuss untouched final tests, dataset versions,
monitoring, authentication, resource budgets and concurrency guarantees as future work.

Assessment: the learner must trace one row and one fitted model through all four stages,
identify every persisted ID/hash, justify the split and gate, and show an observable failure
and repair without altering shared history. Copying commands alone does not meet that goal.
''')

copied = json.loads((EVIDENCE / 'copied_files.json').read_text())
copied_table = '\n'.join(f"| `{name}` | `{str(Path(name).relative_to(Path('lectures/lecture-03-mlflow')))}` | unchanged |" for name in copied)
write('07_SOURCE_CHANGELOG.md', f'''
# Source changelog and preservation

The target directory did not exist on inspection. No existing work needed overwriting.
Protected lecture hash check: **{PRESERVATION['status']}**, {PRESERVATION['protected_files']}
regular source/data/document files; changed: {len(PRESERVATION['changed'])}; missing:
{len(PRESERVATION['missing'])}. Evidence: `protected_before.json` and `source-preservation.json`.
Virtual environments and Python caches were excluded from source fingerprints and were
not copied or modified in protected lectures.

## Copied files

| Original relative workspace path | Destination relative new-project path | File changes |
|---|---|---|
{copied_table}

Only `balance.csv` was copied. Other example CSVs, model files, local databases, Compose,
API modules, old unit tests, virtual environments and caches were not copied. New tests
focus on the four-stage contracts. The original single-process pipeline remains as a
reference inside the new copy, and all original lectures remain unchanged.

## New files

- `PLAN.md`, `README.md`, `CODEX_HANDOFF.md`.
- `src/artifact_io.py`, `src/stages.py`, `src/component_cli.py`.
- `pipelines/four_stage.py` and `pipelines/compiled/bank_marketing_four_stage_v1.yaml`.
- Dedicated `Dockerfile`, `.dockerignore`, `.gitignore`, `requirements-kfp.txt`.
- `tests/test_stages.py`.
- Scripts for local verification, spec inspection, environment evidence, optional run
  submission and report generation.
- Nine requested reports in `docs/`, evidence logs/JSON/environment fingerprints and
  concise copies of actual local stage outputs.
- Ignored `.venv/` and `.verification/` contain the isolated Python environment and newly
  created local MLflow storage. Cached wheel links are local verification aids, not
  deployment dependencies or container contents.

## Reused functions

Configuration: load_config. Validation: validate_dataset. Training: calculate_file_hash,
create_pipeline, split_dataset, train_model. Evaluation: evaluate_model, select_best_model.
Quality: passes_quality_gate. Logging: setup_logging, run_logging_context.
No copied function was edited. The original registration helper and pipeline coordinator
remain available for reference but are not the new KFP orchestration implementation.

## Adaptations and behavioral changes

The new stage wrappers replace in-memory transitions with typed persisted artifacts.
Training's in-memory data path points to the validated artifact, so the reused hash logic
cannot reopen the source CSV. Frozen configuration and source/artifact hashes provide
cross-process consistency. Split IDs and held-out identity are explicit. Evaluation resumes
recorded child contexts and reloads actual MLflow model URIs. Registration adds creation-time
retry tags, lookup and retrieval verification instead of unconditional repeated registration.

Bank model settings, preprocessing, seed, split, positive class, threshold and F1/accuracy
ranking are unchanged. Added fail-fast checks reject target-as-feature, duplicate features,
reserved row IDs, unsupported config and splitter errors. They are documented contract
checks, not hidden changes to the valid bank dataset's quality policy. The all-rejected
path still succeeds without registration.

All stages disable caching. No automatic retries were configured. Run contexts end at
stage boundaries; workflow_state tags distinguish partial progression from whole-workflow
completion. Stage exceptions remain visible; earlier completed runs may remain FINISHED
if a later failure occurs before an active tracking context.

## Compatibility and limitations

Python 3.12 and exact original application dependencies were tested locally. KFP 2.17.0
compiled the spec; current backend version/extension compatibility is NOT VERIFIED.
The explicit Kubernetes imagePullPolicy extension was inspected, not tested live. Actual
MLflow logged-model URIs in local evidence use models:/m-ID semantics. The bank-only validator,
yes/no matrices and four-classifier config must be intentionally generalized for another task.

Concurrent registration may race; sequential retry is verified. Training retry may create
additional parent/candidate runs. CSV schemas are JSON-versioned but not a formal external
schema registry. Models remain in MLflow, not duplicated into KFP Model artifacts. There is
no serving deployment, production monitoring, CV or untouched final test set. `duration`
remains a post-call feature. The image was not built/imported here and live metadata health
was not diagnosed because access was denied.

No shared Kubernetes resource, Secret, database, PVC, MinIO object, registry version or
existing run was deleted, restarted or reconfigured. The only database initialization was
for NEW isolated local verification storage inside this task's directory.
''')

write('08_METADATA_DIAGNOSIS.md', '''
# Missing system.PipelineRun context diagnosis

## Reported historical symptom

User-provided historical run ID: `78cf2a06-1afb-4e3b-a767-ef2844eb5cc7`.
The reported lookup error was:

```text
Cannot find context with {"typeName":"system.PipelineRun","contextName":"78cf2a06-1afb-4e3b-a767-ef2844eb5cc7"}: Cannot find specified context
```

The user reported Argo Succeeded, successful subprocess exits, publish-success messages,
DAG-state update attempts and running metadata-writer/MySQL/metadata-gRPC/ml-pipeline
components. These are historical observations supplied by the user. They were not
independently retrieved during this task.

## Actual investigation performed

Bounded read-only commands attempted current node/pod/workflow/MLflow health and logs from:
`kfp-metadata-writer-0`, `mysql-db-0`, `deployment/metadata-grpc-service`, and
`deployment/ml-pipeline` in kubeflow. Every cluster request failed before retrieving state:
`dial tcp 192.168.1.10:16443: socket: operation not permitted`.
Snap-wrapped MicroK8s/Juju also refused session execution. Direct kubectl bypassed only
the wrapper; it did not bypass the sandbox. The KFP internal DNS health probe failed name
resolution. Current logs, restarts, contexts, database connectivity and artifact persistence
could therefore not be inspected. See environment-checks.json and corresponding raw logs.

| Question | Finding |
|---|---|
| Current backend/MLMD/MySQL/writer healthy? | NOT VERIFIED |
| Historical run has missing context now? | NOT VERIFIED; historical symptom supplied |
| Only old runs affected? | NOT DETERMINED |
| New runs affected too? | NOT DETERMINED; no new live run submitted |
| Argo versus KFP API/UI state consistent? | NOT VERIFIED |
| KFP artifact bytes persist/retrieve? | NOT VERIFIED |
| Root cause established? | NO |

The local workflow/model registry was healthy in newly isolated storage. It did not use
MLMD, KFP's database, MinIO or the shared MLflow server, so it cannot resolve this issue.

## Why the signals can disagree

Argo workflow success describes container/workflow execution. MLflow model registration
records a different service's operation. MLMD context lookup and KFP API/UI retrieval
operate on metadata identities and storage. Success in either earlier layer does not
prove the system.PipelineRun context exists or is retrievable by the service querying it.
A launcher can log a successful subprocess or publication operation and still encounter a
subsequent DAG-state/context lookup error. Logs must be correlated by run ID and timestamp.

Running pods only prove processes are running/ready according to their probes; they do
not demonstrate end-to-end correctness for the affected run's metadata.

## Hypotheses to test, not confirmed causes

1. A context creation/publication operation failed or was interrupted earlier in the run,
   despite later task-success messages. Seek the first relevant context/create error.
2. Services read different metadata endpoints/databases/configuration. Compare managed
   endpoint identities without exposing credentials; do not assume misconfiguration.
3. A restart/version transition left old-run metadata unavailable or incompatible while
   newer metadata still works. Compare timestamps, deployed image versions and old/new runs.
4. UI/API lookup uses a run/context identity that differs from the persisted execution's
   identity, or a version-specific retrieval defect is involved. Compare API run ID,
   workflow labels and actual MLMD context associations.
5. Retrieval occurred before expected context publication completed. Re-query an existing
   affected run read-only and compare ordered timestamps before calling it permanent.

No evidence currently ranks these into a confirmed or most-likely root cause. Database
corruption, credential failure or MinIO data loss must not be asserted from the reported
message alone.

## Evidence collection needed when access is available

- Record current pod readiness/restart counts and relevant deployment image versions for
  API, writer, metadata gRPC and MySQL, without Secret/environment dumps.
- Retain bounded current/previous container logs where accessible; correlate context/run
  IDs and timestamps with workflow/task completion and publish messages.
- Query the old run via the authenticated KFP API and read its workflow status.
- Execute one identifiable new four-stage run once image/runtime checks pass. Query its
  KFP status, context/artifact associations and download its artifacts.
- Use authorized read-only MLMD inspection to check whether the PipelineRun context and
  execution/artifact associations exist for old and new run IDs. Never retrieve database
  passwords into the report. Do not write directly into metadata tables.

If new runs execute, transfer artifacts and expose complete metadata, the historical issue
need not block this migration. If new metadata is also broken, mark deployment partial even
if Argo and MLflow succeeded.

## Proposed remediation boundary

First enable an execution environment that already has permitted cluster/runtime access;
that is a tooling/access prerequisite, not a Kubeflow service repair. No specific cluster
repair is justified yet. Any evidence-supported repair involving managed endpoint/version
changes, restarts or database work must be proposed separately with the affected resource,
expected result, preservation plan and rollback strategy, then explicitly approved.
No restart, Secret extraction, Charmed configuration change, database reset, PVC deletion,
MinIO cleanup or speculative migration was performed. Existing history is preserved.
''')

commands = json.loads((EVIDENCE/'environment-checks.json').read_text())
command_table = '\n'.join(f"| {c['utc']} | `{' '.join(c['command'])}` | {c['exit_code']} | [log](evidence/{c['log']}) |" for c in commands)
local_command_table = '\n'.join(f"| {s['stage']} | `{' '.join(s['command'])}` | {s['exit_code']} | {s['seconds']:.3f} |" for s in LOCAL['stages'])
write('09_COMMANDS_AND_EVIDENCE.md', f'''
# Actual commands and retained evidence

All project operations were executed under
`{ROOT}` unless a command explicitly names a different workspace-relative path.
No credentials, Secrets or sensitive environment values are included. Python source/env
hashes and model/run identifiers are not credentials.

## Infrastructure checks (actual UTC timestamps)

| UTC | Executed command | Exit | Raw evidence |
|---|---|---:|---|
{command_table}

The direct API requests failed with socket operation not permitted. Docker daemon access
was denied; containerd access was forbidden. Snap wrappers refused confinement. Logs are
errors from attempted access, not service logs proving backend health.

An additional elevated Docker attempt was rejected by the tool's automatic approval policy
before command execution. Reason: sandbox_approval is false; escalation cannot be requested.
This rejection is documented here rather than counted as an executed Docker command.

## Dependency installation and compile/test commands

These commands were executed during preparation/verification:

```bash
# New venv created using the existing Python 3.12 interpreter:
../.venv/bin/python -m venv .venv
# The actual install was offline through cached wheel links:
.venv/bin/python -m pip install --no-index --find-links .verification/wheelhouse -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze
../.venv/bin/python pipelines/four_stage.py
../.venv/bin/python scripts/verify_spec.py
MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR=/tmp/four-stage-matplotlib \
  .venv/bin/python -m unittest discover -s tests -v
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/local_verify.py
.venv/bin/python scripts/collect_environment.py
```

The initial virtual-environment creation/install commands were launched from the workspace
with equivalent prefixed paths; no protected environment was modified. Exact offline install
logs: offline-install.log, offline-install-retry.log, offline-subset-install.log,
offline-install-final.log. The first three failed because the reconstructed wheelhouse had
incomplete/unsuitable tags. The final install succeeded with all original pinned versions.
`application-environment.txt` records the resulting environment. `pip check`: no broken
requirements. A first compilation command used `../../.venv/bin/python` and failed path
resolution; `../.venv/bin/python` corrected it. compile.log records successful compilation.

Unit tests ran twice: initially 31 passed; final suite 33 passed after adding registry
failure-propagation and wrong-parent-lineage scenarios. Final raw output: unit-tests.log.
No final test failures remain. No additional full training was run for negative scenarios.

## Independent local stage commands

Local verification start UTC: {LOCAL['started_utc']}. This is one actual complete local
training execution, not a KFP run. The wrapper records unique PIDs and per-stage elapsed
seconds. Every command below ran as a new process.

| Stage | Actual command | Exit | Seconds |
|---|---|---:|---:|
{local_command_table}

The registration command was executed a second time without retraining. It reused the
same local version; see registration-retry.log and retry_registration in local-verification.json.
The source CSV was renamed after validation so later commands could not reopen its original
path. Local storage was new; no existing/shared database was migrated or touched.

The final validation implementation also ran separately with a synthetic executor input:
`python -m src.component_cli validate ... --executor-input JSON`. Its generated artifact
metadata was parsed using `google.protobuf.json_format.ParseDict` against the real
`kfp.pipeline_spec.pipeline_spec_pb2.ExecutorOutput`. Artifact names/types/URIs were preserved.
Raw log: executor-contract.log; result: executor-contract-result.json. Synthetic s3 URIs
were not contacted; this is schema validation, not object-store proof.

## KFP API probe and withheld submission

A read-only urllib request to
`http://ml-pipeline.kubeflow.svc.cluster.local:8888/apis/v2beta1/healthz` ran with a five-second
timeout and failed DNS resolution. Timestamp/error: kfp-api-probe.log. This in-cluster DNS
name is not expected to work from every workstation environment even outside a sandbox;
its failure alone is not evidence of an unhealthy KFP service.

`submit_run.py` was not used to create a run because the image could not be built/imported,
the Kubernetes API was blocked and no reachable authenticated KFP endpoint was verified.
No run ID or workflow name was fabricated. Containerd import and smoke pod creation were
not attempted with a nonexistent image archive.

## Evidence index

| File/directory | Meaning |
|---|---|
| protected_before.json, source-preservation.json | Baseline hashes and unchanged protected lectures |
| copied_files.json | Exact original file-copy list |
| cached_wheels.json, offline-*.log | Offline dependency provenance and failed/successful attempts |
| application-environment.txt | Installed application versions |
| compile.log, spec-verification.json/log | Actual compiled spec and structural checks |
| unit-tests.log | Final 33-test output |
| local-verification.json | Full local experiment, stages, IDs, models, scores and retry results |
| local-run/*.json, heldout.csv, *.log | Concise retained local artifacts and stage/retry logs |
| environment-checks.json, environment-*.log | UTC command ledger and runtime access failures |
| executor-contract-result.json/log | Actual protobuf metadata contract check only |
| kfp-api-probe.log | Read-only internal endpoint/DNS failure |
| ../pipelines/compiled/bank_marketing_four_stage_v1.yaml | Delivered KFP v2 IR/platform package |

Underlying local database and full model artifacts are retained in `{LOCAL['root']}`.
They are ignored by Git/build context but remain on disk. The reports contain no database
credentials. Exact raw commands for local stages are also serialized in execution.json.
''')

handoff = f'''# Codex handoff — Bank Marketing four-stage KFP migration

Generated UTC: {STAMP}. **Overall verdict: PARTIAL**.

## What was delivered

Project: `{ROOT}`.
A genuine four-stage KFP v2 pipeline with tasks validate-dataset → train-candidates →
evaluate-candidates → register-winner. Typed Dataset/Artifact/Metrics edges connect actual
files; no downstream stage rereads the original CSV. Stage 1 freezes configuration and
validates bank rules. Stage 2 makes one seed-42 stratified split and trains the four original
classifiers. Stage 3 reloads actual MLflow model URIs in a new process, logs child metrics,
compares F1 with accuracy tie-breaking and handles all-rejected outcomes. Stage 4 registers
only an eligible winner, queries/loads its version and reuses a matching sequential retry.

Docker image reference: `bank-marketing-kfp-four-stage:v1` (**not built/imported here**).
Compiled package: `pipelines/compiled/bank_marketing_four_stage_v1.yaml`.
KFP compiler 2.17.0; exact four tasks, typed dependencies, resources, disabled caching and
IfNotPresent were verified in the actual generated spec. Container-component ExecutorOutput
metadata was accepted by the actual KFP protobuf. These are not live artifact-transfer tests.

## Real execution evidence

A complete workflow ran successfully in **four independent local Python 3.12 processes**
against NEW isolated SQLite/artifact storage. Source input was renamed after validation;
downstream stages still succeeded. All four models trained, later evaluation loaded their
actual saved artifacts, and explicit MLflow parent/child lineage was checked.
33 automated tests passed; pip check passed. No protected source file changed.

Local execution: `{REG['execution_id']}`.
Local experiment ID: `{REG['experiment_id']}`.
Local parent ID: `{REG['parent_run_id']}`.
Local source/model URI: `{REG['selected_run_id']}` / `{REG['selected_model_uri']}`.

{metrics_table()}

Winner: **{REG['selected_model']}**. Registered **in isolated LOCAL registry** as
`{REG['registered_model_name']}` version `{REG['registered_model_version']}`.
Registered model load/prediction succeeded. Registration-only retry reused this version
and the isolated registry still had one version. This local version 1 is unrelated to the
user-reported prior version 1 on the shared server.

Full IDs, process commands/durations, hashes, artifact paths and outputs:
`docs/evidence/local-verification.json`; stage logs/JSON copies: `docs/evidence/local-run/`.
Full local model/database storage: `{LOCAL['root']}`.

## Live deployment and metadata: unresolved

KFP run ID: **not submitted / not available**.
Argo workflow: **not available**.
All four Kubernetes task statuses: **NOT VERIFIED**.
Dedicated image build/import/pod start: **NOT VERIFIED**.
MinIO/KFP artifact upload/download and live metadata: **NOT VERIFIED**.
Shared MLflow results/version retrieval: **NOT VERIFIED**.

Actual attempts: Docker build/socket permission denied; containerd socket operation not
permitted; direct Kubernetes API socket operation not permitted; Snap wrappers rejected
confinement; KFP internal DNS health probe failed. Automatic approval review rejected
sandbox escalation because policy disables it. No repeated user command requests, cluster
repairs, public image pushes or destructive operations were performed.

Historical context error for `78cf2a06-1afb-4e3b-a767-ef2844eb5cc7` was supplied by the user,
with reported Argo success/publish logs. Current writer/MLMD/MySQL/API log requests were
blocked, so root cause and old/new scope remain **NOT DETERMINED**. Successful model
registration does not establish MLMD context health. Do not present hypotheses as findings.

## Preservation and code changes

Protected 68 regular files in lectures 03/04/05 have unchanged hashes. Seven Lecture 3 core
modules plus active bank CSV, YAML and exact application requirements were copied untouched.
New stage/CLI/artifact modules implement process boundaries; original algorithm/preprocessing/
quality semantics are reused. The new bank config checks explicitly reject unsupported task
contracts. Only the new directory was written. Existing environments were read, not changed.

## Known design limits

Sequential retry safety is best-effort: parallel lookup/create can race. Training rerun can
create another parent/model set; no automatic retries are configured. MLflow FINISHED at a
stage boundary is distinct from KFP workflow completion; workflow_state tags track progress.
This remains binary bank classification with post-call duration and a selection test set,
not an independently evaluated production predictor. No serving/monitoring infrastructure
was added. Live backend compatibility and runtime artifact permissions remain to be verified.

## Next work for an assistant with permitted runtime access

Read the retained evidence first; do not ask the user to repeat completed local tests.
Build the unique image, import it into eligible MicroK8s nodes and inspect a uniquely named
smoke pod. Use existing authenticated KFP access to submit one descriptive run in admin,
monitor it and retain failures. Collect task/workflow/API status, artifacts and registered
source/version. Compare old/new MLMD context/artifact retrieval. Request approval only for
specific evidence-supported shared infrastructure changes; never reset databases/PVCs.
`submit_run.py` and the operations guide provide the prepared commands.

## Nine detailed reports

1. [01_EXECUTION_REPORT.md](docs/01_EXECUTION_REPORT.md)
2. [02_ARCHITECTURE_AND_DATA_FLOW.md](docs/02_ARCHITECTURE_AND_DATA_FLOW.md)
3. [03_CODE_WALKTHROUGH.md](docs/03_CODE_WALKTHROUGH.md)
4. [04_NEW_DATASET_MIGRATION_GUIDE.md](docs/04_NEW_DATASET_MIGRATION_GUIDE.md)
5. [05_DEBUGGING_AND_OPERATIONS.md](docs/05_DEBUGGING_AND_OPERATIONS.md)
6. [06_LEARNING_PLAN.md](docs/06_LEARNING_PLAN.md)
7. [07_SOURCE_CHANGELOG.md](docs/07_SOURCE_CHANGELOG.md)
8. [08_METADATA_DIAGNOSIS.md](docs/08_METADATA_DIAGNOSIS.md)
9. [09_COMMANDS_AND_EVIDENCE.md](docs/09_COMMANDS_AND_EVIDENCE.md)

Implementation and verification are complete to the extent documented. Please share CODEX_HANDOFF.md with ChatGPT so you can review the architecture and implementation step by step.
'''
(ROOT/'CODEX_HANDOFF.md').write_text(handoff)
(ROOT/'README.md').write_text('''# Bank Marketing — four-stage Kubeflow Pipelines v2

Start with [CODEX_HANDOFF.md](CODEX_HANDOFF.md) for the evidence-backed result and
all nine educational reports. Overall status is PARTIAL: 33 tests, actual local
four-process training/evaluation/registration and KFP compilation passed; Docker,
containerd, live KFP/MinIO execution and metadata inspection were permission blocked.

The dedicated image reference is `bank-marketing-kfp-four-stage:v1`. Four typed,
artifact-connected tasks validate, train, evaluate and register. Original lectures
are preserved. See [operations](docs/05_DEBUGGING_AND_OPERATIONS.md) for commands
and [architecture](docs/02_ARCHITECTURE_AND_DATA_FLOW.md) for contracts/lineage.
''')
print('Generated nine reports, README and CODEX_HANDOFF.md from retained evidence.')
