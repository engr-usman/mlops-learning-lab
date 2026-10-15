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
| [src/stages.py:28](../src/stages.py) | `def check_config(config: dict[str, Any]) -> None` |
| [src/stages.py:50](../src/stages.py) | `def validate_stage(dataset_path: str, config_path: str, validated_path: str, snapshot_path: str, report_path: str) -> dict[str, Any]` |
| [src/stages.py:81](../src/stages.py) | `def index_hash(indices: list[int]) -> str` |
| [src/stages.py:85](../src/stages.py) | `def train_stage(validated_path: str, snapshot_path: str, validation_path: str, tracking_uri: str, execution_id: str, manifest_path: str, heldout_path: str, split_path: str) -> dict[str, Any]` |
| [src/stages.py:157](../src/stages.py) | `def evaluate_stage(manifest_path: str, heldout_path: str, split_path: str, snapshot_path: str, tracking_uri: str, evaluation_path: str) -> dict[str, Any]` |
| [src/stages.py:221](../src/stages.py) | `def find_registered_version(client: MlflowClient, name: str, key: str, run_id: str, model_uri: str) -> Any | None` |
| [src/stages.py:239](../src/stages.py) | `def register_stage(evaluation_path: str, snapshot_path: str, tracking_uri: str, registration_path: str) -> dict[str, Any]` |
| [src/component_cli.py:14](../src/component_cli.py) | `def parser() -> argparse.ArgumentParser` |
| [src/component_cli.py:32](../src/component_cli.py) | `def main() -> None` |
| [src/artifact_io.py:11](../src/artifact_io.py) | `def sha256(path: str | Path) -> str` |
| [src/artifact_io.py:20](../src/artifact_io.py) | `def read_json(path: str | Path) -> dict[str, Any]` |
| [src/artifact_io.py:27](../src/artifact_io.py) | `def write_json(path: str | Path, value: dict[str, Any]) -> None` |
| [src/artifact_io.py:36](../src/artifact_io.py) | `def check_hash(path: str | Path, expected: str) -> None` |
| [src/artifact_io.py:41](../src/artifact_io.py) | `def publish_metadata(executor_input: str | None, metadata: dict[str, dict[str, Any]]) -> None` |
| [src/config.py:13](../src/config.py) | `def load_config(config_path='configs/config.yaml')` |
| [src/data_validation.py:26](../src/data_validation.py) | `def validate_dataset(data_path, features, target, numeric_features=None, target_values=None, test_size=None)` |
| [src/training.py:20](../src/training.py) | `def calculate_file_hash(file_path)` |
| [src/training.py:40](../src/training.py) | `def create_pipeline(config, model_name='decision_tree')` |
| [src/training.py:86](../src/training.py) | `def split_dataset(df, config)` |
| [src/training.py:94](../src/training.py) | `def train_model(df, config, run_name=None, mlflow_run=None, model_name='decision_tree', split=None)` |
| [src/evaluation.py:19](../src/evaluation.py) | `def evaluate_model(pipeline, X_test, y_test, positive_label='yes')` |
| [src/evaluation.py:95](../src/evaluation.py) | `def select_best_model(candidates, config)` |
| [src/registration.py:8](../src/registration.py) | `def passes_quality_gate(metrics, config)` |
| [src/registration.py:40](../src/registration.py) | `def register_model(run_id, config, model_uri=None)` |
| [src/pipeline.py:23](../src/pipeline.py) | `def pipeline_stage(summary, stage)` |
| [src/pipeline.py:48](../src/pipeline.py) | `def log_execution_summary(summary)` |
| [src/pipeline.py:57](../src/pipeline.py) | `def run_pipeline()` |
| [src/pipeline.py:95](../src/pipeline.py) | `def _run_pipeline(pipeline_run_id, summary)` |
| [src/logging_config.py:12](../src/logging_config.py) | `def run_logging_context(run_id=None)` |
| [src/logging_config.py:22](../src/logging_config.py) | `class RunIdFilter` |
| [src/logging_config.py:28](../src/logging_config.py) | `def setup_logging(level=logging.INFO)` |
| [pipelines/four_stage.py:15](../pipelines/four_stage.py) | `def validate_dataset(dataset_path: str, config_path: str, validated_dataset: Output[Dataset], configuration: Output[Artifact], validation_report: Output[Artifact])` |
| [pipelines/four_stage.py:26](../pipelines/four_stage.py) | `def train_candidates(validated_dataset: Input[Dataset], configuration: Input[Artifact], validation_report: Input[Artifact], tracking_uri: str, execution_id: str, candidate_manifest: Output[Artifact], heldout_dataset: Output[Dataset], split_metadata: Output[Artifact])` |
| [pipelines/four_stage.py:40](../pipelines/four_stage.py) | `def evaluate_candidates(candidate_manifest: Input[Artifact], heldout_dataset: Input[Dataset], split_metadata: Input[Artifact], configuration: Input[Artifact], tracking_uri: str, evaluation_report: Output[Artifact], evaluation_metrics: Output[Metrics])` |
| [pipelines/four_stage.py:52](../pipelines/four_stage.py) | `def register_winner(evaluation_report: Input[Artifact], configuration: Input[Artifact], tracking_uri: str, registration_result: Output[Artifact])` |
| [pipelines/four_stage.py:61](../pipelines/four_stage.py) | `def configure(task: dsl.PipelineTask, cpu: str, memory: str, limit: str) -> None` |
| [pipelines/four_stage.py:73](../pipelines/four_stage.py) | `def bank_marketing_four_stage(tracking_uri: str='http://mlflow.mlops.svc.cluster.local:5000', dataset_path: str='/app/data/balance.csv', config_path: str='/app/configs/config.yaml')` |

## Dependencies: direct use versus inherited support

`requirements.txt` is byte-identical to Lecture 3, preserving its pinned environment.
Some packages support MLflow servers or the optional API and are not directly needed by
these components. Removing them safely would require a separately tested runtime lockfile;
this task prioritizes compatibility. The compiler is separate and pins KFP 2.17.0,
kfp-pipeline-spec/server-api 2.17.0, protobuf 6.33.6 and PyYAML 6.0.3.

| Exact application requirement | Role |
|---|---|
| `aiohappyeyeballs==2.7.1` | Async connection selection |
| `aiohttp==3.14.3` | Async HTTP support |
| `aiosignal==1.4.0` | Async signal helpers |
| `alembic==1.20.0` | MLflow schema support; new isolated local DB only |
| `annotated-doc==0.0.5` | Documentation/type metadata |
| `annotated-types==0.8.0` | Type constraint helpers |
| `anyio==4.15.1` | Async concurrency support |
| `attrs==26.1.0` | Dependency data structures |
| `blinker==1.9.0` | Flask signal support |
| `cachetools==7.2.0` | Authentication/client caches |
| `certifi==2026.7.22` | TLS certificate bundle |
| `cffi==2.1.1` | Native-library interface |
| `charset-normalizer==3.5.1` | HTTP text encoding detection |
| `click==8.5.0` | CLI parsing in MLflow tooling |
| `cloudpickle==3.1.2` | Python object serialization support |
| `contourpy==1.4.0` | Matplotlib contours |
| `cryptography==50.0.1` | Authentication/TLS cryptographic support |
| `cycler==0.12.1` | Matplotlib style cycles |
| `databricks-sdk==0.143.0` | MLflow integration dependency; no Databricks service configured |
| `docker==7.2.0` | Python Docker SDK inherited; the build uses Docker CLI instead |
| `fastapi==0.141.1` | Inherited optional serving dependency; API not copied into this implementation |
| `Flask==3.1.3` | MLflow web server framework |
| `flask-cors==6.0.5` | MLflow HTTP CORS support |
| `fonttools==4.66.0` | Plot font processing |
| `frozenlist==1.8.0` | Async dependency collections |
| `gitdb==4.0.12` | GitPython object database helper |
| `GitPython==3.1.62` | MLflow Git provenance support |
| `google-auth==2.58.1` | Authentication dependency; no Google cloud integration configured |
| `graphene==3.4.3` | MLflow GraphQL dependency |
| `graphql-core==3.2.12` | GraphQL engine |
| `graphql-relay==3.2.0` | GraphQL connection helpers |
| `gunicorn==26.2.0` | MLflow web server worker manager |
| `h11==0.16.0` | HTTP/1 protocol support |
| `huey==3.4.0` | Inherited background-task support |
| `idna==3.20` | International domain encoding |
| `importlib_metadata==9.0.1` | Distribution metadata support |
| `itsdangerous==2.2.0` | Inherited transitive dependency; not directly called by application code |
| `Jinja2==3.1.6` | Server template rendering |
| `joblib==1.6.0` | Scikit-learn serialization/parallel utilities |
| `kiwisolver==1.5.1` | Plot layout solving |
| `Mako==1.4.3` | Migration/template support |
| `MarkupSafe==3.0.3` | Template escaping |
| `matplotlib==3.11.2` | Confusion-matrix plots |
| `mlflow==3.16.1` | Full experiment tracking/model registry client and server distribution inherited from Lecture 3 |
| `mlflow-skinny==3.16.1` | Core MLflow Python tracking/flavor clients |
| `mlflow-tracing==3.16.1` | MLflow tracing support; not used directly by this pipeline |
| `multidict==6.9.1` | HTTP multivalue mappings |
| `narwhals==2.26.0` | Dataframe interoperability dependency |
| `numpy==2.5.3` | Numerical arrays used by pandas/scikit-learn |
| `opentelemetry-api==1.45.0` | MLflow telemetry API |
| `opentelemetry-proto==1.45.0` | Telemetry protobuf messages |
| `opentelemetry-sdk==1.45.0` | MLflow telemetry implementation |
| `opentelemetry-semantic-conventions==0.66b0` | Telemetry naming conventions |
| `packaging==26.3` | Version/requirement parsing |
| `pandas==3.0.6` | CSV/DataFrame handling |
| `pillow==12.3.0` | Plot/image encoding |
| `prettytable==3.18.0` | Tabular presentation dependency |
| `propcache==0.5.4` | Async HTTP property caches |
| `protobuf==6.33.6` | Structured MLflow/KFP message serialization |
| `pyarrow==25.0.1` | Tabular serialization/data support |
| `pyasn1==0.6.4` | ASN.1 encoding dependency |
| `pyasn1_modules==0.4.2` | Authentication ASN.1 schemas |
| `pycparser==3.0` | C parser for CFFI |
| `pydantic==2.13.5` | Typed validation used by inherited serving/server dependencies |
| `pydantic_core==2.46.5` | Pydantic native validation engine |
| `pyparsing==3.3.3` | Plot/text parsing |
| `python-dateutil==2.9.0.post0` | Datetime utilities |
| `python-dotenv==1.2.3` | Environment-file support; no credentials read from .env here |
| `PyYAML==6.0.3` | Original YAML configuration parsing |
| `requests==2.34.2` | HTTP client used by MLflow/services |
| `scikit-learn==1.9.1` | Preprocessing, split, four classifiers and classification metrics |
| `scipy==1.18.1` | Numerical routines used by scikit-learn |
| `six==1.17.0` | Compatibility helpers |
| `skops==0.16.0` | Safer scikit-learn model serialization; trusted tree internals from original code |
| `smmap==5.0.3` | Git object memory mapping |
| `SQLAlchemy==2.1.1` | MLflow SQL tracking/registry backend |
| `sqlparse==0.6.0` | SQL text utilities |
| `starlette==1.7.0` | ASGI web framework dependency |
| `threadpoolctl==3.7.0` | Numerical thread-pool coordination |
| `typing-inspection==0.4.4` | Runtime typing helpers |
| `typing_extensions==4.16.0` | Typing compatibility |
| `urllib3==2.8.0` | HTTP connection/pooling support |
| `uvicorn==0.54.0` | ASGI server dependency; no new server launched |
| `wcwidth==0.9.1` | Terminal display-width utilities |
| `Werkzeug==3.1.8` | Flask HTTP utilities |
| `yarl==1.25.1` | Async URL handling |
| `zipp==4.1.0` | Zip metadata support |

Use `docs/evidence/application-environment.txt` and the successful offline-install log
for actual installed versions. Python package installation is not a Docker build test.
