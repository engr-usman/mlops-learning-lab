# Source changelog and preservation

The target directory did not exist on inspection. No existing work needed overwriting.
Protected lecture hash check: **PASS**, 68
regular source/data/document files; changed: 0; missing:
0. Evidence: `protected_before.json` and `source-preservation.json`.
Virtual environments and Python caches were excluded from source fingerprints and were
not copied or modified in protected lectures.

## Copied files

| Original relative workspace path | Destination relative new-project path | File changes |
|---|---|---|
| `lectures/lecture-03-mlflow/src/config.py` | `src/config.py` | unchanged |
| `lectures/lecture-03-mlflow/src/data_validation.py` | `src/data_validation.py` | unchanged |
| `lectures/lecture-03-mlflow/src/training.py` | `src/training.py` | unchanged |
| `lectures/lecture-03-mlflow/src/evaluation.py` | `src/evaluation.py` | unchanged |
| `lectures/lecture-03-mlflow/src/registration.py` | `src/registration.py` | unchanged |
| `lectures/lecture-03-mlflow/src/pipeline.py` | `src/pipeline.py` | unchanged |
| `lectures/lecture-03-mlflow/src/logging_config.py` | `src/logging_config.py` | unchanged |
| `lectures/lecture-03-mlflow/configs/config.yaml` | `configs/config.yaml` | unchanged |
| `lectures/lecture-03-mlflow/data/balance.csv` | `data/balance.csv` | unchanged |
| `lectures/lecture-03-mlflow/requirements.txt` | `requirements.txt` | unchanged |

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
