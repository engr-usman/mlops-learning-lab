# Codex handoff — Bank Marketing four-stage KFP migration

Generated UTC: 2026-10-13T11:21:45.128907+00:00. **Overall verdict: PARTIAL**.

## What was delivered

Project: `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage`.
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

Local execution: `local-proof-20261013T110612Z`.
Local experiment ID: `1`.
Local parent ID: `d777eafa6a244c399c07932525b62cf5`.
Local source/model URI: `22f47d882a3747dbbee69b3ae487acfb` / `models:/m-c437b48b98914d898c2ed33ae080625a`.

| Candidate | Accuracy | Precision | Recall | F1 | Eligible |
|---|---:|---:|---:|---:|---|
| logistic_regression | 0.979206049 | 0.987500000 | 0.970699433 | 0.979027645 | True |
| decision_tree | 0.983459357 | 1.000000000 | 0.966918715 | 0.983181163 | True |
| random_forest | 0.982514178 | 0.992285439 | 0.972589792 | 0.982338902 | True |
| gradient_boosting | 0.982986767 | 0.992292871 | 0.973534972 | 0.982824427 | True |

Winner: **decision_tree**. Registered **in isolated LOCAL registry** as
`bank-marketing-pipeline-model` version `1`.
Registered model load/prediction succeeded. Registration-only retry reused this version
and the isolated registry still had one version. This local version 1 is unrelated to the
user-reported prior version 1 on the shared server.

Full IDs, process commands/durations, hashes, artifact paths and outputs:
`docs/evidence/local-verification.json`; stage logs/JSON copies: `docs/evidence/local-run/`.
Full local model/database storage: `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z`.

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
