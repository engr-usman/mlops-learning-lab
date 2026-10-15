# Codex CLI Task: Production AI Engineering — Bank Marketing Kubeflow Pipeline Migration

## 1. Your role and working approach

Act as a senior MLOps / ML Platform Engineer, Python engineer, and technical documentation specialist.

You are working in an existing educational project that teaches production AI engineering, including Python, machine learning, MLflow, Docker, Kubernetes, Kubeflow Pipelines, data engineering, and LLMOps.

**Our teaching philosophy is:**

Run → Observe → Break → Debug → Fix → Automate → Explain Theory → Productionize.

For this task, prioritize:

1. Make the complete workflow run successfully.
2. Verify its behavior with real evidence.
3. Document what was implemented and why.
4. Preserve the original working implementation.
5. Prepare detailed educational documentation so the instructor and student can subsequently study every component, line of code, dependency, artifact, and design decision.

Do not repeatedly ask the user to execute terminal commands or copy/paste outputs. You have access to the Ubuntu environment through Codex CLI. Perform the engineering work directly, within the safety boundaries below.

Do not stop after generating code. Run appropriate tests, submit the pipeline when the environment permits, inspect the results, fix implementation defects, and produce a final report.

Do not claim success without evidence.

---

## 2. Project location and source code

Working directory:

`~/Desktop/mlops-k8s`

Existing Lecture 3 source:

`~/Desktop/mlops-k8s/lectures/lecture-03-mlflow`

Existing Lecture 4 source:

`~/Desktop/mlops-k8s/lectures/lecture-04-kubeflow`

There may also be a previously implemented Lecture 5 pipeline or other experiments in this repository.

**First inspect the actual repository before making changes.**

### Critical source preservation requirements

- Never modify the original `lecture-03-mlflow` directory.
- Never overwrite the existing Lecture 4 implementation.
- Never overwrite or destroy the existing Lecture 5 pipeline.
- Never delete MLflow runs, registered models, Kubeflow experiments, pipeline runs, Kubernetes resources, PVCs, databases, or MinIO objects.
- Never perform destructive cluster resets or database migrations without explicit approval.
- Do not overwrite an existing Git-tracked file unless it is inside the new dedicated implementation directory and you have established that it belongs to this task.

Create a new isolated directory:

`lectures/lecture-04-kubeflow/bank-marketing-four-stage/`

If it already exists, inspect its contents and preserve existing work before proceeding.

Copy the required Lecture 3 application files, configuration, datasets, and dependencies into this new directory. Avoid copying virtual environments, caches, local MLflow databases, large temporary files, and unnecessary datasets.

Document exactly what was copied and what was modified.

Preserve the original business logic wherever practical. Refactor only what is required to support independently executable Kubeflow components.

---

## 3. Existing infrastructure

Environment:

- Ubuntu workstation
- MicroK8s 1.35.x
- Charmed Kubeflow 1.11
- Kubeflow Pipelines v2 backend
- Kubernetes namespace: `admin`
- Kubeflow platform namespace: `kubeflow`
- MLflow namespace: `mlops`
- Python 3.12
- Docker and MicroK8s containerd
- MinIO object storage
- MLflow 3.16.1

MLflow service URI from Kubernetes:

`http://mlflow.mlops.svc.cluster.local:5000`

MinIO service:

`minio.kubeflow.svc.cluster.local:9000`

Existing Docker images:

- `bank-marketing-pipeline:local`
- `bank-marketing-kubeflow:lecture05-v1`
- `ghcr.io/mlflow/mlflow:v3.16.1`

The existing Bank Marketing Docker image has already been imported into MicroK8s containerd.

A previous Kubeflow workflow named `lecture-05-bank-marketing-mlflow` succeeded, and MLflow registered model `bank-marketing-pipeline-model`, version `1`.

**Do not assume this means all infrastructure is healthy. Verify the current state.**

The previous pipeline used two Kubeflow tasks:

1. `validate-data`
2. `train-compare-register`

The second task executed training, evaluation, selection, and registration inside a single Python process.

We now want to replace that architecture with four independent, artifact-connected Kubeflow stages.

---

## 4. Primary objective

Implement a genuine **Kubeflow Pipelines v2 four-stage machine learning pipeline**.

Required architecture:

### Stage 1 — Validate Dataset

Input:

- CSV dataset
- YAML configuration

Responsibilities:

- Load configuration.
- Verify the dataset exists and can be read.
- Validate required columns.
- Validate feature types.
- Validate categorical values.
- Validate target values.
- Check missing values.
- Check minimum rows and class distribution.
- Verify that stratified train/test splitting is feasible.
- Preserve or improve existing validation behavior without silently changing business rules.

Output:

- Validated dataset as a typed KFP `Dataset` artifact.
- Validation metadata, including row count, column count, feature count, dataset hash, and validation status.

The validated dataset must be consumed by downstream tasks through KFP artifact interfaces, not by independently reopening the original CSV.

### Stage 2 — Train Candidates

Input:

- Validated dataset artifact.
- Configuration.
- MLflow tracking URI.

Responsibilities:

- Create one deterministic, stratified train/test split.
- Ensure every candidate model uses exactly the same split.
- Train all four existing classifiers:

  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

- Preserve the existing preprocessing pipeline.
- Preserve feature ordering and categorical/numeric transformations.
- Log training parameters, dataset lineage, model artifacts, and run information to MLflow.
- Maintain traceable MLflow parent/child run relationships where practical.
- Ensure all models are retrievable from subsequent Kubernetes Pods.

Outputs:

- Candidate model manifest containing model names, MLflow run IDs, and model URIs.
- Held-out test dataset artifact.
- Split metadata containing train/test row counts, random seed, split ratio, and reproducibility information.

Do not leak test data into model fitting or preprocessing fitting.

### Stage 3 — Evaluate Candidates

Inputs:

- Candidate model manifest.
- Held-out test dataset artifact.
- Configuration.
- MLflow tracking URI.

Responsibilities:

- Load each candidate model from MLflow using its actual model URI.
- Evaluate each candidate against the same held-out test dataset.
- Calculate:

  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix

- Use positive class `yes`.
- Log evaluation metrics and confusion matrices to the correct MLflow candidate runs.
- Apply the configured minimum accuracy quality gate.
- Select the best eligible model by configured F1 score, using accuracy as the tie-breaker.
- Preserve the existing model-selection semantics.
- Handle the case where no model passes the quality gate.

Outputs:

- Evaluation metrics for all candidates.
- Candidate eligibility results.
- Selected candidate metadata, including model name, MLflow run ID, model URI, and metrics.
- Explicit quality-gate result.

### Stage 4 — Register Winner

Inputs:

- Selected model metadata.
- Quality-gate result.
- Configuration.
- MLflow tracking URI.

Responsibilities:

- Register the winning model in MLflow Model Registry only if the quality gate passes.
- Use the model URI produced by the actual training run.
- Record the selected model's name, source run ID, metrics, and registered model version.
- Provide an explicit non-registration result if no candidate qualifies.
- Avoid duplicate registrations during retries where feasible, and document the chosen idempotency strategy.
- Do not mutate or delete existing registered model versions.

Outputs:

- Registration status.
- Registered model name.
- Registered model version.
- Selected MLflow run ID.
- Selected model URI.
- Summary metrics.

---

## 5. Important architecture requirements

Use **actual KFP v2 typed artifact contracts**.

For example:

- `Input[Dataset]`
- `Output[Dataset]`
- `Input[Artifact]`
- `Output[Artifact]`
- `Output[Metrics]`

Use these where appropriate, rather than forcing all data into strings.

Small metadata objects may be passed as JSON-serializable parameters where supported and sensible.

The pipeline must represent real data dependencies between tasks.

**Do not use only `.after()` relationships while having downstream tasks independently reload the original dataset.**

Use:

- Kubeflow for workflow orchestration and dataset artifacts.
- MinIO through the KFP artifact system.
- MLflow for experiment tracking, model artifacts, evaluation lineage, and model registration.

Ensure that MLflow tracking works across independent Kubernetes Pods.

Do not assume that an MLflow active run context survives between components.

Handle run creation, continuation, logging, and termination explicitly.

Ensure that each component has its own clear input/output contract.

Use reasonable CPU and memory requests/limits for the available workstation.

Disable caching for operations with external side effects, particularly model registration. Explain caching decisions for other stages.

Do not introduce unnecessary technologies, services, frameworks, or complex abstractions.

Prefer a simple, correct, observable implementation.

---

## 6. Python implementation requirements

Use Python 3.12.

Inspect the existing Lecture 3 modules:

- `src/config.py`
- `src/data_validation.py`
- `src/training.py`
- `src/evaluation.py`
- `src/registration.py`
- `src/pipeline.py`
- `src/logging_config.py`

Reuse their existing logic where appropriate.

Create clean component entrypoints with explicit arguments and output paths.

Make sure each component can run independently as a container process.

Avoid:

- Hardcoded workstation paths.
- Hidden dependencies on the developer's virtual environment.
- Passing Python DataFrames directly between Pods.
- Depending on global mutable state.
- Assuming an active MLflow run exists.
- Silent exception handling.
- Unexplained duplicated business logic.

Use structured, stage-level logging with enough context to diagnose failures.

Prefer explicit exceptions for invalid data and failed operations.

Pin dependency versions and document the Python environment.

Add meaningful type hints and docstrings.

---

## 7. Container image requirements

Build a dedicated image for this new implementation.

Use a unique, versioned tag, for example:

`bank-marketing-kfp-four-stage:v1`

Do not overwrite the existing Lecture 3 or Lecture 5 image tags.

Make the image available to MicroK8s containerd.

Set an appropriate image pull policy for locally imported images.

Verify that Kubernetes can start a Pod using the image.

Do not publish images to a public registry unless explicitly authorized.

Do not run destructive Docker cleanup commands.

---

## 8. KFP compilation and execution

Implement a KFP v2 pipeline definition with four visible tasks:

1. `validate-dataset`
2. `train-candidates`
3. `evaluate-candidates`
4. `register-winner`

Compile it into a versioned pipeline YAML file.

Verify that the compiled specification has the expected tasks, artifact dependencies, component images, and resource limits.

Submit a real pipeline run to the existing Kubeflow environment, preferably in the existing `admin` namespace.

Use a clearly identifiable experiment and run name.

If creating a new experiment, use a descriptive name such as:

`lecture-04-bank-marketing-four-stage`

Do not delete existing experiments.

Monitor execution until it succeeds, fails, or reaches a documented timeout.

Collect:

- Pipeline run ID.
- Workflow name.
- Task names.
- Task statuses.
- Task durations.
- Relevant task logs.
- Artifact URIs.
- MLflow experiment ID.
- MLflow run IDs.
- Registered model name and version.

If a task fails, diagnose the cause, fix the implementation if safe, and rerun with a new identifiable run.

Do not hide failed attempts from the final report.

---

## 9. Existing Kubeflow metadata problem

A previous Kubeflow run reported:

`Cannot find context with {"typeName":"system.PipelineRun","contextName":"78cf2a06-1afb-4e3b-a767-ef2844eb5cc7"}: Cannot find specified context`

However, the corresponding Argo workflow was reported as `Succeeded`.

The launcher logs also included:

- `publish success`
- `Attempting to update DAG state`
- Successful subprocess exits

The following components were running:

- `kfp-metadata-writer-0`
- `mysql-db-0`
- `metadata-grpc-service`
- `ml-pipeline`

Investigate this issue carefully.

Tasks:

1. Verify the current KFP backend, MLMD, MySQL, and metadata-writer health.
2. Inspect relevant logs for missing contexts, database errors, metadata publication failures, and restart-related issues.
3. Determine whether the problem affects old runs, new runs, or both.
4. Compare Kubernetes workflow status with KFP API/UI status.
5. Check whether KFP artifacts and metadata are being persisted and retrieved correctly.
6. Identify the most likely root cause using evidence.

**Safety restrictions:**

- Do not delete or reset MLMD/MySQL databases.
- Do not delete PVCs.
- Do not restart or reconfigure Kubeflow services merely as an experiment.
- Do not edit Charmed Kubeflow-managed resources without explicit approval.
- Do not expose database credentials or Kubernetes Secrets in reports.
- If remediation requires infrastructure changes, document the proposed change and request approval before applying it.

Do not block implementation solely because an older run has missing metadata, provided new pipeline execution is safe and functional.

---

## 10. Verification and test scenarios

Do more than a happy-path execution.

Implement appropriate automated tests for:

### Dataset validation

- Valid dataset.
- Missing required column.
- Invalid categorical value.
- Missing feature value.
- Invalid numeric value.
- Missing or invalid target.
- Insufficient target-class distribution.
- Empty dataset.

### Reproducibility

- Same random seed produces the same train/test split.
- All candidate models use the same held-out dataset.
- Dataset hash is recorded.
- No data leakage occurs between training and evaluation.

### Model evaluation

- Correct metric calculations.
- Correct positive label.
- Correct F1-based selection.
- Correct accuracy tie-breaker.
- Correct behavior when a candidate fails the quality gate.
- Correct behavior when all candidates fail.

### Model registration

- Successful registration when quality gate passes.
- No registration when quality gate fails.
- Correct model URI and source run linkage.
- Safe behavior during retries.

### Integration

- Container starts successfully.
- KFP pipeline compiles.
- Components have real artifact dependencies.
- KFP artifacts can be stored and retrieved.
- MLflow model artifacts can be retrieved by another Pod.
- Registry version is created and can be queried.

Do not run unnecessary additional full training pipelines merely to test error scenarios that can be covered through unit tests.

---

## 11. Educational documentation requirements

This task is part of a course, not merely an engineering deliverable.

After implementation, produce documentation explaining the complete system in detail.

Assume the learner understands DevOps and Kubernetes but is still building confidence with Python, Pandas, Scikit-Learn, and machine learning concepts.

Explain:

- What each component does.
- Why it exists.
- What problem it solves.
- What data enters it.
- What it produces.
- What happens if it fails.
- Which Python functions it calls.
- Which libraries are used.
- How it communicates with the next component.
- How artifacts are stored.
- How MLflow run lineage works.
- How KFP schedules and executes tasks.
- How configuration is loaded.
- How the container starts.
- How to inspect execution in Kubeflow and MLflow.
- How to debug common failures.

Include a comparison of the original single-process Python pipeline and the new multi-Pod Kubeflow pipeline.

Explain the difference between:

- Python function calls and KFP components.
- In-memory objects and persisted artifacts.
- A container image and a running Pod.
- KFP metadata and MLflow metadata.
- Dataset artifacts and model artifacts.
- Orchestration dependencies and data dependencies.
- Model training and model evaluation.
- Experiment tracking and model registration.
- Pipeline success and model quality.

Also explain why a successful model registration does not automatically prove that the Kubeflow metadata system is healthy.

---

## 12. New dataset migration guide

We will later repeat the entire exercise using a different dataset.

Create a comprehensive guide explaining how to migrate a new Scikit-Learn project into this four-stage Kubeflow architecture.

Include:

1. How to inspect an unfamiliar dataset.
2. How to identify the prediction target.
3. How to distinguish classification and regression.
4. How to identify numerical and categorical features.
5. How to design a validation schema.
6. How to detect missing values, invalid categories, outliers, and class imbalance.
7. How to choose preprocessing techniques.
8. How to avoid data leakage.
9. How to select an appropriate splitting strategy.
10. How to choose candidate algorithms.
11. How to select suitable evaluation metrics.
12. How to define quality gates.
13. How to adapt the YAML configuration.
14. How to modify Python training and evaluation code.
15. How to design KFP artifact contracts.
16. How to adapt MLflow logging and registration.
17. How to build and import the container image.
18. How to compile and execute the KFP pipeline.
19. How to verify reproducibility and lineage.
20. How to debug the resulting pipeline.

Include a practical checklist and a small worked example or pseudocode showing how the component interfaces change when the dataset changes.

Distinguish reusable infrastructure from dataset-specific business logic.

---

## 13. Required report files

Create the following documentation directory:

`lectures/lecture-04-kubeflow/bank-marketing-four-stage/docs/`

Produce these files:

### `01_EXECUTION_REPORT.md`

Include:

- Executive summary.
- Environment details.
- Files created and modified.
- Docker image information.
- KFP compilation result.
- Pipeline execution result.
- Kubeflow run ID.
- Per-component execution status.
- MLflow experiment/run IDs.
- Model metrics.
- Selected model.
- Registered model name/version.
- Artifact locations.
- Verification commands and summarized outputs.
- Failures encountered and how they were resolved.
- Outstanding issues.
- Final PASS/FAIL/PARTIAL verdict.

### `02_ARCHITECTURE_AND_DATA_FLOW.md`

Include:

- Architecture diagram in Mermaid.
- Four component contracts.
- Dataset and model artifact flows.
- MLflow lineage.
- KFP/MinIO/MLflow responsibilities.
- Failure boundaries.
- Retry and caching strategy.
- Design decisions and alternatives.

### `03_CODE_WALKTHROUGH.md`

Provide a detailed, file-by-file educational walkthrough.

Include function signatures, important code excerpts, explanations, and references to exact file paths.

Explain how the original Lecture 3 Python code was adapted.

### `04_NEW_DATASET_MIGRATION_GUIDE.md`

Provide a reusable, detailed process for applying this architecture to a new dataset and a new ML problem.

Include validation, preprocessing, feature engineering, training, evaluation, model selection, registration, and Kubeflow integration.

### `05_DEBUGGING_AND_OPERATIONS.md`

Include:

- How to inspect KFP runs.
- How to inspect Pods and logs.
- How to inspect artifacts.
- How to inspect MLflow runs and models.
- Common failure scenarios.
- Root-cause investigation workflow.
- Metadata context error investigation.
- Safe recovery procedures.
- Commands that are safe to run.
- Commands requiring explicit approval.

### `06_LEARNING_PLAN.md`

Design a step-by-step teaching sequence that starts from the working implementation.

The instructor and student will review one component at a time.

Each session should follow:

Observe → Explain → Inspect Code → Modify → Break Safely → Debug → Restore → Reproduce.

Include exercises and questions that require the learner to explain the code rather than merely execute it.

### `07_SOURCE_CHANGELOG.md`

Document:

- Original files copied.
- New files created.
- Functions reused.
- Functions modified.
- Behavioral changes.
- Compatibility considerations.
- Known limitations.
- Differences from the original Lecture 3 pipeline.

### `08_METADATA_DIAGNOSIS.md`

Document the missing `system.PipelineRun` context issue separately, including evidence, findings, suspected causes, and proposed remediation.

Do not present speculation as confirmed root cause.

### `09_COMMANDS_AND_EVIDENCE.md`

Record the actual commands executed, relevant outputs, timestamps, and paths to generated logs or artifacts.

Redact credentials and sensitive environment variables.

Use concise log excerpts where appropriate, and retain useful raw evidence files separately.

---

## 14. Final verification requirements

Before reporting completion, verify:

- Original Lecture 3 files remain unchanged.
- Existing Lecture 4 and Lecture 5 implementations remain intact.
- The new image exists in MicroK8s containerd.
- The KFP YAML compiles.
- Exactly four intended pipeline stages are visible.
- Stage outputs are connected through real artifact dependencies.
- Dataset validation passes.
- All four candidate models train successfully.
- Evaluation reads model artifacts produced by training.
- Quality gates behave correctly.
- The selected model is registered in MLflow.
- The registered model version is retrievable.
- KFP run status and MLflow results are recorded.
- Any metadata issues are documented.
- Tests and documentation are complete.

If any item cannot be verified, explicitly mark it as NOT VERIFIED and explain why.

Do not silently substitute a two-stage pipeline for the requested four-stage pipeline.

Do not report the task as fully successful if only Python training and MLflow registration succeed but KFP artifact transfer or pipeline metadata remains broken.

---

## 15. Execution permissions and safety

You may:

- Inspect files and the local environment.
- Create the new isolated project directory.
- Copy required source files.
- Write and refactor code inside that directory.
- Create documentation and tests.
- Build a uniquely tagged Docker image.
- Import the image into MicroK8s.
- Compile KFP specifications.
- Submit clearly named test runs to the existing namespace.
- Inspect logs, events, workflow statuses, and metadata.
- Fix defects in the newly created implementation.

You must request approval before:

- Deleting or modifying existing Kubeflow resources.
- Modifying existing databases, PVCs, or MinIO buckets.
- Restarting or reconfiguring shared infrastructure.
- Installing or upgrading cluster-wide services.
- Changing Charmed Kubeflow applications.
- Performing destructive Git operations.
- Publishing images or artifacts externally.
- Making changes outside the new isolated project directory, except for ordinary non-destructive build and pipeline execution operations.

If an action is blocked by permissions, credentials, or infrastructure, document the blocker and continue with safe work.

---

## 16. Final response format

When finished, do not paste hundreds of lines of terminal output into the conversation.

Provide a concise final summary containing:

1. Overall result: PASS / PARTIAL / FAIL.
2. New project directory.
3. Docker image tag.
4. KFP pipeline run ID and status.
5. Four component statuses.
6. Selected model and evaluation metrics.
7. MLflow registered model and version.
8. Metadata investigation result.
9. Tests passed/failed.
10. Any unresolved blockers.
11. Exact paths to all nine documentation files.

**Most importantly:**

Create one consolidated file:

`lectures/lecture-04-kubeflow/bank-marketing-four-stage/CODEX_HANDOFF.md`

This should be the primary file that the user shares with ChatGPT.

It must summarize the implementation, architecture, actual execution evidence, MLflow results, metadata diagnosis, source changes, unresolved issues, and references to the nine detailed reports.

The handoff must contain enough information for another AI assistant to understand the completed work without requiring the user to repeat terminal commands.

Finish by telling the user:

"Implementation and verification are complete to the extent documented. Please share CODEX_HANDOFF.md with ChatGPT so you can review the architecture and implementation step by step."

Begin by inspecting the existing repository, verifying infrastructure, and writing a short implementation plan. Then execute the task end-to-end within the stated safety boundaries.