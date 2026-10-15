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
