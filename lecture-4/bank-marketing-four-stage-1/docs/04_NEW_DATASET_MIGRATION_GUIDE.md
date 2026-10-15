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
