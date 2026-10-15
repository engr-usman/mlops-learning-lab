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
microk8s kubectl run bank-four-stage-image-smoke-UNIQUE -n admin   --image=bank-marketing-kfp-four-stage:v1 --image-pull-policy=IfNotPresent   --restart=Never --command -- python -m src.component_cli --help
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
