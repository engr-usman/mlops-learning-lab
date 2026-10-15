# Actual commands and retained evidence

All project operations were executed under
`/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage` unless a command explicitly names a different workspace-relative path.
No credentials, Secrets or sensitive environment values are included. Python source/env
hashes and model/run identifiers are not credentials.

## Infrastructure checks (actual UTC timestamps)

| UTC | Executed command | Exit | Raw evidence |
|---|---|---:|---|
| 2026-10-13T11:07:48.277776+00:00 | `microk8s status` | 1 | [log](evidence/environment-microk8s-status.log) |
| 2026-10-13T11:07:48.396439+00:00 | `juju status --format short` | 1 | [log](evidence/environment-juju-status.log) |
| 2026-10-13T11:07:48.516040+00:00 | `docker image inspect bank-marketing-kfp-four-stage:v1 --format {{.Id}}` | 1 | [log](evidence/environment-docker-images.log) |
| 2026-10-13T11:07:48.539392+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s get nodes -o wide` | 1 | [log](evidence/environment-nodes.log) |
| 2026-10-13T11:07:48.600069+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s get pods -n kubeflow -o wide` | 1 | [log](evidence/environment-kubeflow-health.log) |
| 2026-10-13T11:07:48.646102+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s get workflows -n admin` | 1 | [log](evidence/environment-admin-workflows.log) |
| 2026-10-13T11:07:48.705291+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s get pods,svc,pvc -n mlops` | 1 | [log](evidence/environment-mlflow-health.log) |
| 2026-10-13T11:07:48.752750+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s logs kfp-metadata-writer-0 -n kubeflow --tail=100 --since=1h` | 1 | [log](evidence/environment-metadata-writer-logs.log) |
| 2026-10-13T11:07:48.823078+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s logs mysql-db-0 -n kubeflow --tail=100 --since=1h` | 1 | [log](evidence/environment-mysql-logs.log) |
| 2026-10-13T11:07:48.882132+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s logs deployment/metadata-grpc-service -n kubeflow --tail=100 --since=1h` | 1 | [log](evidence/environment-metadata-grpc-logs.log) |
| 2026-10-13T11:07:48.938753+00:00 | `/snap/microk8s/current/kubectl --kubeconfig /home/usman/Desktop/mlops-k8s/infrastructure/juju/microk8s-kubeconfig.yaml --request-timeout=5s logs deployment/ml-pipeline -n kubeflow --tail=100 --since=1h` | 1 | [log](evidence/environment-ml-pipeline-logs.log) |
| 2026-10-13T11:07:48.987764+00:00 | `/snap/microk8s/current/bin/ctr --address /var/snap/microk8s/common/run/containerd.sock --namespace k8s.io images list` | 1 | [log](evidence/environment-containerd-images.log) |
| 2026-10-13T11:07:48.994216+00:00 | `docker build -t bank-marketing-kfp-four-stage:v1 .` | 1 | [log](evidence/environment-docker-build.log) |

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
MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR=/tmp/four-stage-matplotlib   .venv/bin/python -m unittest discover -s tests -v
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

Local verification start UTC: 20261013T110612Z. This is one actual complete local
training execution, not a KFP run. The wrapper records unique PIDs and per-stage elapsed
seconds. Every command below ran as a new process.

| Stage | Actual command | Exit | Seconds |
|---|---|---:|---:|
| validate | `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.venv/bin/python -m src.component_cli validate --dataset /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/source.csv --config /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/configs/config.yaml --validated /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/validated.csv --snapshot /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/configuration.json --validation /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/validation.json` | 0 | 5.025 |
| train | `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.venv/bin/python -m src.component_cli train --validated /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/validated.csv --snapshot /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/configuration.json --validation /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/validation.json --tracking-uri sqlite:////home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/mlflow.db --execution-id local-proof-20261013T110612Z --manifest /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/manifest.json --heldout /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/heldout.csv --split /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/split.json` | 0 | 18.152 |
| evaluate | `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.venv/bin/python -m src.component_cli evaluate --manifest /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/manifest.json --heldout /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/heldout.csv --split /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/split.json --snapshot /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/configuration.json --tracking-uri sqlite:////home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/mlflow.db --evaluation /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/evaluation.json --metrics /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/metrics.json` | 0 | 5.674 |
| register | `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.venv/bin/python -m src.component_cli register --evaluation /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/evaluation.json --snapshot /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/configuration.json --tracking-uri sqlite:////home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/mlflow.db --registration /home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z/registration.json` | 0 | 4.522 |

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

Underlying local database and full model artifacts are retained in `/home/usman/Desktop/mlops-k8s/lectures/lecture-04-kubeflow/bank-marketing-four-stage/.verification/local-20261013T110612Z`.
They are ignored by Git/build context but remain on disk. The reports contain no database
credentials. Exact raw commands for local stages are also serialized in execution.json.
