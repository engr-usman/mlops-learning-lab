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
