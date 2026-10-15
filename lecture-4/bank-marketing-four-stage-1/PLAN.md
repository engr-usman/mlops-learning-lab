# Implementation plan

1. Fingerprint protected lectures and copy only seven required source modules, the
   active bank CSV, YAML configuration and application requirements.
2. Implement four independently executable container stages with typed KFP
   artifacts, immutable configuration snapshots and explicit MLflow run IDs.
3. Reuse validation, preprocessing, metrics, selection and registration rules;
   add artifact integrity checks and a best-effort registration retry key.
4. Test failure paths and reproducibility; execute the actual stages as independent
   local processes against isolated MLflow storage when dependencies permit.
5. Compile and inspect the four-stage KFP v2 spec. Attempt Docker, containerd,
   cluster execution and metadata inspection within available permissions.
6. Preserve logs and hashes; produce nine reports and CODEX_HANDOFF.md. Distinguish
   local verification from Kubernetes/MinIO verification.

Runtime restrictions discovered at inspection: Kubernetes networking, Docker and
containerd sockets are blocked. Sandbox escalation is prohibited by policy. No
shared infrastructure changes are authorized or attempted.
