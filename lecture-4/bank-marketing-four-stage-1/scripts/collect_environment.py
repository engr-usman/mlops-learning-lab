"""Record bounded, non-destructive infrastructure checks without reading Secrets."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    workspace = project.parents[2]
    evidence = project / "docs/evidence"
    kube = ["/snap/microk8s/current/kubectl", "--kubeconfig",
            str(workspace / "infrastructure/juju/microk8s-kubeconfig.yaml"), "--request-timeout=5s"]
    checks = {
        "microk8s-status": ["microk8s", "status"],
        "juju-status": ["juju", "status", "--format", "short"],
        "docker-images": ["docker", "image", "inspect", "bank-marketing-kfp-four-stage:v1", "--format", "{{.Id}}"],
        "nodes": [*kube, "get", "nodes", "-o", "wide"],
        "kubeflow-health": [*kube, "get", "pods", "-n", "kubeflow", "-o", "wide"],
        "admin-workflows": [*kube, "get", "workflows", "-n", "admin"],
        "mlflow-health": [*kube, "get", "pods,svc,pvc", "-n", "mlops"],
        "metadata-writer-logs": [*kube, "logs", "kfp-metadata-writer-0", "-n", "kubeflow", "--tail=100", "--since=1h"],
        "mysql-logs": [*kube, "logs", "mysql-db-0", "-n", "kubeflow", "--tail=100", "--since=1h"],
        "metadata-grpc-logs": [*kube, "logs", "deployment/metadata-grpc-service", "-n", "kubeflow", "--tail=100", "--since=1h"],
        "ml-pipeline-logs": [*kube, "logs", "deployment/ml-pipeline", "-n", "kubeflow", "--tail=100", "--since=1h"],
        "containerd-images": ["/snap/microk8s/current/bin/ctr", "--address", "/var/snap/microk8s/common/run/containerd.sock",
                              "--namespace", "k8s.io", "images", "list"],
        "docker-build": ["docker", "build", "-t", "bank-marketing-kfp-four-stage:v1", "."],
    }
    records = []
    for name, command in checks.items():
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr; exit_code = result.returncode
        except subprocess.TimeoutExpired:
            output = "Timed out after 30 seconds; no infrastructure was restarted."; exit_code = 124
        (evidence / f"environment-{name}.log").write_text(output)
        records.append({"check": name, "utc": stamp, "command": command,
                        "exit_code": exit_code, "log": f"environment-{name}.log"})
        print(f"{name}: exit={exit_code}", flush=True)
    (evidence / "environment-checks.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
