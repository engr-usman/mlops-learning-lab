"""Run four real CLI processes against NEW isolated SQLite/artifact storage.

This verifies Python/file/MLflow boundaries, not Kubernetes, MinIO or MLMD.
Never points at the shared mlops service and never modifies an existing database.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from mlflow import MlflowClient
import mlflow.sklearn
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_directory).resolve() if args.output_directory else project / ".verification" / f"local-{stamp}"
    root.mkdir(parents=True, exist_ok=False)  # Refuse to reuse databases/artifacts.
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    source = root / "source.csv"
    shutil.copy2(project / "data/balance.csv", source)
    uri = f"sqlite:///{root / 'mlflow.db'}"
    artifact_root = root / "mlflow-artifacts"; artifact_root.mkdir()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(config["mlflow"]["experiment_name"], artifact_location=artifact_root.as_uri())
    environment = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MPLCONFIGDIR": str(root / "matplotlib")}
    stages = [
        ("validate", ["--dataset", str(source), "--config", str(project / "configs/config.yaml"),
                      "--validated", str(root / "validated.csv"), "--snapshot", str(root / "configuration.json"),
                      "--validation", str(root / "validation.json")]),
        ("train", ["--validated", str(root / "validated.csv"), "--snapshot", str(root / "configuration.json"),
                   "--validation", str(root / "validation.json"), "--tracking-uri", uri,
                   "--execution-id", f"local-proof-{stamp}", "--manifest", str(root / "manifest.json"),
                   "--heldout", str(root / "heldout.csv"), "--split", str(root / "split.json")]),
        ("evaluate", ["--manifest", str(root / "manifest.json"), "--heldout", str(root / "heldout.csv"),
                      "--split", str(root / "split.json"), "--snapshot", str(root / "configuration.json"),
                      "--tracking-uri", uri, "--evaluation", str(root / "evaluation.json"),
                      "--metrics", str(root / "metrics.json")]),
        ("register", ["--evaluation", str(root / "evaluation.json"), "--snapshot", str(root / "configuration.json"),
                      "--tracking-uri", uri, "--registration", str(root / "registration.json")]),
    ]
    evidence = {"verification_scope": "local processes, files and isolated MLflow only",
                "started_utc": stamp, "tracking_uri": uri, "experiment_id": experiment_id, "stages": []}
    for stage, arguments in stages:
        command = [sys.executable, "-m", "src.component_cli", stage, *arguments]
        started = perf_counter()
        with (root / f"{stage}.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            exit_code = process.wait(timeout=600)
        evidence["stages"].append({"stage": stage, "pid": process.pid, "command": command,
                                   "exit_code": exit_code, "seconds": round(perf_counter() - started, 3)})
        (root / "execution.json").write_text(json.dumps(evidence, indent=2))
        print(f"{stage}: exit={exit_code} | seconds={evidence['stages'][-1]['seconds']}", flush=True)
        if exit_code:
            raise RuntimeError(f"{stage} failed; inspect {root / (stage + '.log')}")
        if stage == "validate":
            source.rename(root / "source.consumed.csv")  # Subsequent stages cannot reopen original input.

    registration = json.loads((root / "registration.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    assert len(manifest["candidates"]) == 4
    for candidate in manifest["candidates"]:
        run = client.get_run(candidate["run_id"])
        assert run.data.tags["mlflow.parentRunId"] == manifest["parent_run_id"]
        assert run.data.tags["heldout_sha256"] == manifest["heldout_sha256"]
        assert all(metric in run.data.metrics for metric in ("accuracy", "precision", "recall", "f1_score"))
        assert run.info.status == "FINISHED"
    assert client.get_run(manifest["parent_run_id"]).data.tags["workflow_state"] == "completed"

    # Retry registration only; do not retrain models just to exercise retry behavior.
    stage, arguments = stages[-1]
    with (root / "registration-retry.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "src.component_cli", stage, *arguments], cwd=project,
                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
    retried = json.loads((root / "registration.json").read_text())
    if registration["status"] == "registered":
        assert retried["reused_existing_version"] is True
        assert retried["registered_model_version"] == registration["registered_model_version"]
        versions = client.search_model_versions(f"name = '{registration['registered_model_name']}'")
        assert len(versions) == 1, "Sequential retry created a duplicate version"
        mlflow.set_tracking_uri(uri)
        model = mlflow.sklearn.load_model(f"models:/{registration['registered_model_name']}/{registration['registered_model_version']}")
        sample = pd.read_csv(root / "heldout.csv")[config["data"]["features"]].head(3)
        evidence["registered_model_predictions"] = model.predict(sample).tolist()
    evidence.update({"manifest": manifest, "evaluation": evaluation, "registration": registration,
                     "retry_registration": retried, "status": "PASS_LOCAL_ONLY"})
    (root / "execution.json").write_text(json.dumps(evidence, indent=2))
    destination = project / "docs/evidence/local-verification.json"
    destination.write_text(json.dumps({"root": str(root), **evidence}, indent=2))
    print(f"Local verification complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
