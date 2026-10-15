"""Submit only after image/platform checks; use an existing authenticated KFP endpoint.

No credential is printed or saved. This script does not modify infrastructure.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import kfp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("KFP_HOST"), required=not bool(os.getenv("KFP_HOST")))
    parser.add_argument("--namespace", default="admin")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--tracking-uri", default="http://mlflow.mlops.svc.cluster.local:5000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    client = kfp.Client(host=args.host, namespace=args.namespace,
                        existing_token=os.getenv("KFP_TOKEN"), cookies=os.getenv("KFP_COOKIES"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = client.create_run_from_pipeline_package(
        str(root / "pipelines/compiled/bank_marketing_four_stage_v1.yaml"),
        arguments={"tracking_uri": args.tracking_uri}, namespace=args.namespace,
        experiment_name="lecture-04-bank-marketing-four-stage",
        run_name=f"four-stage-bank-{stamp}", enable_caching=False,
    )
    record = {"run_id": result.run_id, "namespace": args.namespace,
              "submitted_utc": stamp, "status": "SUBMITTED"}
    evidence = root / "docs/evidence" / f"kfp-run-{stamp}.json"
    evidence.write_text(json.dumps(record, indent=2))
    print(f"Submitted run {result.run_id}", flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        run = client.get_run(result.run_id)
        state = str(getattr(run, "state", "UNKNOWN"))
        record["status"] = state
        # API object representation excludes authentication settings.
        record["run"] = run.to_dict()
        evidence.write_text(json.dumps(record, indent=2, default=str))
        print(f"Run {result.run_id}: {state}", flush=True)
        if state in ("SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "SKIPPED"):
            if state != "SUCCEEDED":
                raise RuntimeError(f"KFP run ended in {state}; retained evidence: {evidence}")
            return
        time.sleep(10)
    raise TimeoutError(f"Monitoring timed out; run was not canceled/deleted: {result.run_id}")


if __name__ == "__main__":
    main()
