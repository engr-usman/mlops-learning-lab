"""Container entrypoint: python -m src.component_cli STAGE --help."""

from __future__ import annotations

import argparse
import logging
from time import perf_counter

from src.artifact_io import publish_metadata, write_json
from src.logging_config import run_logging_context, setup_logging
from src.stages import evaluate_stage, register_stage, train_stage, validate_stage


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="stage", required=True)
    contracts = {
        "validate": ("dataset", "config", "validated", "snapshot", "validation"),
        "train": ("validated", "snapshot", "validation", "tracking-uri", "execution-id",
                  "manifest", "heldout", "split"),
        "evaluate": ("manifest", "heldout", "split", "snapshot", "tracking-uri", "evaluation", "metrics"),
        "register": ("evaluation", "snapshot", "tracking-uri", "registration"),
    }
    for stage, names in contracts.items():
        command = commands.add_parser(stage)
        for name in names:
            command.add_argument(f"--{name}", required=True)
        command.add_argument("--executor-input", help="KFP launcher JSON; omit for local execution")
    return result


def main() -> None:
    args = parser().parse_args()
    setup_logging()
    started = perf_counter()
    with run_logging_context(getattr(args, "execution_id", args.stage)):
        logging.info("COMPONENT | stage=%s | state=RUNNING", args.stage)
        try:
            if args.stage == "validate":
                result = validate_stage(args.dataset, args.config, args.validated, args.snapshot, args.validation)
                metadata = {"validated_dataset": result, "configuration": {"sha256": result["config_sha256"]},
                            "validation_report": result}
            elif args.stage == "train":
                result = train_stage(args.validated, args.snapshot, args.validation, args.tracking_uri,
                                     args.execution_id, args.manifest, args.heldout, args.split)
                metadata = {"candidate_manifest": {"parent_run_id": result["parent_run_id"]},
                            "heldout_dataset": {"sha256": result["heldout_sha256"]},
                            "split_metadata": {"sha256": result["split_metadata_sha256"]}}
            elif args.stage == "evaluate":
                result = evaluate_stage(args.manifest, args.heldout, args.split, args.snapshot,
                                        args.tracking_uri, args.evaluation)
                # Output[Metrics] uses numerical metadata for the UI; also persist JSON bytes.
                metrics = {f"{c['model_name']}_{name}": value
                           for c in result["candidates"] for name, value in c["metrics"].items()}
                metrics["quality_gate_passed"] = float(result["quality_gate"]["passed"])
                write_json(args.metrics, metrics)
                metadata = {"evaluation_report": {"quality_gate_passed": result["quality_gate"]["passed"],
                                                   "parent_run_id": result["parent_run_id"]},
                            "evaluation_metrics": metrics}
            else:
                result = register_stage(args.evaluation, args.snapshot, args.tracking_uri, args.registration)
                metadata = {"registration_result": {k: v for k, v in result.items() if v is not None}}
            publish_metadata(args.executor_input, metadata)
        except Exception:
            logging.exception("COMPONENT | stage=%s | state=FAILED | seconds=%.3f",
                              args.stage, perf_counter() - started)
            raise
        logging.info("COMPONENT | stage=%s | state=SUCCEEDED | seconds=%.3f",
                     args.stage, perf_counter() - started)


if __name__ == "__main__":
    main()
