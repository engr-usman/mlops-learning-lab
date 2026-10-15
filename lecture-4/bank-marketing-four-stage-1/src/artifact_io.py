"""Small, explicit file contracts shared by independent component processes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    """Hash bytes without loading an entire dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write a complete output only after its producer has succeeded."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def check_hash(path: str | Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Artifact integrity check failed: {path}")


def publish_metadata(executor_input: str | None,
                     metadata: dict[str, dict[str, Any]]) -> None:
    """Publish container-component artifact metadata through KFP ExecutorOutput.

    KFP provides artifact URIs/types and the executor output path. Preserve those
    identities while adding JSON-compatible metadata. The launcher uploads files;
    this code never accesses MinIO credentials or buckets directly.
    """
    if executor_input is None:
        return  # Independent local process mode has no KFP launcher.
    spec = json.loads(executor_input)
    output = {"artifacts": {}}
    for name, values in metadata.items():
        artifacts = spec["outputs"]["artifacts"][name]["artifacts"]
        if len(artifacts) != 1:
            raise ValueError(f"Expected one output artifact: {name}")
        artifact = dict(artifacts[0])
        artifact["metadata"] = {**artifact.get("metadata", {}), **values}
        output["artifacts"][name] = {"artifacts": [artifact]}
    write_json(spec["outputs"]["outputFile"], output)
