"""Inspect actual compiled KFP task contracts, image, caching and resources."""

import json
from pathlib import Path
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spec, platform = list(yaml.safe_load_all((root / "pipelines/compiled/bank_marketing_four_stage_v1.yaml").read_text()))
    tasks = spec["root"]["dag"]["tasks"]
    expected = {"validate-dataset", "train-candidates", "evaluate-candidates", "register-winner"}
    assert set(tasks) == expected, tasks.keys()
    dependencies = {
        "train-candidates": {"validated_dataset": ("validate-dataset", "validated_dataset"),
                             "configuration": ("validate-dataset", "configuration"),
                             "validation_report": ("validate-dataset", "validation_report")},
        "evaluate-candidates": {"candidate_manifest": ("train-candidates", "candidate_manifest"),
                                "heldout_dataset": ("train-candidates", "heldout_dataset"),
                                "split_metadata": ("train-candidates", "split_metadata"),
                                "configuration": ("validate-dataset", "configuration")},
        "register-winner": {"evaluation_report": ("evaluate-candidates", "evaluation_report"),
                            "configuration": ("validate-dataset", "configuration")},
    }
    for task, inputs in dependencies.items():
        for name, (producer, output) in inputs.items():
            connection = tasks[task]["inputs"]["artifacts"][name]["taskOutputArtifact"]
            assert connection == {"producerTask": producer, "outputArtifactKey": output}
    images = set()
    for task_name, task in tasks.items():
        assert task.get("cachingOptions", {}).get("enableCache", False) is False
        component = spec["components"][task["componentRef"]["name"]]
        executor = component["executorLabel"]
        container = spec["deploymentSpec"]["executors"][executor]["container"]
        images.add(container["image"])
        assert container["command"] == ["python", "-m", "src.component_cli"]
        resources = container["resources"]
        assert all(key in resources for key in ("resourceCpuRequest", "resourceCpuLimit", "resourceMemoryRequest", "resourceMemoryLimit"))
        assert platform["platforms"]["kubernetes"]["deploymentSpec"]["executors"][executor]["imagePullPolicy"] == "IfNotPresent"
    assert len(images) == 1
    components = spec["components"]
    assert components["comp-validate-dataset"]["outputDefinitions"]["artifacts"]["validated_dataset"]["artifactType"]["schemaTitle"] == "system.Dataset"
    assert components["comp-train-candidates"]["outputDefinitions"]["artifacts"]["heldout_dataset"]["artifactType"]["schemaTitle"] == "system.Dataset"
    assert components["comp-evaluate-candidates"]["outputDefinitions"]["artifacts"]["evaluation_metrics"]["artifactType"]["schemaTitle"] == "system.Metrics"
    result = {"status": "PASS_SPEC_ONLY", "tasks": sorted(tasks), "images": sorted(images),
              "artifact_connections": dependencies, "sdk_version": spec["sdkVersion"],
              "schema_version": spec["schemaVersion"], "pull_policy": "IfNotPresent"}
    (root / "docs/evidence/spec-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
