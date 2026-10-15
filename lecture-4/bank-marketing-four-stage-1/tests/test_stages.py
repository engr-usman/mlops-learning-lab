"""Meaningful unit tests for bank rules, artifact integrity and retry behavior."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.artifact_io import check_hash, publish_metadata, sha256, write_json
from src.config import load_config
from src.data_validation import validate_dataset
from src.evaluation import evaluate_model, select_best_model
from src.stages import check_config, find_registered_version, index_hash, register_stage, validate_stage
from src.training import create_pipeline, split_dataset


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config()
        full = pd.read_csv("data/balance.csv")
        self.frame = pd.concat([full[full.y == label].head(10) for label in ("no", "yes")], ignore_index=True)

    def validate(self, frame):
        path = self.root / "dataset.csv"
        frame.to_csv(path, index=False)
        data = self.config["data"]
        return validate_dataset(path, data["features"], data["target"], data["numeric_features"],
                                data["target_values"], self.config["training"]["test_size"])

    def test_valid_dataset(self):
        self.assertEqual(len(self.validate(self.frame)), 20)

    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            self.validate(self.frame.drop(columns="age"))

    def test_invalid_category(self):
        frame = self.frame.copy(); frame.loc[0, "job"] = "not-a-bank-job"
        with self.assertRaisesRegex(ValueError, "Invalid values"):
            self.validate(frame)

    def test_missing_feature(self):
        frame = self.frame.copy(); frame.loc[0, "job"] = None
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.validate(frame)

    def test_invalid_numeric(self):
        frame = self.frame.copy(); frame["age"] = frame.age.astype(object); frame.loc[0, "age"] = "bad"
        with self.assertRaisesRegex(ValueError, "Invalid numeric"):
            self.validate(frame)

    def test_nonfinite_numeric(self):
        frame = self.frame.copy(); frame["balance"] = float("inf")
        with self.assertRaisesRegex(ValueError, "Invalid numeric"):
            self.validate(frame)

    def test_missing_target_column(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            self.validate(self.frame.drop(columns="y"))

    def test_missing_target_value(self):
        frame = self.frame.copy(); frame.loc[0, "y"] = None
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.validate(frame)

    def test_invalid_target(self):
        frame = self.frame.copy(); frame.loc[0, "y"] = "YES"
        with self.assertRaisesRegex(ValueError, "Invalid target"):
            self.validate(frame)

    def test_insufficient_class_distribution(self):
        with self.assertRaisesRegex(ValueError, "both target classes"):
            self.validate(self.frame[self.frame.y == "no"])

    def test_single_minority_row(self):
        with self.assertRaisesRegex(ValueError, "both target classes"):
            self.validate(pd.concat([self.frame[self.frame.y == "no"], self.frame[self.frame.y == "yes"].head(1)]))

    def test_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.validate(self.frame.head(0))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            validate_dataset(self.root / "absent.csv", ["age"], "y")

    def test_split_infeasible(self):
        frame = self.frame.head(0)
        frame = pd.concat([self.frame[self.frame.y == label].head(3) for label in ("no", "yes")])
        self.config["training"]["test_size"] = 0.01
        with self.assertRaisesRegex(ValueError, "splits"):
            self.validate(frame)

    def test_artifact_metadata_and_hash(self):
        source = self.root / "source.csv"; self.frame.to_csv(source, index=False)
        validated = self.root / "validated.csv"; snapshot = self.root / "config.json"
        result = validate_stage(str(source), "configs/config.yaml", str(validated),
                                str(snapshot), str(self.root / "report.json"))
        self.assertEqual(result["source_dataset_sha256"], sha256(source))
        self.assertEqual(result["validated_dataset_sha256"], sha256(validated))
        self.assertEqual(result["rows"], 20)
        self.assertEqual(result["validation_status"], "PASSED")


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.frame = pd.read_csv("data/balance.csv")

    def test_same_seed_same_split(self):
        a = split_dataset(self.frame, self.config); b = split_dataset(self.frame, self.config)
        for left, right in zip(a, b):
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right)
            else:
                pd.testing.assert_series_equal(left, right)

    def test_split_disjoint_and_complete(self):
        X_train, X_test, _, _ = split_dataset(self.frame, self.config)
        self.assertFalse(set(X_train.index) & set(X_test.index))
        self.assertEqual(set(X_train.index) | set(X_test.index), set(self.frame.index))

    def test_preprocessor_fits_training_only(self):
        config = copy.deepcopy(self.config)
        config["data"].update(features=["age", "job"], numeric_features=["age"], categorical_features=["job"])
        training = pd.DataFrame({"age": [10, 20, 30, 40], "job": ["admin."] * 4})
        model = create_pipeline(config, "logistic_regression").fit(training, ["no", "yes", "no", "yes"])
        preprocessing = model.named_steps["preprocessor"]
        self.assertEqual(preprocessing.named_transformers_["numeric"].mean_[0], 25)
        self.assertEqual(preprocessing.named_transformers_["categorical"].categories_[0].tolist(), ["admin."])
        model.predict(pd.DataFrame({"age": [10000], "job": ["student"]}))
        self.assertEqual(preprocessing.named_transformers_["numeric"].mean_[0], 25)

    def test_feature_partition_invalid(self):
        self.config["data"]["numeric_features"].append("job")
        with self.assertRaisesRegex(ValueError, "partition"):
            check_config(self.config)

    def test_target_not_a_feature(self):
        self.config["data"]["features"].append("y")
        with self.assertRaisesRegex(ValueError, "Target"):
            check_config(self.config)

    def test_wrong_positive_label_rejected(self):
        self.config["data"]["positive_label"] = "no"
        with self.assertRaisesRegex(ValueError, "positive"):
            check_config(self.config)

    def test_row_order_changes_hash(self):
        self.assertNotEqual(index_hash([1, 2, 3]), index_hash([3, 2, 1]))


class EvaluationTests(unittest.TestCase):
    @patch("src.evaluation.mlflow")
    def test_metrics_positive_yes_and_confusion_matrix(self, tracking):
        expected = ["yes", "yes", "no", "no", "yes"]
        predictions = ["yes", "no", "no", "yes", "yes"]
        model = MagicMock(); model.predict.return_value = predictions
        metrics = evaluate_model(model, pd.DataFrame({"x": range(5)}), expected)
        for key, value in {
            "accuracy": accuracy_score(expected, predictions),
            "precision": precision_score(expected, predictions, pos_label="yes"),
            "recall": recall_score(expected, predictions, pos_label="yes"),
            "f1_score": f1_score(expected, predictions, pos_label="yes"),
        }.items():
            self.assertAlmostEqual(metrics[key], value)
        self.assertEqual(tracking.log_dict.call_args.args[0]["counts"], [[1, 1], [1, 2]])

    def test_selection_gate_f1_accuracy_tie_order_and_all_rejected(self):
        config = load_config()
        def candidate(name, accuracy, f1):
            return {"model_name": name, "metrics": {"accuracy": accuracy, "f1_score": f1}}
        rejected = candidate("rejected", .69, .99)
        accurate = candidate("accurate", .95, .70)
        f1_winner = candidate("winner", .80, .85)
        tie = candidate("tie", .90, .85)
        exact = candidate("exact", .90, .85)
        self.assertIs(select_best_model([rejected, accurate, f1_winner], config), f1_winner)
        self.assertIs(select_best_model([f1_winner, tie], config), tie)
        self.assertIs(select_best_model([tie, exact], config), tie)
        self.assertIsNone(select_best_model([rejected], config))
        self.assertIsNotNone(select_best_model([candidate("boundary", .70, .60)], config))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "config.json"; write_json(self.config_path, load_config())
        self.candidate = {"model_name": "random_forest", "run_id": "child", "model_uri": "models:/m-actual",
                          "metrics": {"accuracy": .90, "f1_score": .91}, "eligible": True}
        self.evaluation = {"config_sha256": sha256(self.config_path), "parent_run_id": "parent",
                           "experiment_id": "7", "execution_id": "execution", "candidates": [self.candidate],
                           "selected_candidate": self.candidate,
                           "quality_gate": {"passed": True, "minimum_accuracy": .70, "selection_metric": "f1_score"}}

    def invoke(self, evaluation, existing=False):
        evaluation_path = self.root / "evaluation.json"; write_json(evaluation_path, evaluation)
        with patch("src.stages.MlflowClient") as factory, patch("src.stages.mlflow") as tracking:
            client = factory.return_value
            client.get_run.return_value.data.tags = {"mlflow.parentRunId": "parent"}
            version = SimpleNamespace(version="8", run_id="child", tags={})
            client.get_model_version.return_value = version
            tracking.register_model.return_value = version
            with patch("src.stages.find_registered_version", return_value=version if existing else None):
                result = register_stage(str(evaluation_path), str(self.config_path), "http://test",
                                        str(self.root / "registration.json"))
            return result, tracking, client

    def test_success_and_actual_source_uri(self):
        result, tracking, client = self.invoke(self.evaluation)
        self.assertEqual(result["status"], "registered")
        self.assertEqual(result["registered_model_version"], "8")
        self.assertEqual(tracking.register_model.call_args.args,
                         ("models:/m-actual", "bank-marketing-pipeline-model"))
        self.assertIn("four_stage_registration_key", tracking.register_model.call_args.kwargs["tags"])
        tracking.sklearn.load_model.assert_called_once_with("models:/bank-marketing-pipeline-model/8")

    def test_retry_reuses_version(self):
        result, tracking, _ = self.invoke(self.evaluation, existing=True)
        tracking.register_model.assert_not_called()
        self.assertTrue(result["reused_existing_version"])

    def test_rejected_does_not_register(self):
        candidate = {**self.candidate, "metrics": {"accuracy": .60, "f1_score": .95}, "eligible": False}
        evaluation = {**self.evaluation, "candidates": [candidate], "selected_candidate": None,
                      "quality_gate": {**self.evaluation["quality_gate"], "passed": False}}
        result, tracking, _ = self.invoke(evaluation)
        tracking.register_model.assert_not_called()
        self.assertEqual(result["status"], "not_registered")
        self.assertIsNone(result["registered_model_version"])

    def test_inconsistent_gate_rejected(self):
        evaluation = {**self.evaluation, "quality_gate": {**self.evaluation["quality_gate"], "passed": False}}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.invoke(evaluation)

    def test_registration_error_propagates(self):
        path = self.root / "evaluation.json"; write_json(path, self.evaluation)
        with patch("src.stages.MlflowClient") as factory, patch("src.stages.mlflow") as tracking, \
                patch("src.stages.find_registered_version", return_value=None):
            factory.return_value.get_run.return_value.data.tags = {"mlflow.parentRunId": "parent"}
            tracking.register_model.side_effect = RuntimeError("registry unavailable")
            with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
                register_stage(str(path), str(self.config_path), "http://test", str(self.root / "result.json"))
            self.assertFalse((self.root / "result.json").exists())

    def test_wrong_parent_lineage_rejected(self):
        path = self.root / "evaluation.json"; write_json(path, self.evaluation)
        with patch("src.stages.MlflowClient") as factory, patch("src.stages.mlflow") as tracking:
            factory.return_value.get_run.return_value.data.tags = {"mlflow.parentRunId": "other-parent"}
            with self.assertRaisesRegex(ValueError, "parent lineage"):
                register_stage(str(path), str(self.config_path), "http://test", str(self.root / "result.json"))
            tracking.register_model.assert_not_called()

    def test_version_search_checks_key_uri_source_and_pagination(self):
        class Page(list):
            def __init__(self, values, token=None):
                super().__init__(values); self.token = token
        wrong = SimpleNamespace(run_id="wrong", tags={"four_stage_registration_key": "key", "selected_model_uri": "uri"})
        right = SimpleNamespace(run_id="child", tags={"four_stage_registration_key": "key", "selected_model_uri": "uri"})
        client = MagicMock(); client.search_model_versions.side_effect = [Page([wrong], "next"), Page([right])]
        self.assertIs(find_registered_version(client, "name", "key", "child", "uri"), right)
        self.assertEqual(client.search_model_versions.call_args.kwargs["page_token"], "next")


class ArtifactTests(unittest.TestCase):
    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"; path.write_text("first")
            original = sha256(path); path.write_text("second")
            with self.assertRaisesRegex(ValueError, "integrity"):
                check_hash(path, original)

    def test_executor_metadata_preserves_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "executor_output.json")
            artifact = {"name": "123", "uri": "s3://test/path", "type": {"schemaTitle": "system.Dataset", "schemaVersion": "0.0.1"}}
            spec = {"outputs": {"outputFile": output, "artifacts": {"dataset": {"artifacts": [artifact]}}}}
            publish_metadata(json.dumps(spec), {"dataset": {"rows": 20}})
            value = json.loads(Path(output).read_text())["artifacts"]["dataset"]["artifacts"][0]
            self.assertEqual(value["uri"], artifact["uri"])
            self.assertEqual(value["type"], artifact["type"])
            self.assertEqual(value["metadata"]["rows"], 20)


if __name__ == "__main__":
    unittest.main()
