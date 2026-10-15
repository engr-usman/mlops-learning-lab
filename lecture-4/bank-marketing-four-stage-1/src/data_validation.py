from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

EXPECTED_CATEGORIES = {
    "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"},
    "marital": {"divorced", "married", "single"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "default": {"yes", "no"},
    "housing": {"yes", "no"},
    "loan": {"yes", "no"},
    "contact": {"cellular", "telephone", "unknown"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
              "sep", "oct", "nov", "dec"},
    "poutcome": {"failure", "other", "success", "unknown"},
}

EXPECTED_TARGET_VALUES = {"yes", "no"}
NUMERIC_FEATURES = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}


def validate_dataset(
    data_path,
    features,
    target,
    numeric_features=None,
    target_values=None,
    test_size=None,
):
    logger.info("DATA_VALIDATION | Started | path=%s", data_path)

    data_path = Path(data_path)

    if not data_path.exists():
        logger.error("DATA_VALIDATION | Failed: dataset not found | path=%s", data_path)
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )
    logger.debug("DATA_VALIDATION | Dataset found | path=%s", data_path)

    logger.info("DATA_VALIDATION | Reading dataset")
    df = pd.read_csv(data_path)

    logger.info("DATA_VALIDATION | Dataset loaded | rows=%d | columns=%d", *df.shape)

    required_columns = set(features) | {target}

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        logger.error(
            "DATA_VALIDATION | Failed: missing required columns | columns=%s",
            sorted(missing_columns),
        )
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        logger.error("DATA_VALIDATION | Failed: dataset is empty")
        raise ValueError(
            "Dataset is empty."
        )

    selected_columns = features + [target]

    if df[selected_columns].isnull().any().any():
        logger.error(
            "DATA_VALIDATION | Failed: missing values | columns=%s",
            df[selected_columns].columns[df[selected_columns].isnull().any()].tolist(),
        )
        raise ValueError(
            "Dataset contains missing values."
        )

    for feature in features:

        if feature in EXPECTED_CATEGORIES:

            invalid_values = set(
                df[feature].unique()
            ) - EXPECTED_CATEGORIES[feature]

            if invalid_values:
                logger.error(
                    "DATA_VALIDATION | Failed: invalid categories | feature=%s | count=%d",
                    feature, len(invalid_values),
                )
                raise ValueError(
                    f"Invalid values in {feature}: "
                    f"{sorted(invalid_values)}"
                )

    numeric_columns = NUMERIC_FEATURES.intersection(features) if numeric_features is None else numeric_features
    for feature in numeric_columns:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any() or not values.map(lambda value: float("-inf") < value < float("inf")).all():
            raise ValueError(f"Invalid numeric values in {feature}")
        df[feature] = values

    invalid_targets = (
        set(df[target].unique())
        - set(EXPECTED_TARGET_VALUES if target_values is None else target_values)
    )

    if invalid_targets:
        logger.error("DATA_VALIDATION | Failed: invalid targets | count=%d", len(invalid_targets))
        raise ValueError(
            "Invalid target values: "
            f"{sorted(invalid_targets)}"
        )

    if len(df) < 5:
        logger.error("DATA_VALIDATION | Failed: too few rows | rows=%d | minimum_rows=5", len(df))
        raise ValueError(
            "Dataset has too few rows."
        )
    counts = df[target].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Dataset needs both target classes with at least two rows per class.")
    if test_size is not None:
        from math import ceil
        test_rows = ceil(len(df) * test_size)
        if test_rows < len(counts) or len(df) - test_rows < len(counts):
            raise ValueError("Train and test splits must each have enough rows for both classes.")
    logger.info("DATA_VALIDATION | Passed | rows=%d | features=%d", len(df), len(features))
    return df
