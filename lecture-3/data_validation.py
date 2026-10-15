"""Validate data.csv against the schema and rules defined in config.yaml.

Usage:
    python data_validation.py [path/to/data.csv]

Exits with status 1 if any check fails, so it can gate a pipeline step.
"""

import sys
from pathlib import Path

import pandas as pd

from config_loader import load_config
from logger import get_logger

config = load_config()
logger = get_logger("data_validation")
TARGET = config["target"]["name"]
TARGET_CLASSES = config["target"]["classes"]
COLUMNS = config["columns"]


def check_columns(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    if not missing and not extra and list(df.columns) != COLUMNS:
        errors.append("Columns are present but not in the expected order")
    return errors


def check_row_count(df: pd.DataFrame) -> list[str]:
    min_rows = config["validation"]["min_rows"]
    if len(df) < min_rows:
        return [f"Only {len(df)} rows, expected at least {min_rows}"]
    return []


def check_nulls(df: pd.DataFrame) -> list[str]:
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    return [f"Null values in '{col}': {n}" for col, n in nulls.items()]


def check_numeric(df: pd.DataFrame) -> list[str]:
    errors = []
    for col, (low, high) in config["numeric_features"].items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_numeric = values.isna() & df[col].notna()
        if non_numeric.any():
            errors.append(f"'{col}' has {non_numeric.sum()} non-numeric values")
        out_of_range = (values < low) | (values > high)
        if out_of_range.any():
            errors.append(
                f"'{col}' has {out_of_range.sum()} values outside [{low}, {high}]"
                f" (observed min={values.min()}, max={values.max()})"
            )
    return errors


def check_categorical(df: pd.DataFrame) -> list[str]:
    errors = []
    allowed_by_col = {**config["categorical_features"], TARGET: TARGET_CLASSES}
    for col, allowed in allowed_by_col.items():
        if col not in df.columns:
            continue
        invalid = set(df[col].dropna().unique()) - set(allowed)
        if invalid:
            errors.append(f"'{col}' has unexpected values: {sorted(invalid)}")
    return errors


def check_target(df: pd.DataFrame) -> list[str]:
    if TARGET not in df.columns:
        return []
    present = set(df[TARGET].dropna().unique())
    missing = set(TARGET_CLASSES) - present
    if missing:
        return [f"Target '{TARGET}' is missing classes: {sorted(missing)}"]
    return []


def check_duplicates(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    max_fraction = config["validation"]["max_duplicate_fraction"]
    n_dup = int(df.duplicated().sum())
    fraction = n_dup / len(df) if len(df) else 0.0
    message = f"{n_dup} duplicate rows ({fraction:.2%})"
    if fraction > max_fraction:
        return [message + f", above limit of {max_fraction:.2%}"], []
    return [], [message] if n_dup else []


def check_unknowns(df: pd.DataFrame) -> list[str]:
    """Report how often each categorical column uses "unknown", the dataset's missing-value marker."""
    warnings = []
    for col in config["categorical_features"]:
        if col not in df.columns:
            continue
        n_unknown = int((df[col] == "unknown").sum())
        if n_unknown:
            warnings.append(f"'{col}' has {n_unknown} 'unknown' values ({n_unknown / len(df):.2%})")
    return warnings


def validate(path: Path) -> bool:
    if not path.exists():
        logger.error("File not found: %s", path)
        return False

    df = pd.read_csv(path)
    logger.info("Validating %s (%d rows, %d columns)", path, len(df), len(df.columns))

    errors: list[str] = []
    warnings: list[str] = []
    errors += check_columns(df)
    errors += check_row_count(df)
    errors += check_nulls(df)
    errors += check_numeric(df)
    errors += check_categorical(df)
    errors += check_target(df)
    dup_errors, dup_warnings = check_duplicates(df)
    errors += dup_errors
    warnings += dup_warnings
    warnings += check_unknowns(df)

    if TARGET in df.columns:
        dist = df[TARGET].value_counts(normalize=True).round(3).to_dict()
        logger.info("Target distribution: %s", dist)

    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)

    if errors:
        logger.error("Validation failed with %d error(s)", len(errors))
        return False
    logger.info("Validation passed")
    return True


if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else config["paths"]["data_path"]
    sys.exit(0 if validate(data_path) else 1)
