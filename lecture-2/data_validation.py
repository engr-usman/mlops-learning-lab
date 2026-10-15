"""Validate data.csv against the schema and rules defined in config.py.

Usage:
    python data_validation.py [path/to/data.csv]

Exits with status 1 if any check fails, so it can gate a pipeline step.
"""

import sys
from pathlib import Path

import pandas as pd

import config


def check_columns(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in config.COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.COLUMNS]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    if not missing and not extra and list(df.columns) != config.COLUMNS:
        errors.append("Columns are present but not in the expected order")
    return errors


def check_row_count(df: pd.DataFrame) -> list[str]:
    if len(df) < config.MIN_ROWS:
        return [f"Only {len(df)} rows, expected at least {config.MIN_ROWS}"]
    return []


def check_nulls(df: pd.DataFrame) -> list[str]:
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    return [f"Null values in '{col}': {n}" for col, n in nulls.items()]


def check_numeric(df: pd.DataFrame) -> list[str]:
    errors = []
    for col, (low, high) in config.NUMERIC_FEATURES.items():
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
    allowed_by_col = {**config.CATEGORICAL_FEATURES, config.TARGET: config.TARGET_CLASSES}
    for col, allowed in allowed_by_col.items():
        if col not in df.columns:
            continue
        invalid = set(df[col].dropna().unique()) - set(allowed)
        if invalid:
            errors.append(f"'{col}' has unexpected values: {sorted(invalid)}")
    return errors


def check_target(df: pd.DataFrame) -> list[str]:
    if config.TARGET not in df.columns:
        return []
    present = set(df[config.TARGET].dropna().unique())
    missing = set(config.TARGET_CLASSES) - present
    if missing:
        return [f"Target '{config.TARGET}' is missing classes: {sorted(missing)}"]
    return []


def check_duplicates(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    n_dup = int(df.duplicated().sum())
    fraction = n_dup / len(df) if len(df) else 0.0
    message = f"{n_dup} duplicate rows ({fraction:.2%})"
    if fraction > config.MAX_DUPLICATE_FRACTION:
        return [message + f", above limit of {config.MAX_DUPLICATE_FRACTION:.2%}"], []
    return [], [message] if n_dup else []


def validate(path: Path) -> bool:
    if not path.exists():
        print(f"FAIL: file not found: {path}")
        return False

    df = pd.read_csv(path)
    print(f"Validating {path} ({len(df)} rows, {len(df.columns)} columns)")

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

    if config.TARGET in df.columns:
        dist = df[config.TARGET].value_counts(normalize=True).round(3).to_dict()
        print(f"Target distribution: {dist}")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"FAIL: {e}")

    if errors:
        print(f"Validation failed with {len(errors)} error(s)")
        return False
    print("Validation passed")
    return True


if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_PATH
    sys.exit(0 if validate(data_path) else 1)
