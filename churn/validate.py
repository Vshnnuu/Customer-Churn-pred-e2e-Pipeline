from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_schema(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Check that the dataframe contains required columns and is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    target_col : str
        The target column name (for us: 'Churn Value').
    id_cols : list[str] | None
        Optional columns that identify rows (for us: ['CustomerID']).

    Returns
    -------
    ValidationResult
        ok=True if everything looks good, otherwise ok=False with errors listed.
    """
    errors: List[str] = []
    id_cols = id_cols or []

    # 1) Data must exist
    if df is None or len(df) == 0:
        errors.append("Dataframe is empty (0 rows).")
        return ValidationResult(ok=False, errors=errors)

    # 2) Duplicate column names can break selection later
    if df.columns.duplicated().any():
        errors.append("Duplicate column names found in dataset.")

    # 3) Target column must exist
    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset.")

    # 4) ID columns (if given) must exist
    for col in id_cols:
        if col not in df.columns:
            errors.append(f"ID column '{col}' not found in dataset.")

    return ValidationResult(ok=len(errors) == 0, errors=errors)


def validate_target_binary(
    df: pd.DataFrame,
    target_col: str,
) -> ValidationResult:
    errors: List[str] = []

    if target_col not in df.columns:
        return ValidationResult(ok=False, errors=[f"Target column '{target_col}' missing."])

    # Remove missing values and get unique remaining values
    unique_vals = pd.Series(df[target_col]).dropna().unique()

    if len(unique_vals) < 2:
        errors.append(
            f"Target '{target_col}' has < 2 unique values: {list(unique_vals)}. "
            "Binary classification needs two classes."
        )

    return ValidationResult(ok=len(errors) == 0, errors=errors)


def validate_no_leakage_columns(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
) -> ValidationResult: 
    leakage_cols = leakage_cols or ["Churn Reason", "Churn Score"]
    found = [c for c in leakage_cols if c in df.columns]

    if found:
        return ValidationResult(
            ok=False,
            errors=[
                "Potential data leakage columns found: "
                + ", ".join(found)
                + ". These should be excluded from features."
            ],
        )

    return ValidationResult(ok=True, errors=[])

