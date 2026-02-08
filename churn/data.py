from pathlib import Path
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file from disk and return it as a Pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place your CSV file at this location."
        )

    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal, non-destructive cleaning on the dataset.

    What this function does:
    - Strips whitespace from column names
    - Strips whitespace from string values
    - Converts known numeric-like columns safely

    What this function does NOT do:
    - Feature engineering
    - Dropping rows aggressively
    - Encoding categories
    """

    cleaned_df = df.copy()

    # Clean column names: strip whitespace
    cleaned_df.columns = [col.strip() for col in cleaned_df.columns]

    # Strip whitespace from all strings
    object_columns = cleaned_df.select_dtypes(include="object").columns

    for col in object_columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # "TotalCharges" may appear as a string with empty values (Telco dataset issue)
    if "TotalCharges" in cleaned_df.columns:
        cleaned_df["TotalCharges"] = pd.to_numeric(
            cleaned_df["TotalCharges"], errors="coerce"
        )

    return cleaned_df
