from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#known leakage columns (do not train  on them)
DEFAULT_LEAKAGE_COLS = ["Churn Reason", "Churn Score", "CLTV","Churn Label",]

# Location/admin columns that don't generalize well (do NOT train on them)
DEFAULT_LOCATION_COLS = [
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

# Columns that are typically constant/useless
DEFAULT_CONSTANT_COLS = [
    "Count",  # often always 1
]

@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def split_X_y(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    leakage_cols: Optional[List[str]] = None,
    location_cols: Optional[List[str]] = None,
    constant_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into:
    - X (features)
    - y (target)

    Drops:
    - target column itself
    - id columns (e.g., CustomerID)
    - leakage columns (e.g., Churn Reason, Churn Score, CLTV)
    - location/admin columns (Country/State/City/Zip/Lat/Long)
    - constant/useless columns (e.g., Count)
    """
    df = normalize_columns(df)

    # IMPORTANT: use distinct local names to avoid any shadowing issues
    _id_cols = id_cols or []
    _leakage_cols = leakage_cols or DEFAULT_LEAKAGE_COLS
    _location_cols = location_cols or DEFAULT_LOCATION_COLS
    _constant_cols = constant_cols or DEFAULT_CONSTANT_COLS

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns.")

    candidates_to_drop = [target_col] + _id_cols + _leakage_cols + _location_cols + _constant_cols
    drop_cols = [c for c in candidates_to_drop if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    return X, y



def infer_feature_spec(X: pd.DataFrame) -> FeatureSpec:
    """
    Infer numeric and categorical columns from the feature dataframe X.

    - numeric columns are dtype 'number'
    - everything else is treated as categorical
    """
    X2 = X.dropna(axis=1, how="all")

    numeric_cols = X2.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return FeatureSpec(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def build_preprocess_pipeline(spec: FeatureSpec) -> ColumnTransformer:
    """
    Numeric columns:
    - impute missing values with median
    - scale (StandardScaler)

    Categorical columns:
    - impute missing values with most frequent
    - one-hot encode categories (handle_unknown='ignore')

    Output:
    - A ColumnTransformer that transforms X into a model-ready numeric matrix
    """

    # Pipeline for numeric features
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Apply numeric pipeline to numeric columns and categorical pipeline to categorical columns
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_cols),
            ("cat", categorical_pipeline, spec.categorical_cols),
        ],
        remainder="drop",  # drop any columns not listed above
    )

    return preprocess
    
