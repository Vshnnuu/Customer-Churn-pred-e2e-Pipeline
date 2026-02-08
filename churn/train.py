import argparse
from pathlib import Path
import json
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from churn.data import load_csv, basic_cleaning
from churn.validate import (
    validate_schema,
    validate_target_binary,
    validate_no_leakage_columns,
)
from churn.features import (
    split_X_y,
    infer_feature_spec,
    build_preprocess_pipeline,
)
from churn.evaluate import compute_binary_metrics, find_best_threshold


# Utility
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_model(model_cfg: dict):
    model_type = model_cfg["type"].lower()
    params = model_cfg["params"]

    if model_type == "logistic_regression":
        p = params["logistic_regression"]
        return LogisticRegression(
            max_iter=p.get("max_iter", 300),
            class_weight=p.get("class_weight", None),
        )

    if model_type == "random_forest":
        p = params["random_forest"]
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_split=p.get("min_samples_split", 2),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            class_weight=p.get("class_weight", None),
            random_state=p.get("random_state", 42),
            n_jobs=p.get("n_jobs", -1),
        )
    
    if model_type == "gradient_boosting":
        p = params["gradient_boosting"]
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0),
            random_state=p.get("random_state", 42),
        )
    
    if model_type == "xgboost":
        p = params["xgboost"]
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 500),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 4),
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0),
            random_state=p.get("random_state", 42),
            eval_metric="logloss",
            n_jobs=-1,
        )



    raise ValueError(f"Unknown model type: {model_type}")

# Main training function
def main(config_path: str) -> None:
    print("Starting churn training pipeline...")

    # 1) Load config
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    artifacts_cfg = cfg["artifacts"]

    # 2) Load data
    print("Loading data...")
    df = load_csv(data_cfg["raw_csv_path"])
    df = basic_cleaning(df)

    # 3) Validate data
    print("Validating data...")

    schema_result = validate_schema(
        df,
        target_col=data_cfg["target_col"],
        id_cols=data_cfg["id_cols"],
    )
    if not schema_result.ok:
        raise ValueError(schema_result.errors)

    target_result = validate_target_binary(
        df, target_col=data_cfg["target_col"]
    )
    if not target_result.ok:
        raise ValueError(target_result.errors)

    leakage_result = validate_no_leakage_columns(df)
    if not leakage_result.ok:
        print("WARNING:", leakage_result.errors[0])

    # 4) Split features and target
    X, y = split_X_y(
        df,
        target_col=data_cfg["target_col"],
        id_cols=data_cfg["id_cols"],
    )

    # 5) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # 6) Build preprocessing
    feature_spec = infer_feature_spec(X_train)
    preprocess = build_preprocess_pipeline(feature_spec)

    # 7) Build model
    model = build_model(model_cfg)
    print(f"Using model: {model_cfg['type']}")


    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    # 8) Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 9) Evaluate
    print("Evaluating model...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 1) Metrics at configured threshold
    default_threshold = float(train_cfg["threshold"])
    metrics_default = compute_binary_metrics(
    y_true=y_test,
    y_proba=y_proba,
    threshold=default_threshold,
)
    # 2) Find best threshold for F1 
    best_t, metrics_best, sweep = find_best_threshold(
    y_true=y_test,
    y_proba=y_proba,
    metric="f1",
)

    print(f"Default threshold = {default_threshold:.2f}, F1 = {metrics_default['f1']:.3f}")
    print(f"Best threshold (F1) = {best_t:.2f}, F1 = {metrics_best['f1']:.3f}")

    # Save richer metrics object
    metrics = {
    "default": metrics_default,
    "best_f1": metrics_best,
    "best_threshold_f1": best_t,
    "threshold_sweep": sweep,  
}


    # 10) Save artifacts
    model_dir = Path(artifacts_cfg["model_dir"])
    reports_dir = Path(artifacts_cfg["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / artifacts_cfg["model_filename"]
    metrics_path = reports_dir / artifacts_cfg["metrics_filename"]

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML file",
    )
    args = parser.parse_args()

    main(args.config)
