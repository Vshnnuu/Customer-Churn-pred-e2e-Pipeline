import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline

from churn.data import load_csv, basic_cleaning
from churn.features import split_X_y, infer_feature_spec, build_preprocess_pipeline
from churn.validate import validate_schema, validate_target_binary, validate_no_leakage_columns
from churn.evaluate import compute_binary_metrics, find_best_threshold


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str, n_iter: int, scoring: str, cv_splits: int) -> None:
    print("Starting LIGHT XGBoost tuning with cross-validation...")

    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    artifacts_cfg = cfg["artifacts"]
    seed = int(cfg["project"]["random_seed"])

    # Load and clean data
    df = load_csv(data_cfg["raw_csv_path"])
    df = basic_cleaning(df)

    # Validate data
    schema_result = validate_schema(
        df,
        target_col=data_cfg["target_col"],
        id_cols=data_cfg.get("id_cols", []),
    )
    if not schema_result.ok:
        raise ValueError(f"Schema validation errors: {schema_result.errors}")

    target_result = validate_target_binary(df, target_col=data_cfg["target_col"])
    if not target_result.ok:
        raise ValueError(f"Target validation errors: {target_result.errors}")

    leakage_result = validate_no_leakage_columns(df)
    if not leakage_result.ok:
        print("WARNING:", leakage_result.errors[0])


    # Split X/y + train/test split
    X, y = split_X_y(
        df,
        target_col=data_cfg["target_col"],
        id_cols=data_cfg.get("id_cols", []),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_cfg["test_size"]),
        random_state=seed,
        stratify=y,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Build preprocessing pipeline
    spec = infer_feature_spec(X_train)
    preprocess = build_preprocess_pipeline(spec)

    # Base XGBoost model
    base_model = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", base_model),
        ]
    )

    # Hyperparameter search space (light)
    param_distributions = {
        "model__n_estimators": [300, 500, 700, 900],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 4, 5, 6],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "model__min_child_weight": [1, 3, 5, 10],
    }

    # Cross-validation + RandomizedSearchCV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        refit=True,
    )

    print(f"Tuning XGBoost with scoring='{scoring}', n_iter={n_iter}, cv={cv_splits}...")
    search.fit(X_train, y_train)

    print("Best CV score:", float(search.best_score_))
    print("Best params:", search.best_params_)

    best_pipeline = search.best_estimator_


    # Final evaluation on untouched test set
    print("Evaluating tuned XGBoost on test set...")
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    metrics_default = compute_binary_metrics(y_test, y_proba, threshold=0.5)
    best_t, metrics_best, sweep = find_best_threshold(y_test, y_proba, metric="f1")

    metrics_out = {
        "model": "xgboost_tuned_light",
        "cv_scoring": scoring,
        "best_cv_score": float(search.best_score_),
        "default": metrics_default,
        "best_f1": metrics_best,
        "best_threshold_f1": best_t,
        "threshold_sweep": sweep,
        "best_params": search.best_params_,
    }

    # Save artifacts
    model_dir = Path(artifacts_cfg["model_dir"])
    reports_dir = Path(artifacts_cfg["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model_xgb_tuned.joblib"
    metrics_path = reports_dir / "metrics_xgb_tuned.json"
    params_path = reports_dir / "best_params_xgb.json"
    cv_path = reports_dir / "xgb_cv_results.csv"

    joblib.dump(best_pipeline, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(params_path, "w") as f:
        json.dump(search.best_params_, f, indent=2)

    pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

    print("Light XGBoost tuning complete.")
    print(f"Tuned model saved to: {model_path}")
    print(f"Tuned metrics saved to: {metrics_path}")
    print(f"Best params saved to: {params_path}")
    print(f"CV results saved to: {cv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/train.yaml")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--scoring", type=str, default="average_precision")
    parser.add_argument("--cv_splits", type=int, default=3)
    args = parser.parse_args()

    main(args.config, n_iter=args.n_iter, scoring=args.scoring, cv_splits=args.cv_splits)
