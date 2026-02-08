import argparse
from pathlib import Path
import shap
import numpy as np


import joblib
import pandas as pd
import matplotlib.pyplot as plt

from churn.data import load_csv, basic_cleaning
from churn.features import split_X_y


def main(config_path: str, model_path: str) -> None:
    # Load config (simple YAML read without extra dependencies)
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    reports_dir = Path(cfg["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_csv(data_cfg["raw_csv_path"])
    df = basic_cleaning(df)

    # Split X/y (same logic as training)
    X, y = split_X_y(
        df,
        target_col=data_cfg["target_col"],
        id_cols=data_cfg.get("id_cols", []),
    )

    # Load trained pipeline (preprocess + model)
    pipeline = joblib.load(model_path)

    # --- Feature importance (fast + recruiter-friendly) ---
    # After preprocessing, feature names change because of one-hot encoding.
    # We'll extract transformed feature names from the preprocess step.
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()

    # XGBoost has feature_importances_
    importances = model.feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(25)
    )

    # Save as CSV for docs
    fi_path = reports_dir / "feature_importance_top25.csv"
    fi.to_csv(fi_path, index=False)

    # Plot
    plt.figure()
    features = fi["feature"].tolist()[::-1]
    importances = fi["importance"].tolist()[::-1]
    plt.barh(features, importances)

    plt.xlabel("Importance")
    plt.title("Top 25 Feature Importances (XGBoost)")
    plot_path = reports_dir / "feature_importance_top25.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved: {fi_path}")
    print(f"Saved: {plot_path}")


    # SHAP EXPLAINABILITY
    print("Computing SHAP values...")
    X_sample = X.sample(n=min(500, len(X)), random_state=42)

    # Transform data using the same preprocessing
    X_sample_transformed = preprocess.transform(X_sample)

    # SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample_transformed)

    # SHAP summary plot (Prediction explanability)
    shap.summary_plot(
        shap_values,
        X_sample_transformed,
        feature_names=feature_names,
        show=False
    )

    shap_plot_path = reports_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()

    print(f"Saved SHAP summary plot to: {shap_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/train.yaml")
    parser.add_argument(
        "--model_path",
        default="models/model_xgb_tuned.joblib",
        help="Path to trained model pipeline .joblib",
    )
    args = parser.parse_args()
    main(args.config, args.model_path)
