import pandas as pd
from churn.features import infer_feature_spec, build_preprocess_pipeline

def test_preprocess_pipeline_builds_and_transforms():
    df = pd.DataFrame({
        "tenure": [1, 2, 3],
        "MonthlyCharges": [50.0, 70.0, 80.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["No", "Yes", "No"],
    })

    spec = infer_feature_spec(df, target_col="Churn", id_cols=[])
    pre = build_preprocess_pipeline(spec)

    X = df.drop(columns=["Churn"])
    Xt = pre.fit_transform(X)

    assert Xt.shape[0] == 3
