import pandas as pd
from src.features import infer_feature_spec, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def test_pipeline_fit_predict():
    df = pd.DataFrame({
        "age": [45, 55, 65, 35],
        "sex": [1, 0, 1, 0],
        "target": [1, 1, 0, 0]
    })

    spec = infer_feature_spec(df)
    X = df.drop(columns=["target"])
    y = df["target"]

    preprocessor = build_preprocessor(spec)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)
    