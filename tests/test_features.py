import pandas as pd
from src.features import infer_feature_spec

def test_infer_feature_spec_has_target():
    df = pd.DataFrame({"age":[10,20], "sex":[0,1], "target":[0,1]})
    spec = infer_feature_spec(df, target_col="target")
    assert spec.target_col == "target"
    assert "age" in spec.numeric_cols
