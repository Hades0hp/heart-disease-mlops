from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: str = "target"


def infer_feature_spec(df: pd.DataFrame, target_col: str = "target"):
    """Infer numeric and categorical columns from dataframe dtypes."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return FeatureSpec(numeric_cols=numeric_cols, 
                       categorical_cols=categorical_cols, 
                       target_col=target_col)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Preprocessor matching notebook intent: 
    impute + scale numeric; impute + onehot categorical."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, spec.numeric_cols),
        ("cat", categorical_pipe, spec.categorical_cols),
    ])
    return preprocessor
