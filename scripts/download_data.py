from __future__ import annotations

from pathlib import Path
import pandas as pd

# UCI Heart Disease dataset via ucimlrepo
# Ref: https://archive.ics.uci.edu/dataset/45/heart+disease
from ucimlrepo import fetch_ucirepo


def main():
    out_path = Path("data/raw/heart_disease_uci.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Target can be multi-class (0..4). Convert to binary: 0 => 0, else 1
    # Column name may vary ("num" is common). We'll take the first column.
    y_col = y.columns[0]
    y_bin = (y[y_col].astype(float) > 0).astype(int).rename("target")

    df = pd.concat([X, y_bin], axis=1)

    # Keep only numeric/categorical-friendly columns; drop rows with missing target
    df = df.dropna(subset=["target"])

    df.to_csv(out_path, index=False)
    print(f"Wrote dataset to: {out_path.resolve()}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
