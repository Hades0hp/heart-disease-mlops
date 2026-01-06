from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    raw_csv: Path


DEFAULT_TARGET_COL = "target"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Heart Disease CSV.

    Expected: tabular CSV with a binary target column named 'target' (0/1).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if DEFAULT_TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{DEFAULT_TARGET_COL}' in CSV. Found: {list(df.columns)}")
    return df
