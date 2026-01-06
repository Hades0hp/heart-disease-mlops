import pandas as pd
import tempfile
from src.data import load_dataset


def test_load_dataset_success():
    df = pd.DataFrame({"age": [50, 60], "sex": [1, 0], "target": [1, 0]})

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        loaded = load_dataset(f.name)

    assert not loaded.empty
    assert "target" in loaded.columns
