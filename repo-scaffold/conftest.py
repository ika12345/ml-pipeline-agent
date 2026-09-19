import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_sample_df() -> pd.DataFrame:
    """~20-row fixture standing in for the real dataset during smoke tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(20),
            "text": [f"sample item {i}" for i in range(20)],
            "target": rng.uniform(1, 100, size=20),
        }
    )
