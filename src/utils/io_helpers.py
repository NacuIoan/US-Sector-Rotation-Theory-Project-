"""
io_helpers.py — Save / load processed datasets.

Wrapper around Parquet I/O for the data/processed/ directory.
"""

import pandas as pd

from src.utils.config import PROCESSED_DIR


def save_processed(df: pd.DataFrame, name: str) -> str:
    """
    Save a DataFrame to data/processed/{name}.parquet.

    Returns the path as a string.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow")
    print(f"[IO] Saved → {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return str(path)


def load_processed(name: str) -> pd.DataFrame:
    """
    Load a DataFrame from data/processed/{name}.parquet.

    Raises FileNotFoundError if the file doesn't exist.
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            f"Run the pipeline first to generate it."
        )
    df = pd.read_parquet(path)
    print(f"[IO] Loaded ← {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df
