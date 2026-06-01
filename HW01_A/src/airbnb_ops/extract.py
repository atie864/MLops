from pathlib import Path
import pandas as pd

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists:
        raise FileNotFoundError(f"csv file not found: {path}")
    return pd.read_csv(path)    