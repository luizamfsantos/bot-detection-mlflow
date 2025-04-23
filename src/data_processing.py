import sys
from pathlib import Path

import pandas as pd

sys.path.append("..")

raw_data_path: str = "data/raw/bots_vs_users.csv"


def save_dataset() -> None:
    import subprocess

    subprocess.run(["./src/fetch_data.sh"])


def load_data(
    raw_data_path: str = raw_data_path,
) -> pd.DataFrame:
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        save_dataset()
    return pd.read_csv(raw_data_path)
