import pickle
from ucimlrepo import fetch_ucirepo
import sys
from pathlib import Path
import pandas as pd
sys.path.append('..')

raw_data_path: str = 'data/raw/raw_data.pkl'


def save_dataset(
    raw_data_path: str = raw_data_path,
) -> None:
    data = fetch_ucirepo(id=372)
    with open(raw_data_path, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(
    raw_data_path: str = raw_data_path,
) -> dict:
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        save_dataset()
    with open(raw_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_data(
    raw_data_path: str = raw_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_dataset(raw_data_path)
    X = data.data.features
    y = data.data.targets
    return X, y


def load_metadata(
    raw_data_path: str = raw_data_path,
) -> dict:
    data = load_dataset(raw_data_path)
    metadata = data.metadata
    metadata['variables'] = data.variables
    return metadata
