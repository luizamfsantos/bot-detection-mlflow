from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

# Import the functions
from src.data_processing import load_data, raw_data_path, save_dataset


@pytest.fixture
def mock_subprocess_run():
    with mock.patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_path_exists_true():
    with mock.patch("src.data_processing.Path.exists", return_value=True):
        yield


@pytest.fixture
def mock_path_exists_false():
    with mock.patch("src.data_processing.Path.exists", return_value=False):
        yield


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "has_domain": [True, False],
            "has_short_name": [False, True],
            "has_first_name": [True, True],
            "city": ["CityA", "CityB"],
            "other_feature": [10, 20],
        }
    )


@pytest.fixture
def mock_read_csv(dummy_df):
    with mock.patch(
        "src.data_processing.pd.read_csv", return_value=dummy_df.copy()
    ) as mock_read:
        yield mock_read


def test_save_dataset(mock_subprocess_run):
    save_dataset()
    mock_subprocess_run.assert_called_once_with(["./src/fetch_data.sh"])


def test_load_data_file_exists(mock_path_exists_true, mock_read_csv):
    df = load_data()
    # Check if columns were dropped correctly
    assert "has_domain" not in df.columns
    assert "has_short_name" not in df.columns
    assert "has_first_name" not in df.columns
    assert "city" not in df.columns
    assert "other_feature" in df.columns


def test_load_data_file_not_exists(
    mock_path_exists_false, mock_read_csv, mock_subprocess_run
):
    df = load_data()
    # Should call save_dataset internally
    mock_subprocess_run.assert_called_once_with(["./src/fetch_data.sh"])
    assert "has_domain" not in df.columns
    assert "city" not in df.columns


def test_save_dataset_run():
    save_dataset()
    assert Path(raw_data_path).exists()


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "city" not in df
