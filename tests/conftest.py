import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold

SEED = 42


@pytest.fixture
def sample_data():
    data = {
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "feature3": ["a", "b", "a", "b", "c", "a", "b", "c", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_mlflow_environment(monkeypatch):
    """Setup a mock MLflow environment"""
    # Mock MLflow tracking URI
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///tests/tmp/mlflow.db")


@pytest.fixture
def cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


@pytest.fixture
def scorer():
    return make_scorer(accuracy_score)
