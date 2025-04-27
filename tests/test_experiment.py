from collections.abc import Callable

import pytest
from sklearn.model_selection import GridSearchCV

from src.experiment import load_pipeline_config


def test_load_pipeline_config_success_real_module(cv, scorer):
    create_pipeline = load_pipeline_config("knn")
    assert isinstance(create_pipeline, Callable)
    result = create_pipeline(cv, scorer)
    assert isinstance(result, GridSearchCV)


def test_load_pipeline_config_import_error():
    with pytest.raises(ValueError, match="Pipeline non_existent not found:"):
        load_pipeline_config("non_existent")


def test_load_pipeline_config_attribute_error(monkeypatch):
    class BadModule:
        pass  # no create_pipeline

    def mock_import_module(name):
        return BadModule

    monkeypatch.setattr("src.experiment.import_module", mock_import_module)

    with pytest.raises(ValueError, match="Pipeline bad_pipeline not found:"):
        load_pipeline_config("bad_pipeline")
