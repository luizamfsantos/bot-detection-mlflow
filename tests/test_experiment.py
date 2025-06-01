import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import GridSearchCV

from src.experiment import load_pipeline_config, make_serializable, run_experiment


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


@pytest.mark.skip(reason="require running mlflow")
def test_run_experiment(monkeypatch, sample_data, mock_config):
    def mock_load_data() -> pd.DataFrame:
        return sample_data

    monkeypatch.setattr("src.experiment.load_data", mock_load_data)

    run_experiment(mock_config)


def test_make_serializable():
    sample_cv_results = {
        "mean_fit_time": np.array(
            [
                0.01010291,
                0.00952371,
                0.00998068,
                0.00941507,
                0.01773167,
                0.01646463,
                0.01688274,
                0.01637197,
            ]
        ),
        "std_fit_time": np.array(
            [
                1.70750579e-04,
                1.55893099e-04,
                6.07261060e-05,
                4.97472171e-05,
                9.30568290e-04,
                9.37292004e-05,
                7.55535733e-06,
                7.14754883e-05,
            ]
        ),
        "mean_score_time": np.array(
            [
                0.00281334,
                0.0028007,
                0.00289456,
                0.00303721,
                0.00286404,
                0.00279482,
                0.00296354,
                0.00294367,
            ]
        ),
        "std_score_time": np.array(
            [
                1.93848625e-05,
                3.12143252e-05,
                6.81708537e-06,
                1.71670649e-04,
                9.22433098e-06,
                1.67757399e-05,
                4.17502634e-05,
                1.29308961e-05,
            ]
        ),
        "param_imputer__strategy": np.ma.masked_array(
            data=[
                "mean",
                "mean",
                "mean",
                "mean",
                "most_frequent",
                "most_frequent",
                "most_frequent",
                "most_frequent",
            ],
            mask=[False, False, False, False, False, False, False, False],
            fill_value=np.str_("?"),
            dtype=object,
        ),
        "param_selector__k": np.ma.masked_array(
            data=[10, 10, 15, 15, 10, 10, 15, 15],
            mask=[False, False, False, False, False, False, False, False],
            fill_value=999999,
        ),
        "param_selector__score_func": np.ma.masked_array(
            data=[f_classif, chi2, f_classif, chi2, f_classif, chi2, f_classif, chi2],
            mask=[False, False, False, False, False, False, False, False],
            fill_value=np.str_("?"),
            dtype=object,
        ),
        "params": [
            {
                "imputer__strategy": "mean",
                "selector__k": 10,
                "selector__score_func": f_classif,
            },
            {
                "imputer__strategy": "mean",
                "selector__k": 10,
                "selector__score_func": chi2,
            },
            {
                "imputer__strategy": "mean",
                "selector__k": 15,
                "selector__score_func": f_classif,
            },
            {
                "imputer__strategy": "mean",
                "selector__k": 15,
                "selector__score_func": chi2,
            },
            {
                "imputer__strategy": "most_frequent",
                "selector__k": 10,
                "selector__score_func": f_classif,
            },
            {
                "imputer__strategy": "most_frequent",
                "selector__k": 10,
                "selector__score_func": chi2,
            },
            {
                "imputer__strategy": "most_frequent",
                "selector__k": 15,
                "selector__score_func": f_classif,
            },
            {
                "imputer__strategy": "most_frequent",
                "selector__k": 15,
                "selector__score_func": chi2,
            },
        ],
        "split0_test_score": np.array(
            [
                0.89223698,
                0.90755873,
                0.89223698,
                0.8988764,
                0.89223698,
                0.95199183,
                0.89223698,
                0.92032686,
            ]
        ),
        "split1_test_score": np.array(
            [
                0.90194076,
                0.91164454,
                0.90194076,
                0.90500511,
                0.90194076,
                0.95607763,
                0.90194076,
                0.92645557,
            ]
        ),
        "split2_test_score": np.array(
            [
                0.89989785,
                0.90960163,
                0.89989785,
                0.90245148,
                0.89989785,
                0.96118488,
                0.89989785,
                0.92032686,
            ]
        ),
        "mean_test_score": np.array(
            [
                0.8980252,
                0.90960163,
                0.8980252,
                0.902111,
                0.8980252,
                0.95641811,
                0.8980252,
                0.92236977,
            ]
        ),
        "std_test_score": np.array(
            [
                0.004177,
                0.00166802,
                0.004177,
                0.00251359,
                0.004177,
                0.00376076,
                0.004177,
                0.0028891,
            ]
        ),
        "rank_test_score": np.array([5, 3, 5, 4, 5, 1, 5, 2], dtype=np.int32),
    }
    serializable_results = {
        k: make_serializable(v) for k, v in sample_cv_results.items()
    }
    result = json.dumps(serializable_results, default=str)
    assert isinstance(result, dict) or isinstance(result, str)
