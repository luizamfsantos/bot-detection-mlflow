from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from src.pipelines.gaussian_nb_pipeline import create_pipeline


def test_create_pipeline_default_params(cv, scorer):
    pipeline = create_pipeline(gscv=cv, scorer=scorer)

    assert isinstance(pipeline, GridSearchCV)
    steps = pipeline.estimator.named_steps
    assert "imputer" in steps
    assert "nb" in steps
    assert "selector" in steps
    assert isinstance(steps["nb"], GaussianNB)

    assert "imputer__strategy" in pipeline.param_grid
    assert "selector__k" in pipeline.param_grid
    assert "selector__score_func" in pipeline.param_grid


def test_create_pipeline_custom_params(cv, scorer):
    # Custom parameters
    custom_params = {
        "num_top_features": [15, 10],
        "selector_score_func": ["f_classif", "mutual_info_classif"],
    }

    # Call function with custom parameters
    pipeline = create_pipeline(gscv=cv, scorer=scorer, **custom_params)

    # Check that the parameters were applied correctly
    assert pipeline.param_grid["selector__k"] == [15, 10]
    assert pipeline.param_grid["selector__score_func"] == [
        f_classif,
        mutual_info_classif,
    ]


def test_create_pipeline_missing_params(cv, scorer):
    # Call with only some parameters provided
    pipeline = create_pipeline(
        gscv=cv, scorer=scorer, num_top_features=[10]  # Only override max_depth
    )

    # Check that default values were used for non-provided parameters
    assert pipeline.param_grid["selector__k"] == [10]  # Custom value
    assert pipeline.param_grid["selector__score_func"] == [f_classif]
