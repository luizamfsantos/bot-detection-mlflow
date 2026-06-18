from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.pipelines.svm_linear_pipeline import create_pipeline


def test_create_pipeline_default_params(cv, scorer):
    pipeline = create_pipeline(gscv=cv, scorer=scorer)

    assert isinstance(pipeline, GridSearchCV)
    steps = pipeline.estimator.named_steps
    assert "imputer" in steps
    assert "scaler" in steps
    assert "pca" in steps
    assert "svm" in steps
    assert isinstance(steps["svm"], SVC)
    assert steps["svm"].kernel == "linear"

    assert "imputer__strategy" in pipeline.param_grid
    assert "pca__n_components" in pipeline.param_grid
    assert "svm__C" in pipeline.param_grid


def test_create_pipeline_custom_params(cv, scorer):
    custom_params = {
        "n_pca_components": [3, 7],
        "svm_regularization": [0.01, 1.0],
    }

    pipeline = create_pipeline(gscv=cv, scorer=scorer, **custom_params)

    assert pipeline.param_grid["pca__n_components"] == [3, 7]
    assert pipeline.param_grid["svm__C"] == [0.01, 1.0]


def test_create_pipeline_missing_params(cv, scorer):
    pipeline = create_pipeline(gscv=cv, scorer=scorer, n_pca_components=[5])

    assert pipeline.param_grid["pca__n_components"] == [5]
    assert pipeline.param_grid["svm__C"] == [0.1, 1, 10, 100]
    assert pipeline.param_grid["imputer__strategy"] == ["mean"]
