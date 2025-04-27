from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.pipelines.knn_pipeline import create_pipeline


def test_create_pipeline_default_params(cv, scorer):
    pipeline = create_pipeline(gscv=cv, scorer=scorer)

    assert isinstance(pipeline, GridSearchCV)
    steps = pipeline.estimator.named_steps
    assert "features" in steps
    assert "scaler" in steps
    assert "selector" in steps
    assert "knn" in steps
    assert isinstance(steps["knn"], KNeighborsClassifier)

    assert "selector__threshold" in pipeline.param_grid
    assert "knn__n_neighbors" in pipeline.param_grid
    assert "knn__p" in pipeline.param_grid


def test_create_pipeline_custom_params(cv, scorer):
    # Custom parameters
    custom_params = {
        "thresholds": [0.1, 0.2],
        "n_neighbors": [10, 20],
        "p_values": [3, 4],
    }

    # Call function with custom parameters
    pipeline = create_pipeline(gscv=cv, scorer=scorer, **custom_params)

    # Check that the parameters were applied correctly
    assert pipeline.param_grid["selector__threshold"] == [0.1, 0.2]
    assert pipeline.param_grid["knn__n_neighbors"] == [10, 20]
    assert pipeline.param_grid["knn__p"] == [3, 4]


def test_create_pipeline_missing_params(cv, scorer):
    # Call with only some parameters provided
    pipeline = create_pipeline(
        gscv=cv, scorer=scorer, thresholds=[0.5]  # Only override thresholds
    )

    # Check that default values were used for non-provided parameters
    assert pipeline.param_grid["selector__threshold"] == [0.5]  # Custom value
    assert pipeline.param_grid["knn__n_neighbors"] == [1, 3, 5, 7, 9]
    assert pipeline.param_grid["knn__p"] == [1, 2]  # Default value


def test_pipeline_components(cv, scorer):
    # Create pipeline
    grid_search = create_pipeline(gscv=cv, scorer=scorer)
    pipeline = grid_search.estimator

    # Test the selector component
    selector = pipeline.named_steps["selector"]
    assert isinstance(selector, VarianceThreshold)

    # Test the KNN component
    knn = pipeline.named_steps["knn"]
    assert isinstance(knn, KNeighborsClassifier)
