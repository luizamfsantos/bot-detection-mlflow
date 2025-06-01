from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipelines.base import create_grid_search


def test_create_grid_search(cv, scorer):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    param_grid = {"clf__C": [0.1, 1, 10]}
    grid_search = create_grid_search(pipeline, param_grid, cv, scorer)

    assert isinstance(grid_search, GridSearchCV)
    assert grid_search.param_grid == param_grid
    assert grid_search.scoring == scorer
    assert grid_search.cv == cv
    assert grid_search.estimator == pipeline


def test_create_grid_search_with_empty_param_grid(cv, scorer):
    # Create test data with empty param_grid
    pipeline = Pipeline([("clf", LogisticRegression())])
    param_grid = {}

    # Call the function
    grid_search = create_grid_search(pipeline, param_grid, cv, scorer)

    # Even with empty param_grid, should return a valid GridSearchCV object
    assert isinstance(grid_search, GridSearchCV)
    assert grid_search.param_grid == param_grid
