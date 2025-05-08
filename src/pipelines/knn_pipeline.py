import logging

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.pipelines.base import create_grid_search


def create_pipeline(gscv, scorer, **kwargs):
    """Create a KNN pipeline with grid search"""
    thresholds = kwargs.get("thresholds", [0, 0.01, 0.02, 0.03])
    n_neighbors = kwargs.get("n_neighbors", [1, 3, 5, 7, 9])
    p_values = kwargs.get("p_values", [1, 2])
    impute_strategy = kwargs.get("impute_strategy", "mean")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy, add_indicator=True)),
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("selector", VarianceThreshold()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "selector__threshold": thresholds,
        "knn__n_neighbors": n_neighbors,
        "knn__p": p_values,
    }
    logging.info(
        f"""
    KNN pipeline with imputer with {impute_strategy},
    adding missing indicators, min max scaling to 0-1 scale,
    and removing features with no variance.
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
