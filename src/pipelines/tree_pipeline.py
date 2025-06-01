import logging

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipelines.base import create_grid_search


def create_pipeline(gscv, scorer, **kwargs):
    """Create a tree pipeline with grid search"""
    max_depth = kwargs.get("max_depth", [5, 10, 20, None])
    min_samples_split = kwargs.get("min_samples_split", [2, 5, 10])
    criterion = kwargs.get("criterion", ["gini", "entropy", "log_loss"])
    impute_strategy = kwargs.get("impute_strategy", ["mean"])
    seed = kwargs.get("seed", 27)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(add_indicator=True)),
            ("tree", DecisionTreeClassifier(random_state=seed)),
        ]
    )

    param_grid = {
        "imputer__strategy": impute_strategy,
        "tree__max_depth": max_depth,
        "tree__min_samples_split": min_samples_split,
        "tree__criterion": criterion,
    }
    logging.info(
        f"""
    Tree pipeline with imputer with {impute_strategy=},
    adding missing indicators.
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
