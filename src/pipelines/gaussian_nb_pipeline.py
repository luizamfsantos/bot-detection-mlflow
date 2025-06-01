import logging
from collections.abc import Callable

from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.pipelines.base import create_grid_search

map_feature_func = {
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
    "chi2": chi2,
}


def get_selector_function(func: str | Callable) -> Callable | None:
    if isinstance(func, str):
        func = map_feature_func.get(func)
    elif callable(func):
        func = func
    else:
        return None
    return func


def create_pipeline(gscv, scorer, **kwargs):
    """Create a Gaussian Naive Bayes pipeline with grid search"""
    impute_strategy = kwargs.get("impute_strategy", ["mean"])
    num_top_features = kwargs.get("num_top_features", [3, 5, 10])
    default_score_func = [f_classif]
    selector_score_func = kwargs.get("selector_score_func", default_score_func)
    selector_score_func = [get_selector_function(func) for func in selector_score_func]
    selector_score_func = [func for func in selector_score_func if func]
    selector_score_func = selector_score_func or default_score_func
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(add_indicator=True)),
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest()),
            ("nb", GaussianNB()),
        ]
    )

    param_grid = {
        "imputer__strategy": impute_strategy,
        "selector__k": num_top_features,
        "selector__score_func": selector_score_func,
    }
    logging.info(
        f"""
    Gaussian Naive Bayes pipeline with imputer with {impute_strategy=},
    adding missing indicators and feature selecting top {num_top_features}
    based on {selector_score_func}.
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
