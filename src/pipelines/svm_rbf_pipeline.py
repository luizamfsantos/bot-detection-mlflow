import logging

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pipelines.base import create_grid_search


def create_pipeline(gscv, scorer, **kwargs):
    """Create a KNN pipeline with grid search"""
    impute_strategy = kwargs.get("impute_strategy", "mean")
    n_pca_components = kwargs.get("n_pca_components", [2, 5, 10])
    svm_regularization = kwargs.get("svm_regularization", [0.1, 1, 10, 100])
    svm_gamma = kwargs.get("svm_gamma", [0.1, 1.0, 2.0])
    seed = kwargs.get("seed", 27)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(add_indicator=True)),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("svm", SVC(kernel="rbf", random_state=seed)),
        ]
    )

    param_grid = {
        "imputer__strategy": impute_strategy,
        "pca__n_components": n_pca_components,
        "svm__C": svm_regularization,
        "svm__gamma": svm_gamma,
    }
    logging.info(
        f"""
    SVM pipeline with imputer with {impute_strategy},
    adding missing indicators, standard scaling,
    utilizing PCA components
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
