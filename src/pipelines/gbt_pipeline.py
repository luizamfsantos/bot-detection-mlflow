import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.pipelines.base import create_grid_search


class TFDFGBTClassifier(BaseEstimator, ClassifierMixin):
    """Gradient Boosted Trees via TensorFlow Decision Forests, wrapped as an sklearn estimator."""

    def __init__(self, num_trees=300, max_depth=6):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        import tensorflow_decision_forests as tfdf

        y_vals = y.values if hasattr(y, "values") else np.array(y)
        df = pd.DataFrame(X.astype(np.float32))
        df.columns = [str(c) for c in df.columns]
        df["label"] = y_vals.astype(int)

        ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label="label")
        self.model_ = tfdf.keras.GradientBoostedTreesModel(
            num_trees=self.num_trees,
            max_depth=self.max_depth,
        )
        self.model_.fit(ds)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        import tensorflow_decision_forests as tfdf

        df = pd.DataFrame(X.astype(np.float32))
        df.columns = [str(c) for c in df.columns]
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(df)
        proba = self.model_.predict(ds, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])


def create_pipeline(gscv, scorer, **kwargs):
    """Create a TF Decision Forests GBT pipeline with grid search"""
    num_trees = kwargs.get("num_trees", [100, 300])
    max_depth = kwargs.get("max_depth", [4, 6, 8])
    impute_strategy = kwargs.get("impute_strategy", ["mean"])

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(add_indicator=True)),
            ("gbt", TFDFGBTClassifier()),
        ]
    )

    param_grid = {
        "imputer__strategy": impute_strategy,
        "gbt__num_trees": num_trees,
        "gbt__max_depth": max_depth,
    }

    logging.info(
        """
    GBT pipeline with SimpleImputer with missing indicators
    and TensorFlow Decision Forests GradientBoostedTrees.
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
