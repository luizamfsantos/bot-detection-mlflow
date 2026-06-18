import logging

import keras
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipelines.base import create_grid_search


class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
    ):
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model_ = keras.Sequential(
            [
                keras.layers.Dense(self.hidden_units, activation="relu"),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.Dense(max(1, self.hidden_units // 2), activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        y_vals = y.values if hasattr(y, "values") else np.array(y)
        self.model_.fit(
            X,
            y_vals.astype(float),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        proba = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])


def create_pipeline(gscv, scorer, **kwargs):
    """Create a Keras MLP pipeline with grid search"""
    hidden_units = kwargs.get("hidden_units", [32, 64, 128])
    dropout_rates = kwargs.get("dropout_rates", [0.2, 0.3])
    learning_rates = kwargs.get("learning_rates", [0.001, 0.0001])
    epochs = kwargs.get("epochs", 50)
    batch_size = kwargs.get("batch_size", 32)
    impute_strategy = kwargs.get("impute_strategy", ["mean"])

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(add_indicator=True)),
            ("scaler", StandardScaler()),
            ("mlp", KerasMLPClassifier(epochs=epochs, batch_size=batch_size)),
        ]
    )

    param_grid = {
        "imputer__strategy": impute_strategy,
        "mlp__hidden_units": hidden_units,
        "mlp__dropout_rate": dropout_rates,
        "mlp__learning_rate": learning_rates,
    }

    logging.info(
        """
    MLP pipeline with SimpleImputer with missing indicators,
    StandardScaler, and a two-hidden-layer Keras neural network.
    """
    )
    return create_grid_search(pipeline, param_grid, gscv, scorer)
