import json
import os
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_processing import load_data

SEED = os.environ.get("SEED", 27)


def load_pipeline_config(pipeline_name: str) -> Callable:
    """Load default parameters for a pipeline
    and return the pipeline constructor"""
    try:
        module = import_module(f"src.pipelines.{pipeline_name}_pipeline")
        create_pipeline = getattr(module, "create_pipeline")

        return create_pipeline
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Pipeline {pipeline_name} not found: {e}")


def get_experiment_id(experiment_name: str) -> int:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif callable(obj):
        return obj.__name__
    elif isinstance(obj, Mapping):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, str):  # Leave strings as-is
        return obj
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        return [make_serializable(v) for v in obj]
    else:
        obj = str(obj)
        try:
            json.dumps(obj)  # test if serializable
            return obj
        except TypeError:
            return str(obj)


def run_experiment(
    config,
    tracking_uri: str = "http://127.0.0.1:8080",
    output_basename: str = "results/",
    seed: int = SEED,
) -> None:
    # extract config values
    experiment_name = config["experiment_name"]
    pipeline_name = config["pipeline"]
    cv_folds = config["cv_folds"]
    grid_cv_folds = config["grid_cv_folds"]
    seed = config["seed"]
    pipeline_params = config["pipeline_params"]
    pipeline_params["seed"] = seed

    # set tracking server uri for logging
    mlflow.set_tracking_uri(uri=tracking_uri)

    experiment_id = get_experiment_id(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    # define scorers and cross-validation strategies
    # TODO: add option for precision_score, recall_score, f1_score
    scorer = make_scorer(accuracy_score)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gscv = StratifiedKFold(n_splits=grid_cv_folds, shuffle=True, random_state=seed)

    # load data
    df = load_data()
    X = df.drop(columns=["target"])
    ycla = df["target"] > 0

    # get pipeline creator function
    create_pipeline = load_pipeline_config(pipeline_name)

    # create the pipeline
    clf = create_pipeline(gscv=gscv, scorer=scorer, **pipeline_params)

    # run mlflow experiment
    with mlflow.start_run(run_name=pipeline_name):
        # log metadata
        mlflow.set_tag("algorithm", pipeline_name)
        mlflow.sklearn.log_model(clf, "pipeline")
        steps_info = {
            name: str(estimator)
            for name, estimator in clf.estimator.named_steps.items()
        }
        mlflow.log_param("steps", str(steps_info))
        mlflow.log_param("pipeline_params", str(pipeline_params))

        # perform cross-validation
        cv_scores = cross_val_score(clf, X, ycla, cv=cv)
        output_file = (
            output_basename + f"{experiment_name}_{pipeline_name}_cross_val_scores.json"
        )
        with open(output_file, "w") as f:
            json.dump({pipeline_name: cv_scores.tolist()}, f)
        mlflow.log_artifact(output_file)

        # get best hyperparameters
        clf.fit(X, ycla)
        output_file = (
            output_basename + f"{experiment_name}_{pipeline_name}_hyperparameters.json"
        )
        serializable_results = {
            k: make_serializable(v) for k, v in clf.cv_results_.items()
        }
        with open(output_file, "w") as f:
            json.dump({pipeline_name: serializable_results}, f, default=str)

        mlflow.log_artifact(output_file)
