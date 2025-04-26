import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import load_data

# Constants
SEED = 12
EXPERIMENT_NAME = "multiple_algorithms_comparison"

# Set tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment or use existing one
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(experiment_id=experiment_id)

# Define scorers and cross-validation strategies
scorer = make_scorer(accuracy_score)
cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=SEED)
gscv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

# Define algorithms dictionary with GridSearchCV objects
algorithms = {
    "kNN": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler(feature_range=(0, 1))),
                ("selector", VarianceThreshold()),
                ("knn", KNeighborsClassifier()),
            ]
        ),
        param_grid={
            "selector__threshold": [0, 0.01, 0.02, 0.03],
            "knn__n_neighbors": [1, 3, 5],
            "knn__p": [1, 2],
        },
        scoring=scorer,
        cv=gscv,
    ),
    "tree": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("tree", DecisionTreeClassifier(random_state=SEED)),
            ]
        ),
        param_grid={
            "tree__max_depth": [5, 10, 20],
            "tree__criterion": ["entropy", "gini"],
        },
        scoring=scorer,
        cv=gscv,
    ),
    "bigtree": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                (
                    "tree",
                    DecisionTreeClassifier(
                        max_depth=None, min_samples_split=2, random_state=SEED
                    ),
                ),
            ]
        ),
        param_grid={
            "tree__criterion": ["entropy", "gini"],
        },
        scoring=scorer,
        cv=gscv,
    ),
    "nb": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("selector", SelectKBest()),
                ("nb", GaussianNB()),
            ]
        ),
        param_grid={
            "selector__k": [3, 5, 10],
        },
        scoring=scorer,
        cv=gscv,
    ),
    "svmlinear": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("svm", SVC(kernel="linear", random_state=SEED)),
            ]
        ),
        param_grid={
            "pca__n_components": [2, 5, 10],
            "svm__C": [1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
    ),
    "svmrbf": GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=SEED)),
                ("svm", SVC(kernel="rbf", random_state=SEED)),
            ]
        ),
        param_grid={
            "pca__n_components": [2, 5, 10],
            "svm__C": [1.0, 2.0],
            "svm__gamma": [0.1, 1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
    ),
}

df = load_data()

# TODO: preprocessing

# df, _ = train_test_split(df_interm, train_size=1000,
# stratify=df_interm["target"], random_state=SEED)


X = df.drop(columns=["target"])

yreg = df.target
ycla = yreg > 0


result = {}
for alg, clf in algorithms.items():
    with mlflow.start_run(run_name=alg):
        # log algorithm name
        mlflow.set_tag("algorithm", alg)
        mlflow.set_tag("description", f"Grid search for {alg} classifier")

        # perform cross-validation
        cv_scores = cross_val_score(clf, X, ycla, cv=cv)

        # log cross-validation scores
        mlflow.log_metric("mean_cv_accuracy", cv_scores.mean())
        mlflow.log_metric("std_cv_accuracy", cv_scores.std())

        # fit the model on the entire dataset to get best params
        clf.fit(X, ycla)

        # log best parameters
        for param_name, param_value in clf.best_params_.items():
            mlflow.log_param(param_name, param_value)

        # log best score from grid search
        best_model = clf.best_estimator_

        # make predictions for signature inference
        y_pred = best_model.predict(X)

        # infer the model signature
        signature = infer_signature(X, y_pred)

        # log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=f"{alg}_model",
            signature=signature,
            input_example=X.iloc[:5],
            registered_model_name=f"{alg}-classifier",
        )

        # store cv scores for result dataframe
        result[alg] = cv_scores

result = pd.DataFrame.from_dict(result)
