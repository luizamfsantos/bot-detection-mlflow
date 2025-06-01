from sklearn.model_selection import GridSearchCV

from src.pipelines.svm_linear_pipeline import create_pipeline


def test_create_pipeline_default_params(cv, scorer):
    pipeline = create_pipeline(gscv=cv, scorer=scorer)

    assert isinstance(pipeline, GridSearchCV)
    steps = pipeline.estimator.named_steps
    assert "imputer" in steps
    assert "tree" in steps
    assert isinstance(steps["tree"], DecisionTreeClassifier)

    assert "imputer__strategy" in pipeline.param_grid
    assert "tree__max_depth" in pipeline.param_grid
    assert "tree__min_samples_split" in pipeline.param_grid
    assert "tree__criterion" in pipeline.param_grid


def test_create_pipeline_custom_params(cv, scorer):
    # Custom parameters
    custom_params = {
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"],
    }

    # Call function with custom parameters
    pipeline = create_pipeline(gscv=cv, scorer=scorer, **custom_params)

    # Check that the parameters were applied correctly
    assert pipeline.param_grid["tree__max_depth"] == [5, 10]
    assert pipeline.param_grid["tree__min_samples_split"] == [2, 5]
    assert pipeline.param_grid["tree__criterion"] == ["gini", "entropy"]


def test_create_pipeline_missing_params(cv, scorer):
    # Call with only some parameters provided
    pipeline = create_pipeline(
        gscv=cv, scorer=scorer, max_depth=[5]  # Only override max_depth
    )

    # Check that default values were used for non-provided parameters
    assert pipeline.param_grid["tree__max_depth"] == [5]  # Custom value
    assert pipeline.param_grid["tree__min_samples_split"] == [2, 5, 10]
    assert pipeline.param_grid["tree__criterion"] == ["gini", "entropy", "log_loss"]
