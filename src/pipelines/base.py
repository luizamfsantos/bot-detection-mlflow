from sklearn.model_selection import GridSearchCV


def create_grid_search(pipeline, param_grid, gscv, scorer):
    """
    Create a GridSearchCV object with the given pipeline and parameters

    Args:
        pipeline: The sklearn Pipeline object
        param_grid: Dictionary with parameters names as keys and lists of
            parameter settings
        gscv: Cross-validation generator
        scorer: Function used for model evaluation

    Returns:
        GridSearchCV: Configured grid search object
    """
    return GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer, cv=gscv)
