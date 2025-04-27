from importlib import import_module


def load_pipeline_config(pipeline_name: str):
    """Load default parameters for a pipeline
    and return the pipeline constructor"""
    try:
        module = import_module(f"src.pipelines.{pipeline_name}_pipeline")
        create_pipeline = getattr(module, "create_pipeline")

        return create_pipeline
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Pipeline {pipeline_name} not found: {e}")
