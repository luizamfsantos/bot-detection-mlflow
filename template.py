import logging
import os
from pathlib import Path, PosixPath

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s:%(levelname)s]: %(message)s"
)

project_name = "MLProject"

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/utils.py",
    "src/config.py",
    "src/pipeline.py",
    "src/data_processing.py",
    "src/model.py",
    "src/evaluation.py",
    "src/metrics.py",
    "src/visualization.py",
    "notebooks/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "logs/.gitkeep",
    "tests/__init__.py",
    "tests/test_utils.py",
    "tests/test_config.py",
    "tests/test_pipeline.py",
    "tests/test_data_processing.py",
    "tests/test_model.py",
    "tests/test_evaluation.py",
    "visualizations/.gitkeep",
    "results/.gitkeep",
    "reports/.gitkeep",
    "MILESTONES.md",
    "mlflow.yaml",
    "config.yaml",
    "requirements.txt",
    "main.py",
    "README.md",
    "TEAM.md",
    ".gitignore",
]

for file in list_of_files:
    filepath: PosixPath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            logging.info(f"Creating file: {filepath}")
            f.write("")
    else:
        logging.info(f"File: {filepath} already exists")
