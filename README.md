# Bot Detection using MlFlow and Scikit-learn

This project presents a modular approach to machine learning training and evaluation. It includes experiment tracking using Mlflow and implement robust statistical metrics to evaluate differences in algorithm performances.

Quick start:
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the Mlflow server:
    ```bash
    mlflow server --host 127.0.0.1 --port 8080
    ```
3. Run a training script on one of the configurations. For example:
    ```bash
    python main.py --config configs/svm_linear_experiment.yaml
    ```
