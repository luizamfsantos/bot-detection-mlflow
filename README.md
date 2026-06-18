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

## Serving a model

After training, serve any logged model via the `/predict` endpoint:

1. Find a run ID from the MLflow UI or CLI:
    ```bash
    mlflow runs list --experiment-name knn_experiment
    ```
2. Start the server:
    ```bash
    MODEL_URI="runs:/YOUR_RUN_ID/pipeline" python serve.py
    ```
3. Health check:
    ```bash
    curl http://localhost:8000/health
    ```
4. Predict (send any subset of features — missing ones are imputed automatically):
    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": {"has_photo": 1, "subscribers_count": 50, "posts_count": 10, "avg_likes": 2.5}}'
    # → {"prediction": 0, "label": "user"}
    ```
