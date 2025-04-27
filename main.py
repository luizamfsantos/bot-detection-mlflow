import argparse

import yaml

from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:8080",
        help="MLflow tracking server URI",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run the experiment
    run_experiment(config, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
