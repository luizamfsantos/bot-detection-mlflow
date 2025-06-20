import json
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikit_posthocs import critical_difference_diagram, posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare

plt.rcParams.update(
    {
        "font.family": "Arial",  # "serif", # "DejaVu Sans",  # or "Arial", "Helvetica"
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def get_results_dict(file_list: PosixPath | list) -> dict:
    results = {}
    for file in file_list:
        with open(file, "r") as f:
            results_tmp = json.load(f)
        filename = file.stem.split("_cross_val_scores")[0].split("_experiment_")[-1]
        results[filename] = list(results_tmp.values())[0]
    return results


def calculate_stats_test(file_list):
    results = get_results_dict(file_list)
    labels = list(results.keys())
    data = np.array(list(results.values()))
    stats, p_value = friedmanchisquare(*data)
    print(f"Test statistics: {stats}. P-value: {p_value}")
    nemenyi = posthoc_nemenyi_friedman(data.T)
    nemenyi.index = labels
    nemenyi.columns = labels
    nemenyi.to_csv("results/statistical_tests.csv")
    significant_pairs = [
        (row, col, nemenyi.loc[row, col])
        for i, row in enumerate(nemenyi.index)
        for j, col in enumerate(nemenyi.columns)
        if j > i and not pd.isna(nemenyi.loc[row, col]) and nemenyi.loc[row, col] < 0.05
    ]
    print(significant_pairs)


def generate_critical_difference_diagram(file_list):
    results = get_results_dict(file_list)
    labels = list(results.keys())
    data = np.array(list(results.values()))
    # TODO: calculate average rank for each algorithm
    avg_rank = {alg: np.mean(vals) for alg, vals in results.items()}
    # TODO: calculate nemenyi stats
    nemenyi = posthoc_nemenyi_friedman(data.T)
    nemenyi.index = labels
    nemenyi.columns = labels
    plt.figure(figsize=(10, 3), dpi=100)
    plt.title("Critical difference diagram ($\\alpha = 0.05$)")
    critical_difference_diagram(avg_rank, nemenyi)
    plt.tight_layout()


results_path = Path("results")
assert results_path.exists()
file_list_complete = list(results_path.glob("complete/*cross_val_scores*.json"))
file_list_subset = list(results_path.glob("subset/*cross_val_scores*.json"))
calculate_stats_test(file_list_complete)
calculate_stats_test(file_list_subset)
generate_critical_difference_diagram(file_list_complete)
plt.savefig("results/images/cd_diagram_complete.png", dpi=300, bbox_inches="tight")
plt.close()
generate_critical_difference_diagram(file_list_subset)
plt.savefig("results/images/cd_diagram_subset.png", dpi=300, bbox_inches="tight")
plt.close()
