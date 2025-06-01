import json
from pathlib import Path

import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare

results_path = Path("results")
assert results_path.exists()
file_list = results_path.glob("*cross_val_scores.json")
results = {}
for file in file_list:
    with open(file, "r") as f:
        results_tmp = json.load(f)
    filename = file.stem.split("_cross_val_scores")[0].split("_experiment_")[-1]
    results[filename] = list(results_tmp.values())[0]

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
