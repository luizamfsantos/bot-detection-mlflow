import json
from pathlib import Path

import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare

results_path = Path("results")
assert results_path.exists()
file_list = results_path.glob("*cross_val_scores.json")
results = {}
for file in file_list:
    with open(file, "r") as f:
        results_tmp = json.load(f)
    results[file.stem] = list(results_tmp.values())[0]

results_array = np.array(list(results.values()))
stats, p_value = friedmanchisquare(*results_array)
print(f"Test statistics: {stats}. P-value: {p_value}")
nemenyi = posthoc_nemenyi_friedman(results_array.T)
