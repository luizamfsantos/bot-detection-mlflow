import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data_processing import load_data


def tsne_to_tikz(
    df: pd.DataFrame,
    target_col: str,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
    max_points: int = 1000,
):
    # Sample dataset to simplify tikz plot
    df = df.sample(n=min(len(df), max_points), random_state=random_state)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply TSNE
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    tsne_results = tsne.fit_transform(X_scaled)

    # Prepare the TikZ output
    tikz_output = []
    for i in range(len(tsne_results)):
        tikz_output.append(
            f"\\node at ({tsne_results[i][0]:.2f}, {tsne_results[i][1]:.2f}) [circle, draw] {{ {y.iloc[i]} }};"
        )

    return "\n".join(tikz_output)


def fill_na(df: pd.DataFrame, mode: str = "mean"):
    if mode == "mean":
        return df.fillna(df.mean())
    if mode == "median":
        return df.fillna(df.median())
    else:
        raise NotImplementedError


def pca_to_tikz(
    df: pd.DataFrame,
    target_col: str,
    n_components: int = 2,
    max_points: int = 1000,
    random_state: int = 42,
):
    # Sample dataset to simplify tikz plot
    df = df.sample(n=min(len(df), max_points), random_state=random_state)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(X_scaled)

    # Prepare the TikZ output
    tikz_output = []
    for i in range(len(pca_results)):
        tikz_output.append(
            f"\\node at ({pca_results[i][0]:.2f}, {pca_results[i][1]:.2f}) [circle, draw] {{ {y.iloc[i]} }};"
        )

    return "\n".join(tikz_output)


def main(output_file: str, overwrite: bool = False, impute_mode: str = "mean"):
    df = load_data()
    df = fill_na(df, impute_mode)
    pca = pca_to_tikz(df, "target")
    tsne = tsne_to_tikz(df, "target")
    mode = "w" if overwrite else "a"
    with open(output_file, mode) as f:
        f.write("\\begin{tikzpicture}\n")
        f.write(pca)
        f.write("\n\\end{tikzpicture}\n")
        f.write("\n\\begin{tikzpicture}\n")
        f.write(tsne)
        f.write("\n\\end{tikzpicture}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", default="results/visualizations.tex")
    parser.add_argument("--impute_mode", default="mean")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.output_file, args.overwrite, args.impute_mode)
