import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data_processing import load_data


def fill_na(df: pd.DataFrame, mode: str = "mean"):
    imputer = SimpleImputer(strategy=mode)
    return imputer.fit_transform(df)


def tsne_to_tikz(
    df: pd.DataFrame,
    target_col: str,
    colors: list[str],
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
    max_points: int = 1000,
    max_iter: int = 500,
):
    # Sample dataset to simplify tikz plot
    df_subset, _ = train_test_split(
        df,
        train_size=min(len(df), max_points),
        stratify=df_interm["target"],
        random_state=SEED,
    )

    # Separate features and target
    X = df_subset.drop(columns=[target_col])
    y = df_subset[target_col] > 0

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply TSNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
    )
    tsne_results = tsne.fit_transform(X_scaled)

    # Map class to color
    color_col = y.apply(lambda val: colors[0] if val else colors[1])

    data = pd.DataFrame(tsne_results, columns=["x", "y"])
    data["color"] = color_col
    return data


def pca_to_tikz(
    df: pd.DataFrame,
    target_col: str,
    colors: list[str],
    n_components: int = 2,
    max_points: int = 1000,
    random_state: int = 42,
):
    # Sample dataset to simplify tikz plot
    df_subset, _ = train_test_split(
        df,
        train_size=min(len(df), max_points),
        stratify=df_interm["target"],
        random_state=SEED,
    )

    # Separate features and target
    X = df_subset.drop(columns=[target_col])
    y = df_subset[target_col] > 0

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(X_scaled)

    # Map class to color
    color_col = y.apply(lambda val: colors[0] if val else colors[1])

    data = pd.DataFrame(pca_results, columns=["x", "y"])
    data["color"] = color_col
    return data


def main(
    output_file: str,
    overwrite: bool = False,
    impute_mode: str = "mean",
    colors: list = ["red", "blue"],
):
    df = load_data()
    df = fill_na(df, impute_mode)
    pca = pca_to_tikz(df, "target", colors=colors)
    tsne = tsne_to_tikz(df, "target", colors=colors)
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
