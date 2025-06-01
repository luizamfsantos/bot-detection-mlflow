import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
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
        stratify=df["target"],
        random_state=random_state,
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
        stratify=df["target"],
        random_state=random_state,
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


def plot_boxplot():
    results_path = Path("results")
    assert results_path.exists()
    file_list = results_path.glob("*cross_val_scores.json")
    results = {}
    for file in file_list:
        with open(file, "r") as f:
            results_tmp = json.load(f)
        results[file.stem] = list(results_tmp.values())[0]

    # Create DataFrame
    df = pd.DataFrame(
        [(model, score) for model, scores in results.items() for score in scores],
        columns=["Model", "Accuracy"],
    )

    # Set journal-appropriate styling
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.edgecolor": "black",
            "legend.linewidth": 1,
        }
    )

    # Create figure with journal-standard dimensions (single column width)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Use grayscale palette appropriate for print journals
    n_models = len(df["Model"].unique())
    grays = ["#2c2c2c", "#4d4d4d", "#6e6e6e", "#8f8f8f", "#b0b0b0", "#d1d1d1"]
    palette = (
        grays[:n_models]
        if n_models <= len(grays)
        else [f"#{int(i * 255 / (n_models - 1)):02x}" * 3 for i in range(n_models)]
    )

    # Create clean boxplot
    box_plot = sns.boxplot(
        data=df,
        x="Model",
        y="Accuracy",
        palette=palette,
        width=0.6,
        linewidth=1.5,
        fliersize=4,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            markersize=4,
            markeredgecolor="black",
            markeredgewidth=0.5,
        ),
    )

    # Add clean reference line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    # Calculate and display statistics at bottom
    grouped = df.groupby("Model")["Accuracy"]
    for i, (model, group) in enumerate(grouped):
        mean = group.mean()
        std = group.std()

        # Position text at bottom
        y_min = group.min()
        text_y = y_min - 0.012

        # Add statistics with journal formatting
        ax.text(
            i,
            text_y,
            f"μ = {mean:.3f}",
            ha="center",
            va="top",
            fontsize=10,
            color="black",
        )
        ax.text(
            i,
            text_y - 0.008,
            f"σ = {std:.3f}",
            ha="center",
            va="top",
            fontsize=10,
            color="black",
        )

    # Clean, professional labels
    ax.set_xlabel("Model", fontsize=12, fontweight="normal")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="normal")

    # Set appropriate axis limits
    y_min = df["Accuracy"].min() - 0.04
    y_max = min(df["Accuracy"].max() + 0.015, 1.01)
    ax.set_ylim(y_min, y_max)

    # Clean grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="major", labelsize=11, colors="black")
    if n_models > 4:
        ax.tick_params(axis="x", rotation=45)

    # Tight layout for journal submission
    plt.tight_layout()

    # Show plot
    # plt.show()

    # Save publication-ready version
    # Uncomment the following lines to save:
    plt.savefig(
        "model_accuracy_boxplot.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="pdf",
    )
    plt.savefig(
        "model_accuracy_boxplot.eps",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="eps",
    )


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
