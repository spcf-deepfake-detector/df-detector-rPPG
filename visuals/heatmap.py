
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


def _prepare_save_path(save_path: str) -> None:
    """Create the output directory if one is specified."""
    if not save_path or not save_path.strip():
        raise ValueError("save_path cannot be empty.")

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def plot_feature_heatmap(
    df_features: pd.DataFrame,
    save_path: str = "results/heatmaps/feature_correlation.png",
    cmap: str = "coolwarm",
    show_plot: bool = False
) -> pd.DataFrame:
    """
    Plot Pearson correlations between numeric rPPG features.

    Returns:
        DataFrame containing the correlation matrix.
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError("df_features must be a pandas DataFrame.")

    if df_features.empty:
        raise ValueError("df_features contains no rows.")

    numeric_df = df_features.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two numeric feature columns are required."
        )

    # Remove columns that have no valid values or no variation.
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[
        :, numeric_df.nunique(dropna=True) > 1
    ]

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two numeric features with variation are required."
        )

    # Pearson correlation ignores missing pairs.
    corr_matrix = numeric_df.corr(method="pearson")

    if corr_matrix.empty:
        raise ValueError("Could not calculate the correlation matrix.")

    _prepare_save_path(save_path)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
        ax=ax
    )

    ax.set_title(
        "FalseFacet — Physiological Feature Correlations",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Feature correlation heatmap saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return corr_matrix


def plot_custom_heatmap(
    matrix: Union[np.ndarray, list],
    title: str = "Feature Map Heatmap",
    xlabel: str = "X Dimension",
    ylabel: str = "Y Dimension",
    save_path: str = "results/heatmaps/custom_heatmap.png",
    cmap: str = "viridis",
    show_plot: bool = False
) -> np.ndarray:
    """Plot a validated two-dimensional numeric matrix."""
    matrix_arr = np.asarray(matrix, dtype=np.float64)

    if matrix_arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix, received shape {matrix_arr.shape}."
        )

    if matrix_arr.size == 0:
        raise ValueError("The matrix cannot be empty.")

    if not np.all(np.isfinite(matrix_arr)):
        raise ValueError(
            "The matrix contains NaN or infinite values."
        )

    _prepare_save_path(save_path)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    sns.heatmap(
        matrix_arr,
        cmap=cmap,
        annot=False,
        cbar=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Custom heatmap saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return matrix_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FalseFacet rPPG Heatmap Visualizer"
    )
    parser.add_argument(
        "--save_path",
        default="results/heatmaps/feature_correlation.png"
    )
    args = parser.parse_args()

    print("[i] Testing heatmap visualization with synthetic data...")

    rng = np.random.default_rng(42)
    n_samples = 50

    mock_features = pd.DataFrame({
        "snr_db": rng.uniform(-5, 15, n_samples),
        "spectral_concentration": rng.uniform(0.1, 0.9, n_samples),
        "peak_freq_hz": rng.uniform(0.8, 2.2, n_samples),
        "hr_bpm": rng.uniform(50, 130, n_samples),
        "signal_quality": rng.uniform(0.2, 0.95, n_samples),
        "fake_probability": rng.uniform(0.0, 1.0, n_samples)
    })

    corr_df = plot_feature_heatmap(
        mock_features,
        save_path=args.save_path
    )

    print("[+] Correlation matrix shape:", corr_df.shape)
    print("[i] Synthetic data only; this is not model evaluation.")

#python -m visual.heatmap