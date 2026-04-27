"""Module for visualization of evaluation results."""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .evaluation import evaluate_all_metrics, evaluate_shifted_pi_estimation


def _save_plot(fig, folder, dataset_name, label_frequency, metric, identifier=None):
    """Save plot to file with unique name."""
    os.makedirs(folder, exist_ok=True)
    parts = [dataset_name.lower(), f"{label_frequency}", metric.replace(" ", "_")]
    if identifier:
        parts.append(identifier)
    filename = "_".join(str(p) for p in parts) + ".png"
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {filepath}")


def plot_mae(
    *args,
    is_real: bool = False,
    **kwargs,
):
    """Wrapper for plotting MAE with the same style for both synthetic and real data."""
    if is_real:
        _plot_mae_real(*args, **kwargs)
    else:
        _plot_mae_synth(*args, **kwargs)

def _plot_mae_synth(
    df,
    pi_grid,
    label_frequency,
    verbose=False,
    show_se=True,
    K=None,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot Mean Absolute Error (MAE) with Standard Error (SE) for different methods across varying train and test priors.

    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'mae', 'se', and optionally 'converged'
        pi_grid: List of pi values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    df = df.copy()
    df = df.sort_values(["pi", "new_pi"])

    # Track MLLS convergence info per point for annotations
    mlls_annotations = {}  # (pi, new_pi, method) -> "X/K"
    if "converged" in df.columns and K is not None:
        mlls_mask = df["method"].str.contains("MLLS|mlls", case=False, na=False)
        for idx, row in df[mlls_mask].iterrows():
            key = (row["pi"], row["new_pi"], row["method"])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"

        # Filter out points where MLLS methods didn't converge (converged == 0)
        non_converged_mask = mlls_mask & (df["converged"] == 0)
        df = df[~non_converged_mask]

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="pi", col_wrap=2, height=4, sharey=True)
    palette = sns.color_palette("tab20", n_colors=df["method"].nunique())
    colors = dict(zip(df["method"].unique(), palette))

    if show_se and "se" in df.columns:

        def plot_with_std(data, color, **kwargs):
            methods = data["method"].unique()
            for method in methods:
                sub = data[data["method"] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, "gray")
                plt.plot(
                    sub["new_pi"],
                    sub["mae"],
                    label=method,
                    marker="o",
                    linewidth=1,
                    linestyle="-",
                    markersize=5,
                    color=c,
                )
                plt.fill_between(
                    sub["new_pi"],
                    sub["mae"] - sub["se"],
                    sub["mae"] + sub["se"],
                    alpha=0.2,
                    color=c,
                )
                # Add convergence annotations for MLLS methods
                if K is not None and "MLLS" in method.upper():
                    for _, row in sub.iterrows():
                        key = (row["pi"], row["new_pi"], method)
                        if key in mlls_annotations:
                            plt.annotate(
                                mlls_annotations[key],
                                (row["new_pi"], row["mae"]),
                                textcoords="offset points",
                                xytext=(8, -10),
                                ha="center",
                                fontsize=8,
                                color=c,
                            )

        g.map_dataframe(plot_with_std)
    else:

        def plot_without_std(data, color, **kwargs):
            methods = data["method"].unique()
            for method in methods:
                sub = data[data["method"] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, "gray")
                plt.plot(
                    sub["new_pi"],
                    sub["mae"],
                    label=method,
                    marker="o",
                    linewidth=1,
                    linestyle="-",
                    markersize=5,
                    color=c,
                )
                # Add convergence annotations for MLLS methods
                if K is not None and "MLLS" in method.upper():
                    for _, row in sub.iterrows():
                        key = (row["pi"], row["new_pi"], method)
                        if key in mlls_annotations:
                            plt.annotate(
                                mlls_annotations[key],
                                (row["new_pi"], row["mae"]),
                                textcoords="offset points",
                                xytext=(8, -10),
                                ha="center",
                                fontsize=8,
                                color=c,
                            )

        g.map_dataframe(plot_without_std)

    g.set(xticks=pi_grid)
    for ax in g.axes.flatten():
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)

    if verbose:
        g.set_axis_labels("Target class prior", "MAE")
        g.set_titles("Source class prior = {col_name}")
        g.figure.suptitle(f"label frequency $= {label_frequency}$", fontsize=12, x=0.47)
    else:
        g.set_axis_labels("$\pi'$", "MAE")
        g.set_titles("$\pi$ = {col_name}")
        g.figure.suptitle(f"$c = {label_frequency}$", fontsize=12, x=0.47)

    plt.tight_layout()
    g.add_legend(title="Method", loc="center right")

    if save and dataset_name:
        _save_plot(
            g.figure, save_folder, dataset_name, label_frequency, "mae", identifier
        )

    plt.show()

def _plot_mae_real(
    df,
    train_pi,
    test_pis,
    label_frequency,
    verbose=False,
    show_se=True,
    K=None,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot MAE for real-data setting with a fixed source prior and multiple test priors.

    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'mae', and optionally 'se', 'converged'
        train_pi: Source prior value (float) or a single-value list/tuple
        test_pis: List of target prior values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    df = df.copy()
    df = df[np.isclose(df["pi"], train_pi)].sort_values(["new_pi", "method"])

    # Track MLLS convergence info per point for annotations.
    mlls_annotations = {}  # (new_pi, method) -> "X/K"
    if "converged" in df.columns and K is not None:
        mlls_mask = df["method"].str.contains("MLLS|mlls", case=False, na=False)
        for _, row in df[mlls_mask].iterrows():
            key = (row["new_pi"], row["method"])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"

        # Filter out points where MLLS methods didn't converge (converged == 0).
        non_converged_mask = mlls_mask & (df["converged"] == 0)
        df = df[~non_converged_mask]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = list(df["method"].unique())
    palette = sns.color_palette("tab20", n_colors=max(1, len(methods)))
    colors = dict(zip(methods, palette))

    for method in methods:
        sub = df[df["method"] == method].sort_values("new_pi")
        if len(sub) == 0:
            continue

        c = colors.get(method, "gray")
        ax.plot(
            sub["new_pi"],
            sub["mae"],
            label=method,
            marker="o",
            linewidth=1,
            linestyle="-",
            markersize=6,
            color=c,
        )

        if show_se and "se" in sub.columns:
            ax.fill_between(
                sub["new_pi"],
                sub["mae"] - sub["se"],
                sub["mae"] + sub["se"],
                alpha=0.2,
                color=c,
            )

        if K is not None and "MLLS" in method.upper():
            for _, row in sub.iterrows():
                key = (row["new_pi"], method)
                if key in mlls_annotations:
                    ax.annotate(
                        mlls_annotations[key],
                        (row["new_pi"], row["mae"]),
                        textcoords="offset points",
                        xytext=(8, -10),
                        ha="center",
                        fontsize=8,
                        color=c,
                    )

    ax.set_xticks(test_pis)
    ax.yaxis.grid(True, linestyle="--")
    ax.xaxis.grid(True, linestyle="--")

    if verbose:
        ax.set_xlabel("Target class prior")
        ax.set_ylabel("MAE")
        ax.set_title(
            f"Source class prior = {train_pi}, label frequency = {label_frequency}"
        )
    else:
        ax.set_xlabel("$\\pi'$")
        ax.set_ylabel("MAE")
        ax.set_title(f"$\\pi$ = {train_pi}, $c = {label_frequency}$")

    ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()

    if save and dataset_name:
        _save_plot(
            fig,
            save_folder,
            dataset_name,
            label_frequency,
            "mae",
            identifier,
        )

    plt.show()

def plot_metric(*args, is_real: bool = False, **kwargs):
    """Wrapper for plotting any metric (e.g., accuracy, balanced accuracy) with the same style as MAE."""
    if is_real:
        _plot_metric_real(*args, **kwargs)
    else:
        _plot_metric_synth(*args, **kwargs)


def _plot_metric_synth(
    df,
    metric,
    pi_grid,
    label_frequency,
    verbose=False,
    show_se=False,
    K=None,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot a given metric for different methods across varying train and test priors.

    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'average_value', and optionally 'se', 'converged'
        metric: Name of the metric being plotted (for axis label)
        pi_grid: List of pi values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    df = df.copy()
    df = df.sort_values(["pi", "new_pi"])

    # Track MLLS convergence info per point for annotations
    mlls_annotations = {}  # (pi, new_pi, method) -> "X/K"
    if "converged" in df.columns and K is not None:
        mlls_mask = df["method"].str.contains("MLLS|mlls", case=False, na=False)
        for idx, row in df[mlls_mask].iterrows():
            key = (row["pi"], row["new_pi"], row["method"])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"

        # Filter out points where MLLS methods didn't converge (converged == 0)
        non_converged_mask = mlls_mask & (df["converged"] == 0)
        df = df[~non_converged_mask]

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="pi", col_wrap=2, height=4, sharey=True)
    palette = sns.color_palette("tab20", n_colors=df["method"].nunique())
    colors = dict(zip(df["method"].unique(), palette))

    if show_se and "se" in df.columns:
        # Plot with standard error shading
        def plot_with_se(data, color, **kwargs):
            methods = data["method"].unique()
            for method in methods:
                sub = data[data["method"] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, "gray")
                plt.plot(
                    sub["new_pi"],
                    sub["average_value"],
                    label=method,
                    marker="o",
                    linewidth=1,
                    linestyle="-",
                    markersize=5,
                    color=c,
                )
                plt.fill_between(
                    sub["new_pi"],
                    sub["average_value"] - sub["se"],
                    sub["average_value"] + sub["se"],
                    alpha=0.2,
                    color=c,
                )
                # Add convergence annotations for MLLS methods
                if K is not None and "MLLS" in method.upper():
                    for _, row in sub.iterrows():
                        key = (row["pi"], row["new_pi"], method)
                        if key in mlls_annotations:
                            plt.annotate(
                                mlls_annotations[key],
                                (row["new_pi"], row["average_value"]),
                                textcoords="offset points",
                                xytext=(8, -10),
                                ha="center",
                                fontsize=8,
                                color=c,
                            )

        g.map_dataframe(plot_with_se)
    else:
        # Plot without SE
        def plot_lineplots(data, color, **kwargs):
            methods = data["method"].unique()
            for method in methods:
                sub = data[data["method"] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, "gray")
                plt.plot(
                    sub["new_pi"],
                    sub["average_value"],
                    label=method,
                    marker="o",
                    linewidth=1,
                    linestyle="-",
                    markersize=6,
                    color=c,
                )
                # Add convergence annotations for MLLS methods
                if K is not None and "MLLS" in method.upper():
                    for _, row in sub.iterrows():
                        key = (row["pi"], row["new_pi"], method)
                        if key in mlls_annotations:
                            plt.annotate(
                                mlls_annotations[key],
                                (row["new_pi"], row["average_value"]),
                                textcoords="offset points",
                                xytext=(8, -10),
                                ha="center",
                                fontsize=8,
                                color=c,
                            )

        g.map_dataframe(plot_lineplots)

    g.set(xticks=pi_grid)
    for ax in g.axes.flatten():
        ax.yaxis.grid(True, linestyle="--")
        ax.xaxis.grid(True, linestyle="--")

    if verbose:
        g.set_axis_labels("Target class prior", f"Average {metric}")
        g.set_titles("Source class prior = {col_name}")
        g.figure.suptitle(f"label frequency = {label_frequency}", fontsize=12, x=0.47)
    else:
        g.set_axis_labels("$\pi'$", f"Average {metric}")
        g.set_titles("$\pi$ = {col_name}")
        g.figure.suptitle(f"$c = {label_frequency}$", fontsize=12, x=0.47)

    plt.tight_layout()
    g.add_legend(title="Model + Method", loc="center right")

    if save and dataset_name:
        _save_plot(
            g.figure,
            save_folder,
            dataset_name,
            label_frequency,
            metric.lower(),
            identifier,
        )

    plt.show()


def _plot_metric_real(
    df,
    metric,
    train_pi,
    test_pis,
    label_frequency,
    verbose=False,
    show_se=False,
    K=None,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot metric for real-data setting with a fixed source prior and multiple test priors.

    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'average_value', and optionally 'se', 'converged'
        metric: Name of the metric being plotted (for axis label)
        train_pi: Source prior value (float) or a single-value list/tuple
        test_pis: List of target prior values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    df = df.copy()
    df = df[np.isclose(df["pi"], train_pi)].sort_values(["new_pi", "method"])

    # Track MLLS convergence info per point for annotations.
    mlls_annotations = {}  # (new_pi, method) -> "X/K"
    if "converged" in df.columns and K is not None:
        mlls_mask = df["method"].str.contains("MLLS|mlls", case=False, na=False)
        for _, row in df[mlls_mask].iterrows():
            key = (row["new_pi"], row["method"])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"

        # Filter out points where MLLS methods didn't converge (converged == 0).
        non_converged_mask = mlls_mask & (df["converged"] == 0)
        df = df[~non_converged_mask]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = list(df["method"].unique())
    palette = sns.color_palette("tab20", n_colors=max(1, len(methods)))
    colors = dict(zip(methods, palette))

    for method in methods:
        sub = df[df["method"] == method].sort_values("new_pi")
        if len(sub) == 0:
            continue

        c = colors.get(method, "gray")
        ax.plot(
            sub["new_pi"],
            sub["average_value"],
            label=method,
            marker="o",
            linewidth=1,
            linestyle="-",
            markersize=6,
            color=c,
        )

        if show_se and "se" in sub.columns:
            ax.fill_between(
                sub["new_pi"],
                sub["average_value"] - sub["se"],
                sub["average_value"] + sub["se"],
                alpha=0.2,
                color=c,
            )

        if K is not None and "MLLS" in method.upper():
            for _, row in sub.iterrows():
                key = (row["new_pi"], method)
                if key in mlls_annotations:
                    ax.annotate(
                        mlls_annotations[key],
                        (row["new_pi"], row["average_value"]),
                        textcoords="offset points",
                        xytext=(8, -10),
                        ha="center",
                        fontsize=8,
                        color=c,
                    )

    ax.set_xticks(test_pis)
    ax.yaxis.grid(True, linestyle="--")
    ax.xaxis.grid(True, linestyle="--")

    if verbose:
        ax.set_xlabel("Target class prior")
        ax.set_ylabel(f"Average {metric}")
        ax.set_title(
            f"Source class prior = {train_pi}, label frequency = {label_frequency}"
        )
    else:
        ax.set_xlabel("$\\pi'$")
        ax.set_ylabel(f"Average {metric}")
        ax.set_title(f"$\\pi$ = {train_pi}, $c = {label_frequency}$")

    ax.legend(title="Model + Method", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()

    if save and dataset_name:
        _save_plot(
            fig,
            save_folder,
            dataset_name,
            label_frequency,
            metric.lower(),
            identifier,
        )

    plt.show()


def plot_real_accuracy_grid(
    metric="accuracy",
    dataset_names=None,
    mean=None,
    n=5000,
    label_frequency=0.5,
    train_pis=(0.5,),
    test_pis=(0.2, 0.4, 0.6, 0.8),
    methods=None,
    verbose=True,
    show_se=True,
    K=10,
    save=False,
    save_plot_name="real_datasets",
    identifier=None,
    save_folder="results_img",
    figsize=(11, 13),
):
    """Plot average metric for six real datasets in a 2x3 grid."""
    if dataset_names is None:
        dataset_names = [
            "MNIST",
            "FashionMNIST",
            "ChestXRay",
            "Electricity",
            "Covertype",
            "SMSSpam",
        ]

    if methods is None:
        methods = [
            "DRPU",
            "nnPU",
            "nnPU+TA+KM2",
            "DRPU+TA+KM2",
            "nnPU+Target",
        ]

    if len(dataset_names) != 6:
        raise ValueError("dataset_names must contain exactly 6 datasets for a 2x3 grid.")

    train_pi = train_pis[0]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    palette = sns.color_palette("tab20", n_colors=max(1, len(methods)))
    colors = dict(zip(methods, palette))
    legend_handles = {}

    for ax, dataset in zip(axes_flat, dataset_names):
        try:
            df_metric = evaluate_all_metrics(
                dataset,
                mean,
                n,
                label_frequency,
                train_pis,
                test_pis,
                convert_to_df=True,
                single_exp=False,
            )
        except FileNotFoundError:
            ax.set_title(dataset, fontsize=13)
            ax.text(
                0.5,
                0.5,
                "Missing results",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks(test_pis)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", labelsize=12)
            continue

        df_metric = df_metric[
            (df_metric["metric"] == metric)
            & (df_metric["method"].isin(methods))
            & (np.isclose(df_metric["pi"], train_pi))
        ].sort_values(["new_pi", "method"])

        if df_metric.empty:
            ax.set_title(dataset, fontsize=13)
            ax.text(
                0.5,
                0.5,
                "No matching rows",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks(test_pis)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", labelsize=12)
            continue

        for method in methods:
            sub = df_metric[df_metric["method"] == method].sort_values("new_pi")
            if len(sub) == 0:
                continue

            color = colors.get(method, "gray")
            (line,) = ax.plot(
                sub["new_pi"],
                sub["average_value"],
                label=method,
                marker="o",
                linewidth=1.5,
                linestyle="-",
                markersize=6,
                color=color,
            )

            if method not in legend_handles:
                legend_handles[method] = line

            if show_se and "se" in sub.columns:
                ax.fill_between(
                    sub["new_pi"],
                    sub["average_value"] - sub["se"],
                    sub["average_value"] + sub["se"],
                    alpha=0.2,
                    color=color,
                )

            if K is not None and "converged" in sub.columns and "MLLS" in method.upper():
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"{int(row['converged'])}/{K}",
                        (row["new_pi"], row["average_value"]),
                        textcoords="offset points",
                        xytext=(8, -10),
                        ha="center",
                        fontsize=12,
                        color=color,
                    )

        ax.set_title(dataset, fontsize=13)
        ax.set_xticks(test_pis)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=12)

    for idx, ax in enumerate(axes_flat):
        row_idx, col_idx = divmod(idx, 2)
        ax.set_xlabel("Target class prior", fontsize=12)
        ax.tick_params(axis="x", labelbottom=True)
        if col_idx == 0:
            ax.set_ylabel(f"Average {metric.capitalize()}", fontsize=12)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            ncol=max(1, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
            title="Method",
            fontsize=12,
            title_fontsize=12,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        _save_plot(
            fig,
            save_folder,
            save_plot_name,
            label_frequency,
            metric,
            identifier,
        )

    plt.show()

    return fig, axes


def plot_real_mae_grid(
    dataset_names=None,
    mean=None,
    n=5000,
    label_frequency=0.5,
    train_pis=(0.5,),
    test_pis=(0.2, 0.4, 0.6, 0.8),
    methods=None,
    verbose=True,
    show_se=True,
    K=10,
    save=False,
    save_plot_name="real_datasets_mae",
    identifier=None,
    save_folder="results_img",
    figsize=(11, 13),
):
    """Plot MAE for six real datasets in a 2x3 grid."""
    if dataset_names is None:
        dataset_names = [
            "MNIST",
            "FashionMNIST",
            "ChestXRay",
            "Electricity",
            "Covertype",
            "SMSSpam",
        ]

    if methods is None:
        methods = ["KM2", "DRE"]

    if len(dataset_names) != 6:
        raise ValueError("dataset_names must contain exactly 6 datasets for a 2x3 grid.")

    train_pi = train_pis[0]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    palette = sns.color_palette("tab20", n_colors=max(1, len(methods)))
    colors = dict(zip(methods, palette))
    legend_handles = {}

    for ax, dataset in zip(axes_flat, dataset_names):
        try:
            df_mae = evaluate_shifted_pi_estimation(
                dataset,
                mean,
                n,
                label_frequency,
                train_pis,
                test_pis,
                convert_to_df=True,
                single_exp=False,
                nnpu_only=False,
            )
        except FileNotFoundError:
            ax.set_title(dataset, fontsize=13)
            ax.text(
                0.5,
                0.5,
                "Missing results",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks(test_pis)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", labelsize=12)
            continue

        df_mae = df_mae[
            (df_mae["method"].isin(methods))
            & (np.isclose(df_mae["pi"], train_pi))
        ].sort_values(["new_pi", "method"])

        if df_mae.empty:
            ax.set_title(dataset, fontsize=13)
            ax.text(
                0.5,
                0.5,
                "No matching rows",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks(test_pis)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", labelsize=12)
            continue

        for method in methods:
            sub = df_mae[df_mae["method"] == method].sort_values("new_pi")
            if len(sub) == 0:
                continue

            color = colors.get(method, "gray")
            (line,) = ax.plot(
                sub["new_pi"],
                sub["mae"],
                label=method,
                marker="o",
                linewidth=1.5,
                linestyle="-",
                markersize=6,
                color=color,
            )

            if method not in legend_handles:
                legend_handles[method] = line

            if show_se and "se" in sub.columns:
                ax.fill_between(
                    sub["new_pi"],
                    sub["mae"] - sub["se"],
                    sub["mae"] + sub["se"],
                    alpha=0.2,
                    color=color,
                )

            if K is not None and "converged" in sub.columns and "MLLS" in method.upper():
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"{int(row['converged'])}/{K}",
                        (row["new_pi"], row["mae"]),
                        textcoords="offset points",
                        xytext=(8, -10),
                        ha="center",
                        fontsize=12,
                        color=color,
                    )

        ax.set_title(dataset, fontsize=13)
        ax.set_xticks(test_pis)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=12)

    for idx, ax in enumerate(axes_flat):
        _, col_idx = divmod(idx, 2)
        ax.set_xlabel("Target class prior", fontsize=12)
        ax.tick_params(axis="x", labelbottom=True)
        if col_idx == 0:
            ax.set_ylabel("MAE", fontsize=12)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            ncol=max(1, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
            title="Method",
            fontsize=12,
            title_fontsize=12,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        _save_plot(
            fig,
            save_folder,
            save_plot_name,
            label_frequency,
            "mae",
            identifier,
        )

    plt.show()

    return fig, axes


def plot_roc(
    metrics_contents,
    model="drpu",
    train_pi=None,
    test_pi=None,
    show_optimal=True,
    show_youden=True,
    show_05=True,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot ROC curve with various threshold markers.

    Args:
        metrics_contents: Dictionary containing 'test_pis' and 'roc_curve' data from metrics.json
        model: Model name ('drpu' or 'nnpu') for which to plot ROC curve
        train_pi: Training prior (π) - used for optimal threshold calculation
        test_pi: Test prior (π') - used for optimal threshold calculation
        pi_methods: List of pi estimation methods to show (e.g., ['dre', 'km2']). If None, shows all available.
        show_optimal: If True, show optimal threshold point (requires train_pi and test_pi)
        show_youden: If True, show Youden's J statistic threshold
        show_05: If True, show threshold = 0.5 point
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    roc_data = metrics_contents["roc_curve"][model]
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    thresholds = [float(t) for t in roc_data["thresholds"]]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color="purple", linewidth=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="Random")

    # Threshold = 0.5
    if show_05:
        idx_05 = np.argmin(np.abs([t - 0.5 for t in thresholds]))
        ax.scatter(
            fpr[idx_05],
            tpr[idx_05],
            color="orange",
            s=100,
            zorder=5,
            marker="*",
            label="Threshold = 0.5",
        )

    # Optimal threshold (true π')
    if show_optimal and train_pi is not None and test_pi is not None:
        optimal_threshold = (
            train_pi
            * (1 - test_pi)
            / ((1 - train_pi) * test_pi + train_pi * (1 - test_pi))
        )
        idx_optimal = np.argmin(np.abs([t - optimal_threshold for t in thresholds]))
        ax.scatter(
            fpr[idx_optimal],
            tpr[idx_optimal],
            color="red",
            s=100,
            zorder=5,
            marker="^",
            label=f"Target (true $\\pi'$) = {optimal_threshold:.2f}",
        )

    # Youden's J Statistic - best balanced accuracy
    if show_youden:
        J = [tpr[i] - fpr[i] for i in range(len(fpr))]
        idx_youden = np.argmax(J)
        youden_threshold = thresholds[idx_youden]
        ax.scatter(
            fpr[idx_youden],
            tpr[idx_youden],
            color="blue",
            s=100,
            zorder=5,
            marker="^",
            label=f"Youden's J = {youden_threshold:.2f}",
        )

    for pi_method, color in zip(["km2", "dre"], ["green", "olive"]):
        if pi_method not in metrics_contents.get("test_pis", {}):
            continue
        pi = metrics_contents["test_pis"][pi_method]
        if pi is None or train_pi is None:
            continue
        corrected_threshold = (
            train_pi * (1 - pi) / ((1 - train_pi) * pi + train_pi * (1 - pi))
        )
        idx_corr = np.argmin(np.abs([t - corrected_threshold for t in thresholds]))
        ax.scatter(
            fpr[idx_corr],
            tpr[idx_corr],
            color=color,
            s=100,
            zorder=5,
            marker="x",
            linewidths=2,
            label=f"{pi_method.upper()} = {corrected_threshold:.2f} ($\\hat{{\\pi}}'={pi:.2f}$)",
        )
        ax.annotate(
            f"{pi_method.upper()}",
            (fpr[idx_corr], tpr[idx_corr]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color=color,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"$\\pi={train_pi}, \\pi'={test_pi}$", fontsize=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()

    if save and dataset_name:
        _save_plot(
            fig,
            save_folder,
            dataset_name,
            f"pi{train_pi}_pip{test_pi}",
            f"roc_{model}",
            identifier,
        )

    plt.show()

    return fig, ax


def plot_roc_grid(
    metrics_list,
    model="drpu",
    pi_pairs=None,
    show_optimal=True,
    show_youden=True,
    show_05=True,
    save=False,
    dataset_name=None,
    identifier=None,
    save_folder="results_img",
):
    """Plot ROC curves in a 2x2 grid for multiple (train_pi, test_pi) pairs.

    Args:
        metrics_list: List of 4 dictionaries containing 'test_pis' and 'roc_curve' data,
                      OR a function/callable that takes (train_pi, test_pi) and returns metrics_contents
        model: Model name ('drpu' or 'nnpu') for which to plot ROC curve
        pi_pairs: List of 4 tuples [(train_pi1, test_pi1), (train_pi2, test_pi2), ...].
                  Required if metrics_list is a callable.
        show_optimal: If True, show optimal threshold point
        show_youden: If True, show Youden's J statistic threshold
        show_05: If True, show threshold = 0.5 point
        save: If True, save the plot to file
        dataset_name: Name of the dataset (used for filename)
        identifier: Additional identifier for filename
        save_folder: Folder to save plots to
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, (metrics_contents, (train_pi, test_pi)) in enumerate(
        zip(metrics_list, pi_pairs)
    ):
        ax = axes[idx]

        roc_data = metrics_contents["roc_curve"][model]
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        thresholds = [float(t) for t in roc_data["thresholds"]]

        # Plot ROC curve
        ax.plot(fpr, tpr, color="purple", linewidth=2, label="ROC Curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="Random")

        # Threshold = 0.5
        if show_05:
            idx_05 = np.argmin(np.abs([t - 0.5 for t in thresholds]))
            ax.scatter(
                fpr[idx_05],
                tpr[idx_05],
                color="orange",
                s=80,
                zorder=5,
                marker="*",
                label="Threshold = 0.5",
            )

        # Optimal threshold (true pi')
        if show_optimal and train_pi is not None and test_pi is not None:
            optimal_threshold = (
                train_pi
                * (1 - test_pi)
                / ((1 - train_pi) * test_pi + train_pi * (1 - test_pi))
            )
            idx_optimal = np.argmin(np.abs([t - optimal_threshold for t in thresholds]))
            ax.scatter(
                fpr[idx_optimal],
                tpr[idx_optimal],
                color="red",
                s=80,
                zorder=5,
                marker="^",
                label=f"Target = {optimal_threshold:.2f}",
            )

        # Youden's J Statistic
        if show_youden:
            J = [tpr[i] - fpr[i] for i in range(len(fpr))]
            idx_youden = np.argmax(J)
            youden_threshold = thresholds[idx_youden]
            ax.scatter(
                fpr[idx_youden],
                tpr[idx_youden],
                color="blue",
                s=80,
                zorder=5,
                marker="^",
                label=f"Youden's J = {youden_threshold:.2f}",
            )

        # Estimated pi' methods
        methods_to_show = ["dre", "km2"]

        colors = ["green", "olive"]
        for i, pi_method in enumerate(methods_to_show):
            if pi_method not in metrics_contents.get("test_pis", {}):
                continue
            pi = metrics_contents["test_pis"][pi_method]
            if pi is None or train_pi is None:
                continue
            corrected_threshold = (
                train_pi * (1 - pi) / ((1 - train_pi) * pi + train_pi * (1 - pi))
            )
            idx_corr = np.argmin(np.abs([t - corrected_threshold for t in thresholds]))
            ax.scatter(
                fpr[idx_corr],
                tpr[idx_corr],
                color=colors[i],
                s=80,
                zorder=5,
                marker="x",
                linewidths=2,
                label=f"{pi_method.upper()} = {corrected_threshold:.2f} ($\\hat{{\\pi}}'={pi:.2f}$)",
            )
            ax.annotate(
                f"{pi_method.upper()}",
                (fpr[idx_corr], tpr[idx_corr]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=10,
                color=colors[i],
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"$\\pi={train_pi}, \\pi'={test_pi}$", fontsize=12)
        ax.set_xlabel("FPR", fontsize=11)
        ax.set_ylabel("TPR", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(f"label frequency = {0.5}", fontsize=12, y=1.01)
    plt.tight_layout()

    if save and dataset_name:
        pi_str = "_".join([f"{tp}-{tep}" for tp, tep in pi_pairs])
        _save_plot(
            fig, save_folder, dataset_name, pi_str, f"roc_grid_{model}", identifier
        )

    plt.show()

    return fig, axes


def plot_accuracy_threshold(
    metrics_list, 
    pi_pairs,
    thresholds_list,
    save=True,
    save_folder="results_img",
    dataset_name='',
):
    """Plot accuracy vs threshold for multiple (train_pi, test_pi) pairs in a grid.

    Args:
        metrics_list: List of 4 dictionaries containing 'accuracy_thresholds' data,
                      OR a function/callable that takes (train_pi, test_pi) and returns metrics_contents
        pi_pairs: List of 4 tuples [(train_pi1, test_pi1), (train_pi2, test_pi2), ...].
        save: If True, save the plot to file
    """

    n_rows = len(pi_pairs)
    tick_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True, sharey=True)

    for row, (metrics_contents, (train_pi, test_pi), (km2_threshold, dre_threshold, true_threshold)) in enumerate(zip(metrics_list, pi_pairs, thresholds_list)):
        left_pos = axes[row][0].get_position()
        right_pos = axes[row][1].get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        y_top = (1 / n_rows - 0.013) * (n_rows - row) - 0.01
        fig.text(
            x_center,
            y_top,
            f"Source prior $\\pi={train_pi}$, Target prior $\\pi'={test_pi}$",
            ha="center",
            va="bottom",
            fontsize=14,
        )

        for col, model in enumerate(["nnPU", "DRPU"]):
            ax = axes[row][col]
            accuracies = metrics_contents[model]["accuracy"]
            thresholds = metrics_contents[model]["thresholds"]

            ax.plot(thresholds, accuracies, linewidth=2)
            ax.axvline(0.5, color="darkgrey", linestyle="--", linewidth=1.6)
            ax.axvline(true_threshold, color="tab:green", linestyle="--", linewidth=1.6)
            ax.axvline(km2_threshold, color="tab:orange", linestyle="--", linewidth=1.6)
            ax.axvline(dre_threshold, color="tab:red", linestyle="--", linewidth=1.6)
            ax.set_title(model, fontsize=12)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Accuracy")
            ax.set_xlim(0, 1)
            ax.set_xticks(tick_values)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="tab:green", linestyle="--", linewidth=1.8, label="True"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.8, label="KM2"),
        Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.8, label="DRE"),
        Line2D([0], [0], color="darkgrey", linestyle="--", linewidth=1.8, label="1/2"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.2, 0.93, 0.6, 0.05),
        mode="expand",
        fontsize=12,
    )

    fig.tight_layout(h_pad=3.0, rect=[0, 0, 1, 0.94])

    if save:
        pi_str = "_".join([f"{tp}-{tep}" for tp, tep in pi_pairs])
        _save_plot(
            fig,
            save_folder,
            dataset_name=f"{dataset_name}_accuracy_threshold",
            label_frequency=pi_str,
            metric="",
        )

    plt.show()

    return fig, axes


def plot_boxplots_metric(
    df_metric,
    dataset_name,
    metric="accuracy",
    label_frequency=0.5,
    save=False,
    identifier=None,
    save_folder="results_img",
):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_metric,
        x='method',
        y='value',
        ax=ax,
        color='steelblue',
        showfliers=False,
    )
    ax.set_xlabel('Method')
    ax.set_ylabel(metric.capitalize())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        _save_plot(
            fig,
            save_folder,
            dataset_name,
            label_frequency,
            f"boxplot_{metric}",
            identifier,
        )

    plt.show()

def plot_boxplots_mae(
    df_mae,
    dataset_name,
    label_frequency=0.5,
    save=False,
    identifier=None,
    save_folder="results_img",
):

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(
        data=df_mae,
        x='method',
        y='mae',
        ax=ax,
        color='steelblue',
        showfliers=False,
    )
    ax.set_xlabel('Method')
    ax.set_ylabel('MAE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        _save_plot(
            fig,
            save_folder,
            dataset_name,
            label_frequency,
            "boxplot_mae",
            identifier,
        )

    plt.show()