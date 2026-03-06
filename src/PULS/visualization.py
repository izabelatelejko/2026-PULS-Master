"""Module for visualization of evaluation results."""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_mae(df, pi_grid, label_frequency, verbose=False, show_se=True, K=None):
    """Plot Mean Absolute Error (MAE) with Standard Error (SE) for different methods across varying train and test priors.
    
    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'mae', 'se', and optionally 'converged'
        pi_grid: List of pi values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
    """
    df = df.copy()
    df = df.sort_values(['pi', 'new_pi'])
    
    # Track MLLS convergence info per point for annotations
    mlls_annotations = {}  # (pi, new_pi, method) -> "X/K"
    if 'converged' in df.columns and K is not None:
        mlls_mask = df['method'].str.contains('MLLS|mlls', case=False, na=False)
        for idx, row in df[mlls_mask].iterrows():
            key = (row['pi'], row['new_pi'], row['method'])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"
        
        # Filter out points where MLLS methods didn't converge (converged == 0)
        non_converged_mask = mlls_mask & (df['converged'] == 0)
        df = df[~non_converged_mask]
    
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="pi", col_wrap=2, height=4, sharey=True)
    palette = sns.color_palette('tab20', n_colors=df['method'].nunique())
    colors = dict(zip(df['method'].unique(), palette))

    if show_se and 'se' in df.columns:
        def plot_with_std(data, color, **kwargs):
            methods = data['method'].unique()
            for method in methods:
                sub = data[data['method'] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, 'gray')
                plt.plot(sub['new_pi'], sub['mae'], label=method, marker='o', linewidth=1, linestyle='-', markersize=5, color=c)
                plt.fill_between(
                    sub['new_pi'],
                    sub['mae'] - sub['se'],
                    sub['mae'] + sub['se'],
                    alpha=0.2,
                    color=c
                )
                # Add convergence annotations for MLLS methods
                if K is not None and 'MLLS' in method.upper():
                    for _, row in sub.iterrows():
                        key = (row['pi'], row['new_pi'], method)
                        if key in mlls_annotations:
                            plt.annotate(mlls_annotations[key], 
                                        (row['new_pi'], row['mae']),
                                        textcoords="offset points", xytext=(8, -10),
                                        ha='center', fontsize=8, color=c)
        g.map_dataframe(plot_with_std)
    else:
        def plot_without_std(data, color, **kwargs):
            methods = data['method'].unique()
            for method in methods:
                sub = data[data['method'] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, 'gray')
                plt.plot(sub['new_pi'], sub['mae'], label=method, marker='o', linewidth=1, linestyle='-', markersize=5, color=c)
                # Add convergence annotations for MLLS methods
                if K is not None and 'MLLS' in method.upper():
                    for _, row in sub.iterrows():
                        key = (row['pi'], row['new_pi'], method)
                        if key in mlls_annotations:
                            plt.annotate(mlls_annotations[key], 
                                        (row['new_pi'], row['mae']),
                                        textcoords="offset points", xytext=(8, -10),
                                        ha='center', fontsize=8, color=c)
        g.map_dataframe(plot_without_std)

    g.set(xticks=pi_grid)
    for ax in g.axes.flatten():
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    if verbose:
        g.set_axis_labels("Target class prior", "MAE")
        g.set_titles("Source class prior = {col_name}")
        g.figure.suptitle(f"label frequency $= {label_frequency}$", fontsize=12, x=0.47)
    else:
        g.set_axis_labels("$\pi'$", "MAE")
        g.set_titles("$\pi$ = {col_name}")
        g.figure.suptitle(f"$c = {label_frequency}$", fontsize=12, x=0.47)
        
    plt.tight_layout()
    g.add_legend(title="Method", loc='center right')    
    plt.show()


def plot_metric(df, metric, pi_grid, label_frequency, verbose=False, show_se=False, K=None):
    """Plot a given metric for different methods across varying train and test priors.
    
    Args:
        df: DataFrame with columns 'pi', 'new_pi', 'method', 'average_value', and optionally 'se', 'converged'
        metric: Name of the metric being plotted (for axis label)
        pi_grid: List of pi values for x-axis ticks
        label_frequency: Label frequency value for title
        verbose: If True, use verbose labels
        show_se: If True, plot standard error as shaded area (requires 'se' column in df)
        K: Total number of experiments (used for MLLS convergence annotation)
    """
    df = df.copy()
    df = df.sort_values(['pi', 'new_pi'])
    
    # Track MLLS convergence info per point for annotations
    mlls_annotations = {}  # (pi, new_pi, method) -> "X/K"
    if 'converged' in df.columns and K is not None:
        mlls_mask = df['method'].str.contains('MLLS|mlls', case=False, na=False)
        for idx, row in df[mlls_mask].iterrows():
            key = (row['pi'], row['new_pi'], row['method'])
            mlls_annotations[key] = f"{int(row['converged'])}/{K}"
        
        # Filter out points where MLLS methods didn't converge (converged == 0)
        non_converged_mask = mlls_mask & (df['converged'] == 0)
        df = df[~non_converged_mask]
    
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="pi", col_wrap=2, height=4, sharey=True)
    palette = sns.color_palette('tab20', n_colors=df['method'].nunique())
    colors = dict(zip(df['method'].unique(), palette))
    
    if show_se and 'se' in df.columns:
        # Plot with standard error shading
        def plot_with_se(data, color, **kwargs):
            methods = data['method'].unique()
            for method in methods:
                sub = data[data['method'] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, 'gray')
                plt.plot(sub['new_pi'], sub['average_value'], label=method, marker='o', 
                        linewidth=1, linestyle='-', markersize=5, color=c)
                plt.fill_between(
                    sub['new_pi'],
                    sub['average_value'] - sub['se'],
                    sub['average_value'] + sub['se'],
                    alpha=0.2,
                    color=c
                )
                # Add convergence annotations for MLLS methods
                if K is not None and 'MLLS' in method.upper():
                    for _, row in sub.iterrows():
                        key = (row['pi'], row['new_pi'], method)
                        if key in mlls_annotations:
                            plt.annotate(mlls_annotations[key], 
                                        (row['new_pi'], row['average_value']),
                                        textcoords="offset points", xytext=(8, -10),
                                        ha='center', fontsize=8, color=c)
        g.map_dataframe(plot_with_se)
    else:
        # Plot without SE
        def plot_lineplots(data, color, **kwargs):
            methods = data['method'].unique()
            for method in methods:
                sub = data[data['method'] == method]
                if len(sub) == 0:
                    continue
                c = colors.get(method, 'gray')
                plt.plot(sub['new_pi'], sub['average_value'], label=method, marker='o', 
                        linewidth=1, linestyle='-', markersize=6, color=c)
                # Add convergence annotations for MLLS methods
                if K is not None and 'MLLS' in method.upper():
                    for _, row in sub.iterrows():
                        key = (row['pi'], row['new_pi'], method)
                        if key in mlls_annotations:
                            plt.annotate(mlls_annotations[key], 
                                        (row['new_pi'], row['average_value']),
                                        textcoords="offset points", xytext=(8, -10),
                                        ha='center', fontsize=8, color=c)
        g.map_dataframe(plot_lineplots)

    g.set(xticks=pi_grid)
    for ax in g.axes.flatten():
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

    if verbose:
        g.set_axis_labels("Target class prior", f"Average {metric}")
        g.set_titles("Source class prior = {col_name}")
        g.figure.suptitle(f"label frequency = {label_frequency}", fontsize=12, x=0.47)
    else:
        g.set_axis_labels("$\pi'$", f"Average {metric}")
        g.set_titles("$\pi$ = {col_name}")
        g.figure.suptitle(f"$c = {label_frequency}$", fontsize=12, x=0.47)

    plt.tight_layout()
    g.add_legend(title="Model + Method", loc='center right')
    plt.show()