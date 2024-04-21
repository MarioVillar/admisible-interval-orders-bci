import pandas as pd
import numpy as np
import plotly.graph_objects as go

import config
from utils.disk import check_create_folder


def bar_plot_by_subject(results: pd.DataFrame, save_to_disk: bool = False, img_name: str = None):
    """
    Plot the ROC AUC results by subject and model in a bar plot.
    Plots only test results.

    Parameters
    ----------
    results : pd.DataFrame
        The results dataframe.
    save_to_disk : bool, optional
        Whether to save the plot to disk, by default False.
    img_name : str, optional
        The image name. Mandatory if save_to_disk is True.

    Returns
    -------
    go.Figure
        The plotly figure.
    """
    # Filter test results
    test_results = results[results["session"].str.contains("test")]

    # Create figure
    fig = go.Figure()

    colors = {
        "IntvlMeanEnsemble": "#A4CE95",
        "IntvlChoquetEnsemble": "#5F5D9C",
        "IntvlSugenoEnsemble": "#6196A6",
        "CSP-LDA": "#F7DCB9",
    }

    legend_names = {
        "IntvlMeanEnsemble": "Aggregation by mean",
        "IntvlChoquetEnsemble": "Aggregation by Choquet Integral",
        "IntvlSugenoEnsemble": "Aggregation by Sugeno Integral",
        "CSP-LDA": "CSP + LDA",
    }

    # Plot each model
    for pipeline, pdata in test_results.groupby("pipeline"):
        fig.add_trace(
            go.Bar(
                x=pdata["score"],
                y=pdata["subject"],
                name=legend_names[pipeline],
                error_x=dict(type="data", array=pdata["score_std"], visible=True),
                orientation="h",
                marker_color=colors[pipeline],
            )
        )

    # Format plot
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="#323232", mirror=True, tickfont=dict(size=12), titlefont=dict(size=14)
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="#323232",
        mirror=True,
        range=[0, 1],
        gridcolor="black",
        dtick=0.1,
        tickfont=dict(size=12),
        titlefont=dict(size=14),
    )

    fig.update_layout(
        height=600,
        width=1000,
        title=dict(text="ROC AUC en 5-fold CV por sujeto y modelo", x=0.25),
        xaxis_title="ROC AUC",
        yaxis_title="Sujeto",
        plot_bgcolor="white",
        font=dict(family="Times New Roman"),
        legend=dict(font=dict(size=14), traceorder="reversed"),
        margin=dict(t=40),
    )

    fig.show()

    # Save image to disk
    if save_to_disk and img_name is not None:
        img_folder = f"{config.DISK_PATH}/images"
        check_create_folder(img_folder)
        fig.write_image(f"{img_folder}/{img_name}.svg", format="svg", scale=6)

    return fig


def line_plot_by_time_intvl(w_times: pd.DataFrame, scores_windows, save_to_disk: bool = False, img_name: str = None):
    """
    Plot the classification accuracy over time intervals.

    Parameters
    ----------
    w_times : pd.DataFrame
        The time intervals.
    scores_windows : np.ndarray
        The classification accuracy scores.
    save_to_disk : bool, optional
        Whether to save the plot to disk, by default False
    img_name : str, optional
        The image name. Mandatory if save_to_disk is True.

    Returns
    -------
    go.Figure
        The plotly figure.
    """
    mean_scores = np.mean(scores_windows, 0)

    fig = go.Figure()

    # Add score line
    fig.add_trace(go.Scatter(x=w_times, y=mean_scores, mode="lines", name="Score", line_color="#074173", line_width=3))

    # Add reference lines
    fig.add_vline(x=0, line=dict(color="black", width=2, dash="dash"), name="Onset")
    fig.add_hline(y=0.5, line=dict(color="black", width=2, dash="dash"), name="Chance")

    # Format plot
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#323232", mirror=True, gridcolor="white", gridwidth=0.5)

    fig.update_xaxes(showline=True, linewidth=1, linecolor="#323232", mirror=True, gridcolor="white", gridwidth=0.5)

    fig.update_layout(
        xaxis_title="Time interval (s)",
        yaxis_title="Classification accuracy",
        title=None,
        legend=dict(font=dict(size=14), orientation="h"),
        plot_bgcolor="#f2f2f2",
        height=600,
        width=1000,
        font=dict(family="Times New Roman"),
        margin=dict(t=40),
    )

    fig.show()

    # Save image to disk
    if save_to_disk and img_name is not None:
        img_folder = f"{config.DISK_PATH}/images"
        check_create_folder(img_folder)
        fig.write_image(f"{img_folder}/{img_name}.svg", format="svg", scale=6)

    return fig


if __name__ == "__main__":
    results = pd.read_csv(f"{config.DISK_PATH}/results.csv")

    bar_plot_by_subject(results=results, save_to_disk=config.SAVE_TO_DISK, img_name="roc_auc_5_fold_BNCI2014_001")
