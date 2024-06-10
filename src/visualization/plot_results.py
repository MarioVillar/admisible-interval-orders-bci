import pandas as pd
import numpy as np
import plotly.graph_objects as go

import config
from utils.disk import check_create_folder


def bar_plot_by_subject(
    results: pd.DataFrame, plot_csp_lda: bool = False, save_to_disk: bool = False, img_name: str = None
):
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
    if not plot_csp_lda:
        results = results[results["pipeline"] != "CSP-LDA"]

    # Filter test results
    test_results = results[results["session"].str.contains("test")]

    # Create figure
    fig = go.Figure()

    colors = {
        "IntvlMeanEnsemble": "#A4CE95",
        "IntvlChoquetEnsemble": "#5F5D9C",
        "IntvlChoquetEnsembleNBest": "#a897ff",
        "IntvlSugenoEnsemble": "#6196A6",
        "CSP-LDA": "#F7DCB9",
    }

    legend_names = {
        "CSP-LDA": "CSP + LDA",
        "IntvlChoquetEnsembleNBest": "Ensemble + Choquet (3 best)",
        "IntvlChoquetEnsemble": "Ensemble + Choquet",
        "IntvlSugenoEnsemble": "Ensemble + Sugeno",
        "IntvlMeanEnsemble": "Ensemble + Mean",
    }

    test_results_grouped = test_results.groupby("pipeline")

    for pipeline in list(legend_names.keys())[::-1]:
        # Plot each model
        pdata = test_results_grouped.get_group(pipeline)

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
        showline=True,
        linewidth=1,
        linecolor="#323232",
        mirror=True,
        autorange="reversed",
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="#323232",
        mirror=True,
        range=[0, 1],
        gridcolor="black",
        dtick=0.1,
    )

    fig.update_layout(
        height=900,
        width=800,
        # title=dict(text="ROC AUC per subject and model in BCIGRAZ", x=0.25),
        xaxis_title="ROC AUC",
        yaxis_title="Subject",
        plot_bgcolor="white",
        font=dict(family="Times New Roman", size=22),
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
        ),
        margin=dict(t=20),
    )

    fig.show()

    # Save image to disk
    if save_to_disk and img_name is not None:
        img_folder = f"{config.DISK_PATH}/images"
        check_create_folder(img_folder)
        fig.write_image(f"{img_folder}/{img_name}.svg", format="svg", scale=6)

    return fig


def line_plot_by_time_intvl(results: pd.DataFrame, save_to_disk: bool = False, img_name: str = None):
    """
    Plot the classification accuracy over time intervals.

    Parameters
    ----------
    results : pd.DataFrame
        The results dataframe.
    save_to_disk : bool, optional
        Whether to save the plot to disk, by default False
    img_name : str, optional
        The image name. Mandatory if save_to_disk is True.

    Returns
    -------
    go.Figure
        The plotly figure.
    """

    fig = go.Figure()

    colors = {
        "IntvlMeanEnsemble": "#A4CE95",
        "IntvlChoquetEnsemble": "#5F5D9C",
        "IntvlChoquetEnsembleNBest": "#a897ff",
        "IntvlSugenoEnsemble": "#6196A6",
        "CSP-LDA": "#F7DCB9",
    }

    legend_names = {
        "IntvlMeanEnsemble": "Ensemble + Mean",
        "IntvlSugenoEnsemble": "Ensemble + Sugeno",
        "IntvlChoquetEnsemble": "Ensemble + Choquet",
        "IntvlChoquetEnsembleNBest": "Ensemble + Choquet (3 best)",
        "CSP-LDA": "CSP + LDA",
    }

    # For each model in the results dataframe
    for pipeline in legend_names.keys():
        if pipeline in results["pipeline"].values:
            scores = results[results["pipeline"] == pipeline].iloc[0]["scores"]
            w_times = results[results["pipeline"] == pipeline].iloc[0]["w_times"]

            mean_scores = np.Mean(scores, 0)

            # Add score line
            fig.add_trace(
                go.Scatter(
                    x=w_times,
                    y=mean_scores,
                    mode="lines",
                    name=legend_names[pipeline],
                    line_color=colors[pipeline],  # "#074173"
                    line_width=3,
                )
            )

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
