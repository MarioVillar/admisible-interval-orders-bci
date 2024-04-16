import pandas as pd
import plotly.graph_objects as go

import config
from utils.disk import check_create_folder

# Import results from disk
results = pd.read_csv(f"{config.DISK_PATH}/results.csv")

# Plot test results
test_results = results[results["session"].str.contains("test")]

fig = go.Figure()

colors = {"IntvlMeanEnsemble": "#A4CE95", "IntvlChoquetEnsemble": "#5F5D9C", "IntvlSugenoEnsemble": "#6196A6"}
legend_names = {
    "IntvlMeanEnsemble": "Aggregation by mean",
    "IntvlChoquetEnsemble": "Aggregation by Choquet Integral",
    "IntvlSugenoEnsemble": "Aggregation by Sugeno Integral",
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

fig.show()

# Save image to disk
if config.SAVE_TO_DISK:
    img_folder = f"{config.DISK_PATH}/images"
    check_create_folder(img_folder)
    fig.write_image(f"{img_folder}/roc_auc_5_fold_BNCI2014_001.svg", format="svg", scale=6)
