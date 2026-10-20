import plotly.graph_objects as go


def forecast_fan_chart(history, paths, summary, value_col="storage_m3", title="Freshwater storage forecast"):
    fig = go.Figure()
    hist_x = history["month"].to_numpy()
    hist_y = history[value_col].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=hist_y,
            mode="lines",
            name="History",
            line=dict(color="#111111", width=3),
        )
    )
    offset = int(history["month"].max())
    sample_ids = sorted(paths["path_id"].unique().tolist())[:40]
    for i, pid in enumerate(sample_ids):
        part = paths.loc[paths["path_id"] == pid]
        fig.add_trace(
            go.Scatter(
                x=part["month"] + offset,
                y=part[value_col],
                mode="lines",
                line=dict(color="#B0B0B0", width=1),
                opacity=0.35,
                showlegend=(i == 0),
                name="Simulated futures",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=summary["month"] + offset,
            y=summary["median"],
            mode="lines",
            name="Median forecast",
            line=dict(color="#C62828", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(summary["month"] + offset) + list((summary["month"] + offset)[::-1]),
            y=list(summary["p90"]) + list(summary["p10"][::-1]),
            fill="toself",
            fillcolor="rgba(198,40,40,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P10–P90",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Month index",
        yaxis_title=value_col.replace("_", " "),
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def simple_line(frame, x, y, title):
    fig = go.Figure(go.Scatter(x=frame[x], y=frame[y], mode="lines", line=dict(color="#2F6F8F", width=3)))
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig
