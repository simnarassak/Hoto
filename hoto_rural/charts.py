import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "ink": "#1F2937",
    "history": "#111111",
    "future": "#9CA3AF",
    "median": "#C62828",
    "band": "rgba(198,40,40,0.14)",
    "water": "#0284C7",
    "econ": "#B45309",
    "labour": "#5B21B6",
    "nutrient": "#047857",
    "stress": "#DC2626",
    "grid": "#E5E7EB",
}


def _base_layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Arial", color=COLORS["ink"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(255,255,255,0.9)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    return fig


def forecast_fan_chart(
    history,
    paths,
    summary,
    value_col="storage_m3",
    title="Freshwater storage forecast",
    now_month=None,
):
    fig = go.Figure()
    hist_x = history["month"].to_numpy()
    hist_y = history[value_col].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=hist_y,
            mode="lines",
            name="History (observed path)",
            line=dict(color=COLORS["history"], width=3.5),
        )
    )

    sample_ids = sorted(paths["path_id"].unique().tolist())
    step = max(1, len(sample_ids) // 35)
    sample_ids = sample_ids[::step][:35]
    for i, pid in enumerate(sample_ids):
        part = paths.loc[paths["path_id"] == pid].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=part["month"],
                y=part[value_col],
                mode="lines",
                line=dict(color=COLORS["future"], width=1.1),
                opacity=0.45,
                showlegend=(i == 0),
                name="Simulated futures",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=list(summary["month"]) + list(summary["month"][::-1]),
            y=list(summary["p90"]) + list(summary["p10"][::-1]),
            fill="toself",
            fillcolor=COLORS["band"],
            line=dict(color="rgba(0,0,0,0)"),
            name="P10–P90 band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["month"],
            y=summary["median"],
            mode="lines",
            name="Median forecast",
            line=dict(color=COLORS["median"], width=3.5),
        )
    )

    if now_month is None:
        now_month = int(history["month"].max())
    fig.add_vline(
        x=now_month + 0.5,
        line_width=1.5,
        line_dash="dash",
        line_color="#6B7280",
        annotation_text="Forecast start",
        annotation_position="top",
    )
    fig = _base_layout(fig, title, height=460)
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text=value_col.replace("_", " "))
    return fig


def simple_line(frame, x, y, title, color=None):
    color = color or COLORS["water"]
    fig = go.Figure(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="lines",
            line=dict(color=color, width=3),
            fill="tozeroy",
            fillcolor="rgba(2,132,199,0.08)" if color == COLORS["water"] else "rgba(180,83,9,0.08)",
        )
    )
    return _base_layout(fig, title, height=340)


def dual_axis_story(frame, title="Linked Rural loop"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["storage_m3"],
            name="Storage",
            line=dict(color=COLORS["water"], width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["farm_profit"],
            name="Farm profit",
            line=dict(color=COLORS["econ"], width=3),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["river_nutrient_mg_l"],
            name="River nutrients",
            line=dict(color=COLORS["nutrient"], width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig = _base_layout(fig, title, height=400)
    fig.update_yaxes(title_text="Storage (m3)", secondary_y=False)
    fig.update_yaxes(title_text="Profit / nutrients", secondary_y=True)
    return fig


def labour_grouped_bars(frame, title="Labour demand by month"):
    months = frame["month"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Seasonal",
            x=months,
            y=frame["labour_demand_seasonal"],
            marker_color=COLORS["labour"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Immigrant farm",
            x=months,
            y=frame["labour_demand_immigrant"],
            marker_color="#A78BFA",
        )
    )
    fig.update_layout(barmode="group")
    fig = _base_layout(fig, title, height=360)
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Workers demanded")
    return fig


def category_bars(labels, values, title, color=COLORS["econ"]):
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color),
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
        )
    )
    fig = _base_layout(fig, title, height=max(320, 40 * len(labels) + 80))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    return fig


def stress_heatmap(paths, title="Water stress across futures"):
    pivot = paths.pivot_table(
        index="path_id", columns="month", values="water_stress_index", aggfunc="mean"
    )
    sample = pivot.head(min(40, len(pivot)))
    fig = go.Figure(
        go.Heatmap(
            z=sample.to_numpy(),
            x=list(sample.columns),
            y=[f"path {i}" for i in sample.index],
            colorscale="YlOrRd",
            colorbar=dict(title="Stress"),
        )
    )
    return _base_layout(fig, title, height=380)


def rural_demographics_heatmap(people, title="Rural age band by population group"):
    table = pd.crosstab(people["age_band"], people["population_group"])
    age_order = ["0-17", "18-34", "35-49", "50-64", "65+"]
    table = table.reindex([a for a in age_order if a in table.index])
    fig = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=list(table.columns),
            y=list(table.index),
            colorscale="Blues",
            text=table.to_numpy(),
            texttemplate="%{text}",
            colorbar=dict(title="People"),
        )
    )
    return _base_layout(fig, title, height=380)


def rural_labour_roles(people, title="Rural labour roles"):
    counts = people["worker_role"].value_counts()
    return category_bars(
        counts.index.tolist(),
        counts.values.tolist(),
        title,
        color=COLORS["labour"],
    )


def rural_water_snapshot(water_bodies, title="Rural freshwater and river"):
    row = water_bodies.iloc[0]
    labels = [
        "Reservoir capacity",
        "Available storage",
        "River environmental flow / month",
    ]
    values = [
        float(row["capacity_m3"]),
        float(row["initial_storage_m3"]),
        float(row["environmental_flow_m3_month"]),
    ]
    colors = [COLORS["water"], "#38BDF8", COLORS["nutrient"]]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:,.0f} m3" for v in values],
            textposition="outside",
        )
    )
    return _base_layout(fig, title, height=320)


def food_market_month_chart(frame, title="Food supply, demand, import and export"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["local_food_supply_t"],
            name="Local supply",
            line=dict(color=COLORS["nutrient"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["food_demand_t"],
            name="Household demand",
            line=dict(color=COLORS["ink"], width=3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["month"],
            y=frame["food_import_t"],
            name="Import",
            marker_color="#F59E0B",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["month"],
            y=frame["food_export_t"],
            name="Export",
            marker_color="#0284C7",
            opacity=0.75,
        )
    )
    summers = frame.loc[frame["is_summer"] == 1, "month"].tolist()
    for i, month in enumerate(summers):
        fig.add_vrect(
            x0=month - 0.5,
            x1=month + 0.5,
            fillcolor="rgba(251, 191, 36, 0.12)",
            line_width=0,
            annotation_text="summer" if i == 0 else None,
            annotation_position="top left",
        )
    fig.update_layout(barmode="overlay")
    fig = _base_layout(fig, title, height=400)
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Tonnes")
    return fig


def labour_match_chart(frame, title="Labour demand vs employed (seasonal)"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["labour_demand_seasonal"],
            name="Demand",
            line=dict(color=COLORS["labour"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["labour_employed_seasonal"],
            name="Employed",
            line=dict(color="#A78BFA", width=3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["month"],
            y=frame["labour_unmet_seasonal"],
            name="Unmet",
            marker_color=COLORS["stress"],
            opacity=0.55,
        )
    )
    summers = frame.loc[frame["is_summer"] == 1, "month"].tolist()
    for month in summers:
        fig.add_vrect(
            x0=month - 0.5,
            x1=month + 0.5,
            fillcolor="rgba(251, 191, 36, 0.10)",
            line_width=0,
        )
    return _base_layout(fig, title, height=380)


def rural_home_metrics(people, farms, water_bodies):
    water = water_bodies.iloc[0]
    farm_roles = {"permanent_farm", "seasonal_farm", "immigrant_farm"}
    labour_people = int(people["worker_role"].isin(farm_roles).sum())
    return {
        "population": int(len(people)),
        "farms": int(len(farms)),
        "available_water_m3": float(water["initial_storage_m3"]),
        "capacity_m3": float(water["capacity_m3"]),
        "river_flow_m3_month": float(water["environmental_flow_m3_month"]),
        "farm_labour": labour_people,
        "seasonal_farm": int((people["worker_role"] == "seasonal_farm").sum()),
        "immigrant_farm": int((people["worker_role"] == "immigrant_farm").sum()),
        "permanent_farm": int((people["worker_role"] == "permanent_farm").sum()),
    }
