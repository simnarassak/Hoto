"""Plotly visuals for the Hoto simulation sandbox."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _empty(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def _impact_kpis(summary):
    if summary is None or summary.empty:
        return {
            "top_region": "-",
            "top_demand_abs": 0.0,
            "top_demand_pct": 0.0,
            "demand_cost_change": 0.0,
            "served_expenditure_change": 0.0,
            "budget_overrun": 0.0,
            "regions_over_budget": 0,
            "backlog_change": 0.0,
            "regions_affected": 0,
            "regions_total": 0,
        }

    frame = summary.copy()
    top = frame.loc[frame["abs_change_claims_demand"].abs().idxmax()]
    affected = int((frame["abs_change_claims_demand"].abs() > 1e-9).sum())
    demand_cost_col = (
        "abs_change_demand_cost"
        if "abs_change_demand_cost" in frame.columns
        else "abs_change_expenditure"
    )
    served_col = (
        "abs_change_served_expenditure"
        if "abs_change_served_expenditure" in frame.columns
        else "abs_change_expenditure"
    )
    budget_overrun = (
        float(frame["scenario_budget_gap"].sum())
        if "scenario_budget_gap" in frame.columns
        else 0.0
    )
    if "over_budget" in frame.columns:
        regions_over_budget = int(frame["over_budget"].sum())
    elif "scenario_budget_gap" in frame.columns:
        regions_over_budget = int((frame["scenario_budget_gap"] > 1e-6).sum())
    else:
        regions_over_budget = 0
    return {
        "top_region": str(top["region"]),
        "top_demand_abs": float(top["abs_change_claims_demand"]),
        "top_demand_pct": float(top["mean_pct_change_demand"]),
        "demand_cost_change": float(frame[demand_cost_col].sum()),
        "served_expenditure_change": float(frame[served_col].sum()),
        "budget_overrun": budget_overrun,
        "regions_over_budget": regions_over_budget,
        "backlog_change": float(frame["abs_change_backlog"].sum()),
        "regions_affected": affected,
        "regions_total": int(len(frame)),
    }

def region_impact_cards_html(summary, value_col="mean_pct_change_demand"):
    """Clean KPI card HTML styled like a results summary card."""
    del value_col
    kpis = _impact_kpis(summary)
    demand_color = "#16a34a" if kpis["top_demand_abs"] >= 0 else "#2563eb"
    cost_color = "#16a34a" if kpis["demand_cost_change"] >= 0 else "#2563eb"
    budget_color = "#b45309" if kpis["budget_overrun"] > 0 else "#111827"
    return f"""
<div style="
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 22px 26px 26px 26px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  font-family: Segoe UI, Arial, sans-serif;
">
  <div style="color:#6b7280; font-size:14px; margin-bottom:18px;">
    Hoto · Regional impact (scenario vs baseline)
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:26px 48px;">
    <div>
      <div style="color:#4b5563; font-size:13px; margin-bottom:4px;">Top impacted region</div>
      <div style="color:{demand_color}; font-size:30px; font-weight:700; line-height:1.15;">
        {kpis["top_region"]}
      </div>
      <div style="color:{demand_color}; font-size:16px; font-weight:600; margin-top:2px;">
        {kpis["top_demand_abs"]:+,.1f} claims ({kpis["top_demand_pct"]:+.2f}%)
      </div>
    </div>
    <div>
      <div style="color:#4b5563; font-size:13px; margin-bottom:4px;">Regions affected</div>
      <div style="color:#111827; font-size:30px; font-weight:700; line-height:1.15;">
        {kpis["regions_affected"]} of {kpis["regions_total"]}
      </div>
    </div>
    <div>
      <div style="color:#4b5563; font-size:13px; margin-bottom:4px;">Demand cost pressure</div>
      <div style="color:{cost_color}; font-size:30px; font-weight:700; line-height:1.15;">
        {kpis["demand_cost_change"]:+,.0f}
      </div>
      <div style="color:#6b7280; font-size:12px; margin-top:2px;">
        Based on demand x average claim amount
      </div>
    </div>
    <div>
      <div style="color:#4b5563; font-size:13px; margin-bottom:4px;">Budget overrun</div>
      <div style="color:{budget_color}; font-size:30px; font-weight:700; line-height:1.15;">
        {kpis["budget_overrun"]:+,.0f}
      </div>
      <div style="color:#6b7280; font-size:12px; margin-top:2px;">
        Demand cost vs population-share budget
        ({kpis["regions_over_budget"]} regions over)
      </div>
    </div>
    <div>
      <div style="color:#4b5563; font-size:13px; margin-bottom:4px;">Served expenditure change</div>
      <div style="color:#111827; font-size:30px; font-weight:700; line-height:1.15;">
        {kpis["served_expenditure_change"]:+,.0f}
      </div>
      <div style="color:#6b7280; font-size:12px; margin-top:2px;">
        After capacity constraints
      </div>
    </div>
  </div>
</div>
"""

def region_impact_rank_bars(summary, value_col="abs_change_claims_demand"):
    """Ranked bars by absolute demand change (percent shown as labels)."""
    if summary is None or summary.empty:
        return _empty("No summary")

    if value_col not in summary.columns:
        value_col = "abs_change_claims_demand"

    frame = summary.sort_values(value_col, ascending=True).copy()
    colors = ["#16a34a" if v >= 0 else "#2563eb" for v in frame[value_col]]
    labels = [
        f"{abs_v:+,.1f} ({pct:+.1f}%)"
        for abs_v, pct in zip(
            frame[value_col], frame["mean_pct_change_demand"]
        )
    ]
    fig = go.Figure(
        go.Bar(
            x=frame[value_col],
            y=frame["region"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=labels,
            textposition="outside",
            customdata=np.column_stack([
                frame["mean_pct_change_demand"],
                frame.get(
                    "abs_change_demand_cost",
                    frame.get("abs_change_expenditure", 0),
                ),
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Demand change: %{x:+,.1f} claims (%{customdata[0]:+.2f}%)<br>"
                "Demand cost pressure: %{customdata[1]:+,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Demand change by region (absolute claims)",
            font=dict(size=14, color="#6b7280"),
        ),
        height=300,
        margin=dict(l=20, r=70, t=40, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Extra claims demand (scenario - baseline)",
            zeroline=True,
            zerolinecolor="#d1d5db",
            gridcolor="#f3f4f6",
        ),
        yaxis_title="",
        font=dict(family="Segoe UI, Arial", color="#111827"),
    )
    return fig

def region_impact_map(summary, value_col="mean_pct_change_demand"):
    """
    Combined Plotly view for Gradio / smoke tests:
    KPI-style annotations + ranked bars.
    """
    if summary is None or summary.empty:
        return _empty("No simulation results")

    del value_col
    kpis = _impact_kpis(summary)
    demand_color = "#16a34a" if kpis["top_demand_abs"] >= 0 else "#2563eb"
    cost_color = "#16a34a" if kpis["demand_cost_change"] >= 0 else "#2563eb"

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=0,
        y0=0.52,
        x1=1,
        y1=1,
        xref="paper",
        yref="paper",
        fillcolor="#ffffff",
        line=dict(color="#e5e7eb", width=1),
    )

    fig.add_annotation(
        x=0.03,
        y=0.95,
        xref="paper",
        yref="paper",
        text="Hoto · Regional impact (scenario vs baseline)",
        showarrow=False,
        font=dict(size=13, color="#6b7280"),
        xanchor="left",
    )

    cards = [
        (
            0.03,
            0.82,
            "Top impacted region",
            (
                f"{kpis['top_region']}<br>"
                f"<span style='font-size:16px'>"
                f"{kpis['top_demand_abs']:+,.1f} ({kpis['top_demand_pct']:+.2f}%)"
                f"</span>"
            ),
            demand_color,
        ),
        (
            0.52,
            0.82,
            "Regions affected",
            f"{kpis['regions_affected']} of {kpis['regions_total']}",
            "#111827",
        ),
        (
            0.03,
            0.62,
            "Demand cost pressure",
            f"{kpis['demand_cost_change']:+,.0f}",
            cost_color,
        ),
        (
            0.52,
            0.62,
            "Served expenditure change",
            f"{kpis['served_expenditure_change']:+,.0f}",
            "#111827",
        ),
    ]
    for x, y, label, value, color in cards:
        fig.add_annotation(
            x=x,
            y=y + 0.08,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=12, color="#4b5563"),
            xanchor="left",
        )
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>{value}</b>",
            showarrow=False,
            font=dict(size=26, color=color),
            xanchor="left",
            align="left",
        )

    frame = summary.sort_values("abs_change_claims_demand", ascending=True)
    colors = [
        "#16a34a" if v >= 0 else "#2563eb"
        for v in frame["abs_change_claims_demand"]
    ]
    fig.add_trace(
        go.Bar(
            x=frame["abs_change_claims_demand"],
            y=frame["region"],
            orientation="h",
            marker=dict(color=colors),
            text=[
                f"{v:+,.1f} ({p:+.1f}%)"
                for v, p in zip(
                    frame["abs_change_claims_demand"],
                    frame["mean_pct_change_demand"],
                )
            ],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Demand change: %{x:+,.1f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=70, t=20, b=20),
        paper_bgcolor="#f9fafb",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            domain=[0.05, 0.95],
            anchor="y",
            title="Extra claims demand",
            zeroline=True,
            zerolinecolor="#d1d5db",
            gridcolor="#f3f4f6",
        ),
        yaxis=dict(domain=[0.0, 0.45], title=""),
        font=dict(family="Segoe UI, Arial", color="#111827"),
    )
    return fig

def series_baseline_vs_scenario(
    baseline,
    scenario,
    region="Industrial",
    metric="claims_demand",
):
    if baseline is None or scenario is None:
        return _empty("No simulation series")

    b = baseline.loc[baseline["region"].eq(region), ["month", metric]].copy()
    s = scenario.loc[scenario["region"].eq(region), ["month", metric]].copy()
    b["path"] = "Baseline"
    s["path"] = "Scenario"
    frame = pd.concat([b, s], ignore_index=True)

    fig = px.line(
        frame,
        x="month",
        y=metric,
        color="path",
        markers=True,
        title=f"{region}: {metric.replace('_', ' ')} over time",
        labels={"month": "Month", metric: metric.replace("_", " ").title()},
        color_discrete_map={"Baseline": "#2F6F8F", "Scenario": "#C45C26"},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=9))
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
    )
    return fig

def expenditure_comparison_bars(summary):
    if summary is None or summary.empty:
        return _empty("No summary")

    cost_col = (
        "abs_change_demand_cost"
        if "abs_change_demand_cost" in summary.columns
        else "abs_change_expenditure"
    )
    frame = summary.sort_values(cost_col, ascending=True).copy()
    colors = ["#16a34a" if v >= 0 else "#2563eb" for v in frame[cost_col]]
    fig = go.Figure(
        go.Bar(
            x=frame[cost_col],
            y=frame["region"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=[f"{v:+,.0f}" for v in frame[cost_col]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Demand cost pressure: %{x:+,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Demand cost pressure by region (demand x average claim amount)",
        height=380,
        margin=dict(l=20, r=50, t=50, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_title="Cost pressure change",
        yaxis_title="",
    )
    return fig

def budget_pressure_bars(summary):
    """Scenario demand cost vs population-share budget by region."""
    if summary is None or summary.empty:
        return _empty("No summary")
    if "scenario_demand_cost" not in summary.columns:
        return _empty("Budget metrics unavailable")

    frame = summary.sort_values("scenario_budget_gap", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Budget",
            x=frame["regional_budget"],
            y=frame["region"],
            orientation="h",
            marker=dict(color="#94a3b8"),
            hovertemplate="<b>%{y}</b><br>Population-share budget: %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Demand cost",
            x=frame["scenario_demand_cost"],
            y=frame["region"],
            orientation="h",
            marker=dict(color="#c2410c"),
            hovertemplate="<b>%{y}</b><br>Scenario demand cost: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Demand cost vs population-share budget",
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
            font=dict(size=16),
        ),
        barmode="group",
        height=420,

        margin=dict(l=16, r=16, t=72, b=56),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Amount over horizon",
            domain=[0.0, 0.78],
            automargin=True,
        ),
        yaxis=dict(title="", automargin=True),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.86,
            x=0.80,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=0,
            font=dict(size=12),
            tracegroupgap=8,
        ),
    )
    return fig

def utilisation_heatmap(comparison):
    if comparison is None or comparison.empty:
        return _empty("No comparison")
    pivot = comparison.pivot(
        index="region", columns="month", values="scenario_provider_utilisation"
    )
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        title="Scenario provider utilisation by month",
        labels=dict(x="Month", y="Region", color="Utilisation"),
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#ffffff",
    )
    return fig
