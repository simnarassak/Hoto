"""
Hugging Face Spaces entrypoint (Gradio).

IA: Home | Risk lab | Case explain | Model lab | Simulation
"""

from pathlib import Path
import sys

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ISLAND_MAP_PATH = ROOT / "images" / "Hoto_ An_Isometric_Island_Map.png"

from hoto import charts
from hoto.app_data import (
    apply_threshold_flags,
    build_investigation_view,
    ensure_app_ready,
    load_live_claim_scores,
    load_provider_risk_table,
)
from hoto.contributions import explain_claim_contributions, force_waterfall_figure
from hoto.features import CLAIM_MODEL_CATEGORICAL, CLAIM_MODEL_NUMERIC
from hoto.infer import load_artifact_csv, load_claim_models
from hoto.simulation import (
    regional_impact_summary,
    run_baseline_and_scenario,
)
from hoto.simulation_charts import (
    budget_pressure_bars,
    expenditure_comparison_bars,
    region_impact_cards_html,
    region_impact_map,
    region_impact_rank_bars,
    series_baseline_vs_scenario,
    utilisation_heatmap,
)
from hoto.simulation_config import (
    REGION_CHOICES,
    capacity_mode_choices,
    default_config,
    hazard_choices,
)

_BOOT = ensure_app_ready(DATA_DIR, MODELS_DIR)
_MODELS = load_claim_models(MODELS_DIR)
_SCORED, _ = load_live_claim_scores(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    models=_MODELS,
)
_PROVIDER_RISK = load_provider_risk_table(
    scored_claims=_SCORED, data_dir=DATA_DIR, models_dir=MODELS_DIR
)

_POPULATION = load_artifact_csv("hoto_population.csv", DATA_DIR)
_CLAIMS = load_artifact_csv("hoto_claims_synthetic.csv", DATA_DIR)
_PROVIDERS = load_artifact_csv("hoto_providers.csv", DATA_DIR)
_PAYMENTS = load_artifact_csv("hoto_payments_synthetic.csv", DATA_DIR)
_EVENTS = load_artifact_csv("hoto_claim_events.csv", DATA_DIR)

_SAMPLE_IDS = (
    _SCORED.sort_values("ensemble_score", ascending=False)["claim_id"]
    .head(30)
    .tolist()
)


def refresh_risk(threshold, top_n, score_column):
    threshold = float(threshold)
    top_n = int(top_n)
    flagged = apply_threshold_flags(_SCORED, threshold=threshold)
    view = build_investigation_view(
        _SCORED,
        score_column=score_column,
        top_n=top_n,
        threshold=threshold,
    )
    summary = (
        f"Queue rows: {len(view)} | "
        f"ensemble flagged: {int((flagged['ensemble_flag'] == 1).sum())} | "
        f"any rule hit: {int((_SCORED['rule_hit_count'] > 0).sum())}"
    )
    hist = charts.score_histogram(_SCORED, score_column, threshold)
    bubble = charts.provider_risk_bubble(
        _PROVIDER_RISK, top_n=min(40, top_n)
    )
    return summary, hist, bubble, view, _PROVIDER_RISK.head(top_n)


def explain_case(claim_pick, claim_manual, method):
    claim_id = (
        claim_manual.strip()
        if claim_manual and claim_manual.strip()
        else claim_pick
    )
    if not claim_id:
        return "Enter a claim ID.", None, None, None, None, None, None

    score_row = _SCORED.loc[_SCORED["claim_id"] == claim_id]
    claim_tbl = (
        _CLAIMS.loc[_CLAIMS["claim_id"] == claim_id]
        if _CLAIMS is not None
        else pd.DataFrame()
    )
    pay_tbl = (
        _PAYMENTS.loc[_PAYMENTS["claim_id"] == claim_id]
        if _PAYMENTS is not None
        else pd.DataFrame()
    )
    event_tbl = (
        _EVENTS.loc[_EVENTS["claim_id"] == claim_id].sort_values("event_timestamp")
        if _EVENTS is not None
        else pd.DataFrame()
    )

    if score_row.empty and claim_tbl.empty:
        return "Claim ID not found.", None, None, None, None, None, None

    fig = None
    contrib_tbl = None
    try:
        feature_cols = CLAIM_MODEL_CATEGORICAL + CLAIM_MODEL_NUMERIC
        row = _SCORED.loc[_SCORED["claim_id"] == claim_id, feature_cols]
        if len(row) == 1:
            explanation = explain_claim_contributions(
                row.iloc[0],
                method=method,
                xgb_pipeline=_MODELS["xgboost"],
                logistic_pipeline=_MODELS["logistic"],
                top_k=12,
            )
            fig = force_waterfall_figure(
                explanation,
                title=f"{explanation['method']} for {claim_id}",
            )
            contrib_tbl = explanation["contributions"]
    except Exception as exc:
        contrib_tbl = pd.DataFrame({"error": [str(exc)]})

    return (
        f"Loaded claim {claim_id} ({method})",
        score_row,
        claim_tbl,
        pay_tbl,
        event_tbl,
        fig,
        contrib_tbl,
    )


with gr.Blocks(title="Hoto Claims Risk Demo") as demo:
    gr.Markdown("# Hoto Claims Risk Demo")

    with gr.Tab("Home"):
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("### The country of Hoto")
                if ISLAND_MAP_PATH.exists():
                    gr.Image(
                        value=str(ISLAND_MAP_PATH),
                        label="Illustrative isometric map of synthetic Hoto (static artwork)",
                        height=420,
                    )
                else:
                    gr.Markdown(f"Island map not found at `{ISLAND_MAP_PATH}`.")
            with gr.Column(scale=5):
                gr.HTML("<div style='height: 3.5rem;'></div>")
                gr.Markdown(
                    """
This is a **public sandbox** for exploring a fully synthetic country called **Hoto**:
its people, claims activity, providers, money flows, and a risk-triage layer.

Use **Home** to understand the world, **Risk lab** / **Case explain** / **Model lab**
for triage, and **Simulation** to change regional conditions and compare outcomes.

A world created using synthetic data for learning; it showcases risk analysis of insurance claims.
"""
                )
                if _BOOT.get("status") == "built_demo":
                    gr.Markdown(
                        f"Cold-start demo world "
                        f"(population={_BOOT.get('population'):,}, "
                        f"claims={_BOOT.get('n_claims'):,})."
                    )

        gr.Markdown("### 1. People of Hoto")
        with gr.Row():
            gr.Plot(value=charts.population_by_region(_POPULATION))
            gr.Plot(value=charts.age_by_group_heatmap(_POPULATION))
        gr.Plot(value=charts.employment_mix(_POPULATION))

        gr.Markdown("### 2. Incidents and claims")
        with gr.Row():
            gr.Plot(value=charts.claims_over_time(_CLAIMS))
            gr.Plot(value=charts.claim_type_mix(_CLAIMS))
        gr.Plot(value=charts.claims_by_region(_CLAIMS))

        gr.Markdown("### 3. Providers")
        gr.Plot(value=charts.providers_by_type_region(_PROVIDERS))

        gr.Markdown("### 4. Money decisions")
        gr.Plot(value=charts.approved_amount_year_on_year(_CLAIMS))

    with gr.Tab("Risk lab"):
        with gr.Row():
            threshold = gr.Slider(0, 1, value=0.35, step=0.01, label="Score threshold")
            top_n = gr.Slider(10, 500, value=100, step=10, label="Top N")
            score_column = gr.Dropdown(
                choices=[
                    "ensemble_score",
                    "xgboost_probability",
                    "logistic_probability",
                    "anomaly_score",
                    "rule_hit_count",
                ],
                value="ensemble_score",
                label="Rank by",
            )
        risk_btn = gr.Button("Refresh risk lab")
        risk_summary = gr.Markdown()
        with gr.Row():
            hist_plot = gr.Plot()
            bubble_plot = gr.Plot()
        queue_table = gr.Dataframe(label="Investigation queue")
        provider_table = gr.Dataframe(label="Provider risk")
        risk_btn.click(
            refresh_risk,
            inputs=[threshold, top_n, score_column],
            outputs=[
                risk_summary,
                hist_plot,
                bubble_plot,
                queue_table,
                provider_table,
            ],
        )
        demo.load(
            refresh_risk,
            inputs=[threshold, top_n, score_column],
            outputs=[
                risk_summary,
                hist_plot,
                bubble_plot,
                queue_table,
                provider_table,
            ],
        )

    with gr.Tab("Case explain"):
        with gr.Row():
            claim_pick = gr.Dropdown(
                choices=_SAMPLE_IDS,
                value=_SAMPLE_IDS[0] if _SAMPLE_IDS else None,
                label="High-risk sample claim",
            )
            claim_manual = gr.Textbox(label="Or paste claim ID")
            method = gr.Radio(
                choices=[
                    "Model contributions (SHAP)",
                    "Simple linear contributions",
                ],
                value="Model contributions (SHAP)",
                label="Contribution method",
            )
        case_btn = gr.Button("Explain case")
        case_status = gr.Markdown()
        score_out = gr.Dataframe(label="Scores / rules")
        claim_out = gr.Dataframe(label="Claim")
        pay_out = gr.Dataframe(label="Payments")
        event_out = gr.Dataframe(label="Event timeline")
        force_plot = gr.Plot(label="Custom force-style contributions")
        contrib_out = gr.Dataframe(label="Contribution table")
        case_btn.click(
            explain_case,
            inputs=[claim_pick, claim_manual, method],
            outputs=[
                case_status,
                score_out,
                claim_out,
                pay_out,
                event_out,
                force_plot,
                contrib_out,
            ],
        )

    with gr.Tab("Model lab"):
        comparison = load_artifact_csv("model_lab_comparison.csv", DATA_DIR)
        capacity = load_artifact_csv("model_lab_review_capacity.csv", DATA_DIR)
        registry = load_artifact_csv("hoto_scenario_registry.csv", DATA_DIR)
        gr.Dataframe(value=comparison, label="Classification comparison")
        gr.Dataframe(value=capacity, label="Review capacity")
        gr.Dataframe(value=registry, label="Scenario registry")

    with gr.Tab("Simulation"):
        gr.Markdown(
            "Synthetic policy sandbox: flood, earthquake, pandemic, storm, tsunami, "
            "all-region stress, and tight capacity modes. Baseline versus scenario. "
            "Rules are illustrative — ML does not drive demand."
        )

        def run_sim(
            hazard,
            region,
            shock_pct,
            capacity_mode,
            horizon,
            beta,
            prevention,
            metric,
            chart_region,
        ):
            cfg = default_config(
                hazard_type=hazard,
                shocked_region=region,
                stress_shock_pct=float(shock_pct),
                capacity_mode=capacity_mode,
                horizon_months=int(horizon),
                stress_to_demand_beta=float(beta),
                prevention_budget_pct=float(prevention),
            )
            result = run_baseline_and_scenario(_CLAIMS, _PROVIDERS, config=cfg)
            summary = regional_impact_summary(result["comparison"])
            return (
                region_impact_cards_html(summary),
                region_impact_rank_bars(summary),
                series_baseline_vs_scenario(
                    result["baseline"],
                    result["scenario"],
                    region=chart_region,
                    metric=metric,
                ),
                expenditure_comparison_bars(summary),
                budget_pressure_bars(summary),
                utilisation_heatmap(result["comparison"]),
                summary,
                result["comparison"],
                result["baselines"],
            )

        with gr.Row():
            sim_hazard = gr.Dropdown(
                choices=hazard_choices(),
                value="Economic stress",
                label="Hazard / scenario",
            )
            sim_region = gr.Dropdown(
                choices=list(REGION_CHOICES),
                value="Industrial",
                label="Focus region",
            )
            sim_capacity = gr.Dropdown(
                choices=capacity_mode_choices(),
                value="Normal",
                label="Capacity mode",
            )
        with gr.Row():
            sim_shock = gr.Slider(0, 80, value=10, step=1, label="Stress shock (%)")
            sim_horizon = gr.Slider(3, 12, value=6, step=1, label="Horizon (months)")
            sim_beta = gr.Slider(0, 1, value=0.25, step=0.05, label="Stress-to-demand beta")
        with gr.Row():
            sim_prevention = gr.Slider(
                0, 50, value=0, step=1, label="Prevention budget change (%)"
            )
            sim_metric = gr.Dropdown(
                choices=[
                    "claims_demand",
                    "claims_served",
                    "backlog",
                    "provider_utilisation",
                    "claims_expenditure",
                ],
                value="backlog",
                label="Time-series metric",
            )
            sim_chart_region = gr.Dropdown(
                choices=["Capital", "Metro", "Rural", "Coastal", "Industrial"],
                value="Industrial",
                label="Chart region",
            )
        sim_btn = gr.Button("Run simulation")
        sim_kpi_card = gr.HTML()
        sim_rank = gr.Plot()
        with gr.Row():
            sim_series = gr.Plot()
            sim_bars = gr.Plot()
        sim_budget = gr.Plot()
        sim_heat = gr.Plot()
        sim_summary = gr.Dataframe(label="Regional impact summary")
        sim_compare = gr.Dataframe(label="Month-by-region comparison")
        sim_baselines = gr.Dataframe(label="Derived monthly baselines")

        sim_inputs = [
            sim_hazard,
            sim_region,
            sim_shock,
            sim_capacity,
            sim_horizon,
            sim_beta,
            sim_prevention,
            sim_metric,
            sim_chart_region,
        ]
        sim_outputs = [
            sim_kpi_card,
            sim_rank,
            sim_series,
            sim_bars,
            sim_budget,
            sim_heat,
            sim_summary,
            sim_compare,
            sim_baselines,
        ]
        sim_btn.click(run_sim, inputs=sim_inputs, outputs=sim_outputs)
        demo.load(run_sim, inputs=sim_inputs, outputs=sim_outputs)


if __name__ == "__main__":
    demo.launch()
