"""Public Hoto Streamlit demo."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ISLAND_MAP_PATH = ROOT / "images" / "Hoto_ An_Isometric_Island_Map.png"

@st.cache_resource(show_spinner="Preparing Hoto demo artifacts...")
def _boot():
    from hoto.app_data import ensure_app_ready

    return ensure_app_ready(DATA_DIR, MODELS_DIR)

@st.cache_resource(show_spinner="Loading claim models...")
def _claim_models():
    from hoto.infer import load_claim_models

    return load_claim_models(MODELS_DIR)

@st.cache_data(show_spinner=False)
def _load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data(show_spinner="Scoring claims...")
def _live_scores():
    from hoto.app_data import load_live_claim_scores

    scored, _ = load_live_claim_scores(
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        models=_claim_models(),
    )
    return scored

st.set_page_config(page_title="Hoto Claims Risk Demo", layout="wide")
boot = _boot()

tab_home, tab_risk, tab_case, tab_lab, tab_sim = st.tabs(
    ["Home", "Risk lab", "Case explain", "Model lab", "Simulation"]
)

with tab_home:
    st.title("Hoto Claims Risk Demo")

    map_col, copy_col = st.columns([1.15, 1])
    with map_col:
        st.subheader("The country of Hoto")
        if ISLAND_MAP_PATH.exists():
            st.image(
                str(ISLAND_MAP_PATH),
                caption="Illustrative isometric map of synthetic Hoto (static artwork).",
                width=560,
            )
        else:
            st.warning(f"Island map not found at `{ISLAND_MAP_PATH}`.")

    with copy_col:

        st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
This is a **public sandbox** for exploring a fully synthetic country called **Hoto**:
its people, claims activity, providers, money flows, and a risk-triage layer.

Use **Home** to understand the world, **Risk lab** / **Case explain** / **Model lab**
for triage, and **Simulation** to change regional conditions and compare outcomes.
"""
        )
        st.info(
            "A world created using synthetic data for learning; "
            "it showcases risk analysis of insurance claims."
        )
        if boot.get("status") == "built_demo":
            st.caption(
                f"Cold-start demo world "
                f"(population={boot.get('population'):,}, claims={boot.get('n_claims'):,})."
            )

    from hoto import charts

    population = _load_csv("hoto_population.csv")
    claims = _load_csv("hoto_claims_synthetic.csv")
    providers = _load_csv("hoto_providers.csv")

    st.subheader("1. People of Hoto")
    st.caption("Who lives in this synthetic country?")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.population_by_region(population), use_container_width=True)
        st.plotly_chart(charts.employment_mix(population), use_container_width=True)
    with c2:
        st.plotly_chart(charts.age_by_group_heatmap(population), use_container_width=True)

    st.subheader("2. Incidents and claims")
    st.caption("What claim activity occurs over time and across the map of Hoto?")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.claims_over_time(claims), use_container_width=True)
        st.plotly_chart(charts.claims_by_region(claims), use_container_width=True)
    with c4:
        st.plotly_chart(charts.claim_type_mix(claims), use_container_width=True)

    st.subheader("3. Providers")
    st.caption("Who delivers services in each region?")
    st.plotly_chart(
        charts.providers_by_type_region(providers), use_container_width=True
    )

    st.subheader("4. Money decisions")
    st.caption("Approved claim volume and amounts year on year.")
    st.plotly_chart(
        charts.approved_amount_year_on_year(claims), use_container_width=True
    )

with tab_risk:
    from hoto import charts
    from hoto.app_data import apply_threshold_flags, build_investigation_view, load_provider_risk_table

    st.header("Risk lab")
    st.caption(
        "Triage claims with an ensemble score, transparent rules, and provider risk."
    )

    scored = _live_scores()
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.35, 0.01)
    top_n = st.slider("Top N records", 10, 500, 100, 10)
    score_column = st.selectbox(
        "Rank by",
        [
            "ensemble_score",
            "xgboost_probability",
            "logistic_probability",
            "anomaly_score",
            "rule_hit_count",
        ],
        index=0,
    )

    flagged = apply_threshold_flags(scored, threshold=threshold)
    view = build_investigation_view(
        scored,
        score_column=score_column,
        top_n=top_n,
        threshold=threshold,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("In queue view", len(view))
    m2.metric(
        "Ensemble flagged",
        int((flagged["ensemble_flag"] == 1).sum())
        if "ensemble_flag" in flagged.columns
        else 0,
    )
    m3.metric(
        "Any rule hit",
        int((scored["rule_hit_count"] > 0).sum())
        if "rule_hit_count" in scored.columns
        else 0,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.score_histogram(scored, score_column, threshold),
            use_container_width=True,
        )
    with right:
        provider_risk = load_provider_risk_table(
            scored_claims=scored, data_dir=DATA_DIR, models_dir=MODELS_DIR
        )
        st.plotly_chart(
            charts.provider_risk_bubble(provider_risk, top_n=min(40, top_n)),
            use_container_width=True,
        )

    st.subheader("Investigation queue")
    st.dataframe(view, use_container_width=True)

    st.subheader("Provider risk table")
    st.dataframe(provider_risk.head(top_n), use_container_width=True)

with tab_case:
    from hoto.contributions import explain_claim_contributions, force_waterfall_figure
    from hoto.features import CLAIM_MODEL_CATEGORICAL, CLAIM_MODEL_NUMERIC

    st.header("Case explain")
    st.caption("Inspect one claim's timeline and contribution chart.")

    scored = _live_scores()
    claims = _load_csv("hoto_claims_synthetic.csv")
    payments = _load_csv("hoto_payments_synthetic.csv")
    events = _load_csv("hoto_claim_events.csv")

    sample_ids = (
        scored.sort_values("ensemble_score", ascending=False)["claim_id"]
        .head(30)
        .tolist()
    )
    claim_id = st.selectbox(
        "High-risk sample claim",
        options=sample_ids,
        index=0 if sample_ids else None,
    )
    manual_id = st.text_input("Or paste a claim ID", value="")
    if manual_id.strip():
        claim_id = manual_id.strip()

    if claim_id:
        score_row = scored.loc[scored["claim_id"] == claim_id]
        claim_rows = (
            claims.loc[claims["claim_id"] == claim_id]
            if claims is not None
            else pd.DataFrame()
        )
        if score_row.empty and claim_rows.empty:
            st.warning("Claim ID not found.")
        else:
            st.subheader("Scores and rule hits")
            st.dataframe(score_row, use_container_width=True)

            if not claim_rows.empty:
                st.subheader("Claim")
                st.dataframe(claim_rows, use_container_width=True)

            if payments is not None:
                st.subheader("Payments")
                st.dataframe(
                    payments.loc[payments["claim_id"] == claim_id],
                    use_container_width=True,
                )

            if events is not None:
                st.subheader("Event timeline")
                st.dataframe(
                    events.loc[events["claim_id"] == claim_id].sort_values(
                        "event_timestamp"
                    ),
                    use_container_width=True,
                )

            st.subheader("Contribution force chart")
            try:
                models = _claim_models()
                feature_cols = CLAIM_MODEL_CATEGORICAL + CLAIM_MODEL_NUMERIC
                row = scored.loc[scored["claim_id"] == claim_id, feature_cols]
                if len(row) != 1:
                    st.info("Could not align a single feature row.")
                else:
                    explanation = explain_claim_contributions(
                        row.iloc[0],
                        method="Model contributions",
                        xgb_pipeline=models["xgboost"],
                        logistic_pipeline=models["logistic"],
                        top_k=12,
                    )
                    fig = force_waterfall_figure(
                        explanation,
                        title=f"Contributions for {claim_id}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        explanation["contributions"], use_container_width=True
                    )
            except Exception as exc:
                st.warning(f"Contribution chart unavailable: {exc}")

with tab_lab:
    st.header("Model lab")
    st.caption("Held-out comparison metrics and synthetic scenario labels.")

    comparison = _load_csv("model_lab_comparison.csv")
    capacity = _load_csv("model_lab_review_capacity.csv")
    curve_summary = _load_csv("model_lab_curve_summary.csv")
    registry = _load_csv("hoto_scenario_registry.csv")
    labels = _load_csv("hoto_fraud_labels.csv")

    if comparison is not None:
        st.subheader("Classification comparison")
        st.dataframe(comparison, use_container_width=True)
    if capacity is not None:
        st.subheader("Review capacity")
        st.dataframe(capacity, use_container_width=True)
    if curve_summary is not None:
        st.subheader("Curve summary")
        st.dataframe(curve_summary, use_container_width=True)

    st.subheader("Scenarios")
    if registry is not None:
        st.dataframe(registry, use_container_width=True)
    if labels is not None:
        summary = (
            labels.groupby(["entity_type", "scenario_id", "fraud_label"])
            .size()
            .reset_index(name="count")
        )
        st.dataframe(summary, use_container_width=True)

with tab_sim:
    from hoto.simulation import (
        regional_impact_summary,
        run_baseline_and_scenario,
    )
    from hoto.simulation_charts import (
        budget_pressure_bars,
        expenditure_comparison_bars,
        region_impact_cards_html,
        region_impact_rank_bars,
        series_baseline_vs_scenario,
        utilisation_heatmap,
    )
    from hoto.simulation_config import (
        HAZARD_PROFILES,
        REGION_CHOICES,
        capacity_mode_choices,
        default_config,
        hazard_choices,
    )

    st.header("Simulation")
    st.caption(
        "Synthetic policy sandbox: hazards, all-region stress, and capacity modes. "
        "Baseline versus scenario over months. Rules based model — not a ML forecast."
    )

    claims = _load_csv("hoto_claims_synthetic.csv")
    providers = _load_csv("hoto_providers.csv")
    if claims is None or providers is None:
        st.warning("Claims/providers data missing. Run the training pipeline first.")
    else:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Scenario controls")
            hazard_type = st.selectbox("Hazard / scenario", options=hazard_choices(), index=0)
            profile = HAZARD_PROFILES[hazard_type]
            st.caption(profile["description"])

            if profile["allows_region_pick"]:
                shocked_region = st.selectbox(
                    "Focus region",
                    options=list(REGION_CHOICES),
                    index=list(REGION_CHOICES).index("Industrial")
                    if hazard_type == "Economic stress"
                    else list(REGION_CHOICES).index("All profile regions"),
                )
            else:
                shocked_region = "All profile regions"
                st.write("Affected regions: all Hoto regions")

            stress_shock_pct = st.slider(
                "Stress shock (%)",
                0.0,
                80.0,
                float(profile["default_stress_shock_pct"]),
                1.0,
            )
            capacity_mode = st.selectbox(
                "Capacity mode",
                options=capacity_mode_choices(),
                index=0,
                help="Use Tight / Very tight to demo backlog under pressure.",
            )
            horizon = st.slider("Horizon (months)", 3, 12, 6, 1)
            beta = st.slider("Stress-to-demand beta", 0.0, 1.0, 0.25, 0.05)
            prevention = st.slider("Prevention budget change (%)", 0.0, 50.0, 0.0, 1.0)
            metric = st.selectbox(
                "Time-series metric",
                options=[
                    "claims_demand",
                    "claims_served",
                    "backlog",
                    "provider_utilisation",
                    "claims_expenditure",
                ],
                index=2 if capacity_mode != "Normal" else 0,
            )
            chart_region = st.selectbox(
                "Chart region",
                options=["Capital", "Metro", "Rural", "Coastal", "Industrial"],
                index=4,
            )
            run = st.button("Run simulation", type="primary")

        if run or "sim_result" not in st.session_state:
            cfg = default_config(
                hazard_type=hazard_type,
                shocked_region=shocked_region,
                stress_shock_pct=stress_shock_pct,
                horizon_months=horizon,
                stress_to_demand_beta=beta,
                prevention_budget_pct=prevention,
                capacity_mode=capacity_mode,
            )
            st.session_state["sim_result"] = run_baseline_and_scenario(
                claims, providers, config=cfg
            )
            st.session_state["sim_metric"] = metric
            st.session_state["sim_region"] = chart_region

        result = st.session_state["sim_result"]
        view_region = st.session_state["sim_region"]
        view_metric = st.session_state["sim_metric"]
        if run:
            st.session_state["sim_metric"] = metric
            st.session_state["sim_region"] = chart_region
            view_region = chart_region
            view_metric = metric
        summary = regional_impact_summary(result["comparison"])

        with right:
            st.markdown(
                region_impact_cards_html(summary),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                region_impact_rank_bars(summary),
                use_container_width=True,
            )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                series_baseline_vs_scenario(
                    result["baseline"],
                    result["scenario"],
                    region=view_region,
                    metric=view_metric,
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                expenditure_comparison_bars(summary),
                use_container_width=True,
            )

        st.plotly_chart(
            budget_pressure_bars(summary),
            use_container_width=True,
        )
        st.plotly_chart(
            utilisation_heatmap(result["comparison"]),
            use_container_width=True,
        )

        st.subheader("Regional impact summary")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Month-by-region comparison")
        show_cols = [
            c
            for c in result["comparison"].columns
            if c.startswith(("month", "region", "baseline_", "scenario_", "abs_change_", "pct_change_"))
        ]
        st.dataframe(
            result["comparison"][show_cols],
            use_container_width=True,
        )

        st.subheader("Derived monthly baselines")
        st.dataframe(result["baselines"], use_container_width=True)
