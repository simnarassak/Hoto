"""Verify demand cost pressure is demand x amount, not capacity-served spend."""

from hoto.simulation import (
    load_sim_inputs,
    regional_impact_summary,
    run_baseline_and_scenario,
)
from hoto.simulation_charts import _impact_kpis, region_impact_cards_html
from hoto.simulation_config import default_config


def check(name, **cfg):
    claims, providers = load_sim_inputs("data")
    result = run_baseline_and_scenario(
        claims, providers, config=default_config(**cfg)
    )
    comparison = result["comparison"]
    summary = regional_impact_summary(comparison)
    kpis = _impact_kpis(summary)

    # Manual rebuild of demand cost pressure from month-region rows.
    manual = (
        comparison["scenario_claims_demand"]
        * comparison["scenario_mean_claim_amount"]
        - comparison["baseline_claims_demand"]
        * comparison["baseline_mean_claim_amount"]
    ).sum()
    from_summary = summary["abs_change_demand_cost"].sum()
    served = summary["abs_change_served_expenditure"].sum()

    print(f"=== {name} ===")
    print(f"manual demand cost pressure: {manual:,.2f}")
    print(f"summary abs_change_demand_cost: {from_summary:,.2f}")
    print(f"KPI demand_cost_change: {kpis['demand_cost_change']:,.2f}")
    print(f"served expenditure change: {served:,.2f}")
    print(
        summary[
            [
                "region",
                "abs_change_claims_demand",
                "abs_change_demand_cost",
                "abs_change_served_expenditure",
            ]
        ].to_string(index=False)
    )

    assert abs(manual - from_summary) < 1e-6
    assert abs(manual - kpis["demand_cost_change"]) < 1e-6

    html = region_impact_cards_html(summary)
    assert "Demand cost pressure" in html
    assert f"{kpis['demand_cost_change']:+,.0f}" in html

    # For capacity-cutting hazards, served spend can diverge from demand cost.
    if cfg.get("hazard_type") == "Tsunami" and cfg.get("capacity_mode") == "Very tight":
        coastal = summary.loc[summary["region"].eq("Coastal")].iloc[0]
        assert coastal["abs_change_demand_cost"] > 0
        assert coastal["abs_change_served_expenditure"] != coastal["abs_change_demand_cost"]
        print(
            "Tsunami Coastal: demand cost > 0 and differs from served spend: OK"
        )

    print("PASS")
    print()


if __name__ == "__main__":
    check(
        "Economic stress Industrial",
        hazard_type="Economic stress",
        shocked_region="Industrial",
        stress_shock_pct=10,
        stress_to_demand_beta=0.25,
        capacity_mode="Normal",
    )
    check(
        "All-region stress",
        hazard_type="All-region stress",
        stress_shock_pct=15,
        stress_to_demand_beta=0.25,
        capacity_mode="Normal",
    )
    check(
        "Tsunami Coastal Very tight",
        hazard_type="Tsunami",
        shocked_region="Coastal",
        capacity_mode="Very tight",
    )
    print("DEMAND_COST_VERIFIED")
