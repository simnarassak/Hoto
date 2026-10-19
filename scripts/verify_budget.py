"""Verify population-share regional budgets vs demand cost."""

from hoto.config import REGIONS, WORLD
from hoto.simulation import (
    load_sim_inputs,
    population_budget_shares,
    regional_impact_summary,
    regional_populations,
    run_baseline_and_scenario,
)
from hoto.simulation_charts import _impact_kpis, budget_pressure_bars, region_impact_cards_html
from hoto.simulation_config import default_config


def main():
    claims, providers = load_sim_inputs("data")
    pops = regional_populations()
    shares = population_budget_shares()
    assert abs(pops.sum() - WORLD["population"]) < 1e-6
    assert abs(shares.sum() - 1.0) < 1e-9
    for region, share in REGIONS.items():
        assert abs(shares.loc[region] - share) < 1e-9

    result = run_baseline_and_scenario(
        claims,
        providers,
        config=default_config(
            hazard_type="Tsunami",
            shocked_region="Coastal",
            capacity_mode="Very tight",
        ),
    )
    baselines = result["baselines"]
    assert "monthly_budget" in baselines.columns
    assert abs(baselines["budget_share"].sum() - 1.0) < 1e-9
                                                                                      
    assert abs(
        baselines["monthly_budget"].sum()
        - baselines["baseline_monthly_demand_cost"].sum()
    ) < 1e-6

    summary = regional_impact_summary(result["comparison"])
    coastal = summary.loc[summary["region"].eq("Coastal")].iloc[0]
    assert coastal["scenario_demand_cost"] > coastal["regional_budget"]
    assert bool(coastal["over_budget"]) is True

    kpis = _impact_kpis(summary)
    assert kpis["regions_over_budget"] >= 1
    html = region_impact_cards_html(summary)
    assert "Budget overrun" in html
    assert budget_pressure_bars(summary) is not None

    print("populations:")
    print(pops.round(0).astype(int).to_string())
    print()
    print(
        summary[
            [
                "region",
                "regional_budget",
                "scenario_demand_cost",
                "scenario_budget_gap",
                "over_budget",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nbudget overrun KPI: {kpis['budget_overrun']:+,.0f} "
        f"({kpis['regions_over_budget']} regions over)"
    )
    print("BUDGET_VERIFIED")


if __name__ == "__main__":
    main()
