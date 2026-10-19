from hoto.simulation import (
    load_sim_inputs,
    regional_impact_summary,
    run_baseline_and_scenario,
)
from hoto.simulation_charts import budget_pressure_bars
from hoto.simulation_config import default_config

claims, providers = load_sim_inputs("data")
result = run_baseline_and_scenario(
    claims,
    providers,
    config=default_config(hazard_type="Tsunami", shocked_region="Coastal"),
)
fig = budget_pressure_bars(regional_impact_summary(result["comparison"]))
out = "scripts/_budget_chart_check.png"
try:
    fig.write_image(out, width=900, height=400, scale=2)
    print("wrote", out)
except Exception as exc:
    fig.write_html("scripts/_budget_chart_check.html", include_plotlyjs="cdn")
    print("png failed:", exc)
    print("wrote scripts/_budget_chart_check.html")
print("title", fig.layout.title)
print("legend", fig.layout.legend)
print("margin", fig.layout.margin)
