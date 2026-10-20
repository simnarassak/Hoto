from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hoto_rural import config
from hoto_rural.charts import forecast_fan_chart, simple_line
from hoto_rural.monte_carlo import build_forecast_bundle
from hoto_rural.schema import SCENARIO_NAMES
from hoto_rural.simulate import run_monthly_simulation
from hoto_rural.world import generate_rural_world, load_world, save_world

DATA_DIR = ROOT / "data" / "rural"


@st.cache_resource(show_spinner="Building Hoto Rural world...")
def _world():
    if (DATA_DIR / "people.csv").exists():
        return load_world(DATA_DIR)
    world = generate_rural_world()
    save_world(world, DATA_DIR)
    return world


@st.cache_data(show_spinner="Running Monte Carlo...")
def _bundle(scenario, n_paths, value_col):
    return build_forecast_bundle(
        _world(),
        scenario=scenario,
        value_col=value_col,
        n_paths=n_paths,
        seed=config.RANDOM_SEED,
    )


st.set_page_config(page_title="Hoto Rural", layout="wide")
world = _world()

st.title("Hoto Rural")
st.caption(
    f"Independent regional MVP · population={world['meta']['population']:,} "
    f"(matches island Rural share) · farms={world['meta']['n_farms']}"
)

tab_home, tab_econ, tab_water, tab_lab = st.tabs(
    ["Overview", "Economics", "Freshwater", "Scenario Lab"]
)

with tab_home:
    c1, c2, c3, c4 = st.columns(4)
    people = world["people"]
    c1.metric("People", f"{len(people):,}")
    c2.metric("Farms", f"{len(world['farms']):,}")
    c3.metric("Cows", int(world["herds"].loc[world["herds"]["species"].eq("cow"), "n_animals"].sum()))
    c4.metric("Sheep", int(world["herds"].loc[world["herds"]["species"].eq("sheep"), "n_animals"].sum()))
    st.subheader("Employment roles")
    st.dataframe(people["worker_role"].value_counts().rename_axis("role").reset_index(name="people"), width="stretch")
    st.subheader("Farm types")
    st.dataframe(world["farms"]["farm_type"].value_counts().rename_axis("farm_type").reset_index(name="farms"), width="stretch")
    st.subheader("Crops")
    st.dataframe(world["crops"][["commodity", "archetype", "area_ha"]].groupby(["archetype", "commodity"], as_index=False)["area_ha"].sum(), width="stretch")

with tab_econ:
    baseline = run_monthly_simulation(world, scenario="baseline", n_months=24, seed=config.RANDOM_SEED)
    st.plotly_chart(simple_line(baseline, "month", "farm_profit", "Farm profit (baseline)"), width="stretch")
    st.plotly_chart(simple_line(baseline, "month", "food_price_index", "Food price index"), width="stretch")
    st.plotly_chart(simple_line(baseline, "month", "labour_demand_seasonal", "Seasonal labour demand"), width="stretch")
    st.dataframe(baseline[["month", "farm_revenue", "farm_opex", "farm_profit", "subsidy_need", "crop_supply_t"]], width="stretch")

with tab_water:
    drought = run_monthly_simulation(world, scenario="drought", n_months=24, seed=config.RANDOM_SEED + 3)
    st.plotly_chart(simple_line(drought, "month", "storage_m3", "Reservoir storage under drought"), width="stretch")
    st.plotly_chart(simple_line(drought, "month", "river_nutrient_mg_l", "River nutrient concentration"), width="stretch")
    st.plotly_chart(simple_line(drought, "month", "water_stress_index", "Water stress index"), width="stretch")
    st.dataframe(
        drought[
            [
                "month",
                "rainfall_mm",
                "storage_m3",
                "irrigation_use_m3",
                "manure_kg",
                "river_nutrient_mg_l",
                "nutrient_exceedance",
            ]
        ],
        width="stretch",
    )

with tab_lab:
    left, right = st.columns([1, 2])
    with left:
        scenario = st.selectbox("Scenario", options=list(SCENARIO_NAMES), index=1)
        value_col = st.selectbox(
            "Forecast variable",
            options=["storage_m3", "farm_profit", "crop_supply_t", "water_stress_index"],
            index=0,
        )
        n_paths = st.slider("Monte Carlo paths", 20, 120, 60, 10)
        run = st.button("Run Scenario Lab", type="primary")
    if run or "rural_bundle" not in st.session_state:
        st.session_state["rural_bundle"] = _bundle(scenario, n_paths, value_col)
        st.session_state["rural_value_col"] = value_col
        st.session_state["rural_scenario"] = scenario
    bundle = st.session_state["rural_bundle"]
    value_col = st.session_state.get("rural_value_col", value_col)
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("P(storage < 25% capacity)", f"{bundle['metrics']['prob_storage_below_25pct']:.0%}")
        m2.metric("Nutrient exceedance (path mean)", f"{bundle['metrics']['mean_path_nutrient_exceedance']:.2f}")
        m3.metric("Median final farm profit", f"{bundle['metrics']['median_final_profit']:,.0f}")
        st.plotly_chart(
            forecast_fan_chart(
                bundle["history"],
                bundle["paths"],
                bundle["summary"],
                value_col=value_col,
                title=f"{st.session_state.get('rural_scenario', scenario)} · {value_col} (black history, grey futures, red median)",
            ),
            width="stretch",
        )
        st.caption("Red line is the pointwise median of simulated futures, not a single most-likely trajectory.")
        st.subheader("Contract exports (for later Capital/Metro coupling)")
        st.dataframe(bundle["exports"], width="stretch")
