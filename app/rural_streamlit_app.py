from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hoto_rural import config
from hoto_rural.charts import (
    category_bars,
    dual_axis_story,
    food_market_month_chart,
    forecast_fan_chart,
    labour_grouped_bars,
    labour_match_chart,
    simple_line,
    stress_heatmap,
)
from hoto_rural.monte_carlo import build_forecast_bundle
from hoto_rural.schema import SCENARIO_NAMES
from hoto_rural.simulate import run_monthly_simulation, summer_month_summary
from hoto_rural.world import build_water_body, generate_rural_world, load_world, save_world

DATA_DIR = ROOT / "data" / "rural"


@st.cache_resource(show_spinner="Building Hoto Rural world...")
def _world():
    if (DATA_DIR / "people.csv").exists():
        world = load_world(DATA_DIR)
    else:
        world = generate_rural_world()
        save_world(world, DATA_DIR)
    wb = build_water_body()
    wb.to_csv(DATA_DIR / "water_bodies.csv", index=False)
    world["water_bodies"] = wb
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

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hoto Rural")
st.caption(
    f"Regional MVP · population={world['meta']['population']:,} "
    f"· farms={world['meta']['n_farms']} · history {config.HISTORY_MONTHS}m + forecast {config.FORECAST_MONTHS}m"
)

tab_home, tab_loop, tab_market, tab_econ, tab_water, tab_lab = st.tabs(
    ["Overview", "Linked loop", "Markets by month", "Economics", "Freshwater", "Scenario Lab"]
)

with tab_home:
    c1, c2, c3, c4 = st.columns(4)
    people = world["people"]
    c1.metric("People", f"{len(people):,}")
    c2.metric("Farms", f"{len(world['farms']):,}")
    c3.metric(
        "Cows",
        int(world["herds"].loc[world["herds"]["species"].eq("cow"), "n_animals"].sum()),
    )
    c4.metric(
        "Sheep",
        int(world["herds"].loc[world["herds"]["species"].eq("sheep"), "n_animals"].sum()),
    )
    left, right = st.columns(2)
    with left:
        role_counts = people["worker_role"].value_counts()
        st.plotly_chart(
            category_bars(
                role_counts.index.tolist(),
                role_counts.values.tolist(),
                "Employment roles",
                color="#5B21B6",
            ),
            width="stretch",
        , key='rural_1')
    with right:
        farm_counts = world["farms"]["farm_type"].value_counts()
        st.plotly_chart(
            category_bars(
                farm_counts.index.tolist(),
                farm_counts.values.tolist(),
                "Farm types",
                color="#B45309",
            ),
            width="stretch",
        , key='rural_2')
    crop_area = (
        world["crops"][["commodity", "archetype", "area_ha"]]
        .groupby(["archetype", "commodity"], as_index=False)["area_ha"]
        .sum()
        .sort_values("area_ha", ascending=False)
    )
    st.plotly_chart(
        category_bars(
            [f"{r.archetype} / {r.commodity}" for r in crop_area.itertuples()],
            crop_area["area_ha"].tolist(),
            "Crop area (ha)",
            color="#047857",
        ),
        width="stretch",
    , key='rural_3')

with tab_loop:
    st.subheader("Water → feed → manure → profit")
    st.caption(
        "One baseline path showing how reservoir storage, farm profit, and river nutrients move together."
    )
    loop = run_monthly_simulation(
        world, scenario="baseline", n_months=36, seed=config.RANDOM_SEED, history=True
    )
    st.plotly_chart(dual_axis_story(loop, "Linked Rural loop (baseline history)"), width="stretch", key='rural_4')
    a, b, c = st.columns(3)
    a.plotly_chart(
        simple_line(loop, "month", "feed_gap_t", "Feed gap", color="#B45309"),
        width="stretch",
    )
    b.plotly_chart(
        simple_line(loop, "month", "water_stress_index", "Water stress", color="#DC2626"),
        width="stretch",
    )
    c.plotly_chart(
        simple_line(loop, "month", "river_nutrient_mg_l", "River nutrients", color="#047857"),
        width="stretch",
    )
    st.plotly_chart(labour_grouped_bars(loop, "Labour demand follows water/season"), width="stretch", key='rural_5')

with tab_market:
    st.subheader("Month-wise supply and demand")
    st.caption(
        "Household food demand vs local supply, shop stock, import/export, and labour matching. "
        "Yellow bands mark southern-hemisphere summer (Dec–Feb)."
    )
    market = run_monthly_simulation(
        world, scenario="baseline", n_months=36, seed=config.RANDOM_SEED, history=True
    )
    summer = summer_month_summary(market)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Summer irrig (avg m3)", f"{summer['summer_irrig_m3']:,.0f}")
    s2.metric("Other irrig (avg m3)", f"{summer['other_irrig_m3']:,.0f}")
    s3.metric("Summer seasonal labour demand", f"{summer['summer_labour_demand']:,.0f}")
    s4.metric("Other seasonal labour demand", f"{summer['other_labour_demand']:,.0f}")
    st.plotly_chart(
        food_market_month_chart(market, "Food: local supply, demand, import, export"),
        width="stretch",
    , key='rural_6')
    st.plotly_chart(
        labour_match_chart(market, "Labour: seasonal demand vs employed (unmet in red)"),
        width="stretch",
    , key='rural_7')
    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(
            simple_line(market, "month", "retail_stock_t", "Shop / supermarket stock (t)", color="#B45309"),
            width="stretch",
        , key='rural_8')
    with m2:
        st.plotly_chart(
            simple_line(market, "month", "food_price_index", "Food price index", color="#C62828"),
            width="stretch",
        , key='rural_9')
    show_summer_only = st.checkbox("Show summer months only in table", value=False)
    table = market.loc[market["is_summer"] == 1] if show_summer_only else market
    st.dataframe(
        table[
            [
                "month",
                "calendar_month",
                "is_summer",
                "storage_m3",
                "water_stress_index",
                "local_food_supply_t",
                "food_demand_t",
                "food_import_t",
                "food_export_t",
                "retail_stock_t",
                "labour_demand_seasonal",
                "labour_employed_seasonal",
                "labour_unmet_total",
                "farm_profit",
            ]
        ],
        width="stretch",
    )

with tab_econ:
    baseline = run_monthly_simulation(
        world, scenario="baseline", n_months=36, seed=config.RANDOM_SEED, history=True
    )
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(
            simple_line(baseline, "month", "farm_profit", "Farm profit", color="#B45309"),
            width="stretch",
        , key='rural_10')
        st.plotly_chart(
            simple_line(baseline, "month", "food_price_index", "Food price index", color="#C62828"),
            width="stretch",
        , key='rural_11')
    with e2:
        st.plotly_chart(
            labour_grouped_bars(baseline, "Seasonal vs immigrant labour"),
            width="stretch",
        , key='rural_12')
        st.plotly_chart(
            simple_line(baseline, "month", "crop_supply_t", "Crop supply (t)", color="#047857"),
            width="stretch",
        , key='rural_13')
    st.dataframe(
        baseline[
            [
                "month",
                "farm_revenue",
                "farm_opex",
                "farm_profit",
                "subsidy_need",
                "crop_supply_t",
                "feed_gap_t",
            ]
        ],
        width="stretch",
    )

with tab_water:
    drought = run_monthly_simulation(
        world, scenario="drought", n_months=36, seed=config.RANDOM_SEED + 3, history=True
    )
    w1, w2 = st.columns(2)
    with w1:
        st.plotly_chart(
            simple_line(drought, "month", "storage_m3", "Reservoir storage (drought path)"),
            width="stretch",
        , key='rural_14')
        st.plotly_chart(
            simple_line(
                drought, "month", "river_nutrient_mg_l", "River nutrient concentration", color="#047857"
            ),
            width="stretch",
        , key='rural_15')
    with w2:
        st.plotly_chart(
            simple_line(
                drought, "month", "water_stress_index", "Water stress index", color="#DC2626"
            ),
            width="stretch",
        , key='rural_16')
        st.plotly_chart(
            simple_line(drought, "month", "rainfall_mm", "Rainfall (mm)", color="#0284C7"),
            width="stretch",
        , key='rural_17')
    st.dataframe(
        drought[
            [
                "month",
                "rainfall_mm",
                "storage_m3",
                "storage_ratio",
                "irrigation_use_m3",
                "manure_kg",
                "river_nutrient_mg_l",
                "nutrient_exceedance",
            ]
        ],
        width="stretch",
    )

with tab_lab:
    left, right = st.columns([1, 2.4])
    with left:
        st.markdown("#### Scenario controls")
        scenario = st.selectbox("Scenario", options=list(SCENARIO_NAMES), index=1)
        value_col = st.selectbox(
            "Forecast variable",
            options=["storage_m3", "farm_profit", "crop_supply_t", "water_stress_index"],
            index=0,
        )
        n_paths = st.slider("Monte Carlo paths", 20, 120, 60, 10)
        run = st.button("Run Scenario Lab", type="primary")
        st.markdown(
            """
            **Story**
            1. History runs on baseline through month 36  
            2. Futures continue from ending storage  
            3. Grey paths = simulated futures  
            4. Red = pointwise median  
            """
        )
    key = (scenario, n_paths, value_col)
    if run or st.session_state.get("rural_key") != key:
        st.session_state["rural_bundle"] = _bundle(scenario, n_paths, value_col)
        st.session_state["rural_key"] = key
    bundle = st.session_state["rural_bundle"]
    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(storage < 25%)", f"{bundle['metrics']['prob_storage_below_25pct']:.0%}")
        m2.metric(
            "Nutrient exceedance",
            f"{bundle['metrics']['mean_path_nutrient_exceedance']:.2f}",
        )
        m3.metric("Median final profit", f"{bundle['metrics']['median_final_profit']:,.0f}")
        m4.metric("Start storage (m3)", f"{bundle['metrics']['start_storage_m3']:,.0f}")
        st.plotly_chart(
            forecast_fan_chart(
                bundle["history"],
                bundle["paths"],
                bundle["summary"],
                value_col=value_col,
                title=f"{scenario} · {value_col.replace('_', ' ')}",
                now_month=bundle["now_month"],
            ),
            width="stretch",
        , key='rural_18')
        st.caption(
            "Black = history · grey = futures · red = median · band = P10–P90. "
            "Forecast continues from history ending storage (no reset)."
        )

    st.markdown("#### Linked outcomes under this scenario")
    median_path = (
        bundle["paths"]
        .groupby("month", as_index=False)
        .agg(
            storage_m3=("storage_m3", "median"),
            farm_profit=("farm_profit", "median"),
            river_nutrient_mg_l=("river_nutrient_mg_l", "median"),
            labour_demand_seasonal=("labour_demand_seasonal", "median"),
            labour_demand_immigrant=("labour_demand_immigrant", "median"),
            water_stress_index=("water_stress_index", "median"),
            feed_gap_t=("feed_gap_t", "median"),
        )
    )
    l1, l2 = st.columns(2)
    with l1:
        st.plotly_chart(
            dual_axis_story(median_path, "Median future · storage / profit / nutrients"),
            width="stretch",
        , key='rural_19')
        st.plotly_chart(
            labour_grouped_bars(median_path, "Median future labour demand"),
            width="stretch",
        , key='rural_20')
    with l2:
        st.plotly_chart(
            stress_heatmap(bundle["paths"], "Water stress across Monte Carlo paths"),
            width="stretch",
        , key='rural_21')
        st.plotly_chart(
            simple_line(median_path, "month", "feed_gap_t", "Median feed gap", color="#B45309"),
            width="stretch",
        , key='rural_22')

    st.subheader("Contract exports (for later Capital/Metro coupling)")
    st.dataframe(bundle["exports"], width="stretch")
