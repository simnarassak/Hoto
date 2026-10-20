import numpy as np
import pandas as pd

from hoto_rural import config
from hoto_rural.schema import REGION_ID, SPECIES


def _seasonal_rainfall(month_index, base_mm, rainfall_factor, rng):
    season = 1.0 + 0.25 * np.sin(2 * np.pi * (month_index % 12) / 12.0)
    noise = float(rng.normal(0.0, 0.12))
    return max(5.0, base_mm * season * rainfall_factor * (1.0 + noise))


def _crop_price(commodity, food_price_index):
    return float(config.COMMODITY_PRICE.get(commodity, 1000.0) * food_price_index)


def run_monthly_simulation(world, scenario="baseline", n_months=None, seed=None, history=False):
    n_months = config.HISTORY_MONTHS if history else (n_months or config.FORECAST_MONTHS)
    seed = config.RANDOM_SEED if seed is None else seed
    params = config.SCENARIO_PARAMS[scenario]
    rng = np.random.default_rng(seed)

    farms = world["farms"]
    herds = world["herds"]
    crops = world["crops"]
    people = world["people"]
    water0 = world["water_bodies"].iloc[0]
    storage = float(water0["initial_storage_m3"])
    capacity = float(water0["capacity_m3"])
    env_flow = float(water0["environmental_flow_m3_month"])

    n_perm = int((people["worker_role"] == "permanent_farm").sum())
    n_seasonal = int((people["worker_role"] == "seasonal_farm").sum())
    n_immigrant = int((people["worker_role"] == "immigrant_farm").sum())

    herd_animals = float(herds["n_animals"].sum()) if len(herds) else 0.0
    manure_per_month = 0.0
    livestock_water_m3 = 0.0
    livestock_revenue = 0.0
    if len(herds):
        for _, h in herds.iterrows():
            manure_per_month += float(h["n_animals"] * h["manure_kg_month_per_animal"])
            spec = SPECIES[h["species"]]
            livestock_water_m3 += float(h["n_animals"] * spec["water_l_day"] * 30.0 / 1000.0)
            livestock_revenue += float(h["n_animals"] * spec["revenue_per_animal_month"])

    crop_water_demand = float((crops["area_ha"] * crops["water_m3_ha_month"]).sum())
    household_water = float(world["meta"]["population"] * 0.15)

    rows = []
    food_price_index = 1.0
    for m in range(n_months):
        rainfall = _seasonal_rainfall(m, config.BASE_RAINFALL_MM, params["rainfall_factor"], rng)
        catchment_area_m2 = 180_000_000.0
        inflow = rainfall * 0.001 * catchment_area_m2 * 0.35
        storage = min(capacity, storage + inflow)

        efficiency = float(params["irrigation_efficiency"])
        irrig_need = crop_water_demand / max(efficiency, 0.4)
        essential = household_water + livestock_water_m3 + env_flow
        available = max(0.0, storage - env_flow)

        if params["allocation_priority"] == "essential_first":
            irrig_alloc = min(irrig_need, max(0.0, available - household_water - livestock_water_m3))
            hh_alloc = min(household_water, available)
            live_alloc = min(livestock_water_m3, max(0.0, available - hh_alloc))
        else:
            total_need = irrig_need + household_water + livestock_water_m3
            scale = 1.0 if total_need <= 0 else min(1.0, available / total_need)
            irrig_alloc = irrig_need * scale
            hh_alloc = household_water * scale
            live_alloc = livestock_water_m3 * scale

        used = irrig_alloc + hh_alloc + live_alloc + env_flow
        storage = max(0.0, storage - used)
        unmet = max(0.0, irrig_need + household_water + livestock_water_m3 - (irrig_alloc + hh_alloc + live_alloc))
        stress = min(1.0, unmet / max(1.0, irrig_need + household_water + livestock_water_m3))

        water_factor = 1.0 if irrig_need <= 0 else irrig_alloc / irrig_need
        monthly_crop_t = float((crops["area_ha"] * crops["yield_t_ha_year"] / 12.0 * water_factor).sum())
        forage_t = float(
            crops.loc[crops["commodity"].isin(["forage", "paddy_feed"]), "area_ha"].sum()
            * 0.7
            * water_factor
        )
        feed_need_t = 0.0
        if len(herds):
            feed_need_t = float((herds["n_animals"] * herds["feed_need_kg_month_per_animal"]).sum() / 1000.0)
        feed_gap = max(0.0, feed_need_t - forage_t)
        herd_factor = 1.0 if feed_need_t <= 0 else max(0.55, 1.0 - 0.5 * (feed_gap / feed_need_t))

        manure_kg = manure_per_month * herd_factor
        runoff_share = 0.12 + 0.25 * stress
        nutrient_load = manure_kg * 0.004 * runoff_share
        river_flow = max(50_000.0, inflow * 0.2 + env_flow)
        nutrient_mg_l = nutrient_load * 1000.0 / river_flow
        exceed = int(nutrient_mg_l > config.NUTRIENT_LIMIT_MG_L)

        food_price_index = float(np.clip(food_price_index * (1.0 + 0.15 * stress - 0.02 * water_factor), 0.8, 1.8))
        crop_revenue = 0.0
        for _, crop in crops.iterrows():
            tons = float(crop["area_ha"] * crop["yield_t_ha_year"] / 12.0 * water_factor)
            crop_revenue += tons * _crop_price(crop["commodity"], food_price_index)

        farm_revenue = crop_revenue + livestock_revenue * herd_factor
        opex = (
            float(farms["area_ha"].sum()) * 45.0
            + livestock_water_m3 * 0.4
            + irrig_alloc * 0.25
            + feed_gap * 220.0
            + exceed * 8000.0
        )
        subsidy = farm_revenue * float(params["subsidy_rate"]) * (0.5 + stress)
        profit = farm_revenue + subsidy - opex
        subsidy_need = max(0.0, opex - farm_revenue)

        labour_scale = 0.7 + 0.6 * water_factor
        seasonal_demand = n_seasonal * labour_scale * (1.2 if (m % 12) in {10, 11, 0, 1} else 0.8)
        immigrant_demand = n_immigrant * labour_scale * (1.15 if stress < 0.35 else 0.75)

        rows.append(
            {
                "region_id": REGION_ID,
                "month": m + 1,
                "rainfall_mm": rainfall,
                "inflow_m3": inflow,
                "storage_m3": storage,
                "irrigation_use_m3": irrig_alloc,
                "household_water_m3": hh_alloc,
                "livestock_water_m3": live_alloc,
                "unmet_water_m3": unmet,
                "water_stress_index": stress,
                "manure_kg": manure_kg,
                "nutrient_load_kg": nutrient_load,
                "river_nutrient_mg_l": nutrient_mg_l,
                "nutrient_exceedance": exceed,
                "crop_supply_t": monthly_crop_t,
                "farm_revenue": farm_revenue,
                "farm_opex": opex,
                "farm_profit": profit,
                "labour_demand_permanent": n_perm,
                "labour_demand_seasonal": seasonal_demand,
                "labour_demand_immigrant": immigrant_demand,
                "food_price_index": food_price_index,
                "subsidy_need": subsidy_need,
                "scenario": scenario,
            }
        )
    return pd.DataFrame(rows)


def contract_exports(monthly):
    out = monthly.copy()
    return pd.DataFrame(
        {
            "region_id": out["region_id"],
            "month": out["month"],
            "crop_supply": out["crop_supply_t"],
            "food_price_index": out["food_price_index"],
            "labour_demand_seasonal": out["labour_demand_seasonal"],
            "immigrant_labour_demand": out["labour_demand_immigrant"],
            "water_stress_index": out["water_stress_index"],
            "nutrient_exceedance": out["nutrient_exceedance"],
            "farm_profit": out["farm_profit"],
            "subsidy_need": out["subsidy_need"],
        }
    )
