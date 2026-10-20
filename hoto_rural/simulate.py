import numpy as np
import pandas as pd

from hoto_rural import config
from hoto_rural.schema import REGION_ID, SPECIES


def _seasonal_rainfall(month_index, base_mm, rainfall_factor, noise_scale, rng):
    season = 1.0 + 0.28 * np.sin(2 * np.pi * ((month_index % 12) - 2) / 12.0)
    noise = float(rng.normal(0.0, noise_scale))
    return max(4.0, base_mm * season * rainfall_factor * (1.0 + noise))


def _crop_price(commodity, food_price_index):
    return float(config.COMMODITY_PRICE.get(commodity, 1000.0) * food_price_index)


def run_monthly_simulation(
    world,
    scenario="baseline",
    n_months=None,
    seed=None,
    history=False,
    start_storage=None,
    start_food_price_index=None,
    month_offset=0,
):
    n_months = config.HISTORY_MONTHS if history else (n_months or config.FORECAST_MONTHS)
    seed = config.RANDOM_SEED if seed is None else seed
    params = config.SCENARIO_PARAMS[scenario]
    rng = np.random.default_rng(seed)

    farms = world["farms"]
    herds = world["herds"]
    crops = world["crops"]
    people = world["people"]
    water0 = world["water_bodies"].iloc[0]
    capacity = float(water0["capacity_m3"])
    env_flow = float(water0["environmental_flow_m3_month"])
    storage = (
        float(water0["initial_storage_m3"])
        if start_storage is None
        else float(start_storage)
    )
    storage = float(np.clip(storage, 0.0, capacity))

    n_perm = int((people["worker_role"] == "permanent_farm").sum())
    n_seasonal = int((people["worker_role"] == "seasonal_farm").sum())
    n_immigrant = int((people["worker_role"] == "immigrant_farm").sum())

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
    household_water = float(world["meta"]["population"] * config.HOUSEHOLD_WATER_M3_PER_PERSON)

    rows = []
    food_price_index = 1.0 if start_food_price_index is None else float(start_food_price_index)
    for m in range(n_months):
        abs_month = month_offset + m
        rainfall = _seasonal_rainfall(
            abs_month,
            config.BASE_RAINFALL_MM,
            params["rainfall_factor"],
            params["rainfall_noise"],
            rng,
        )
        inflow = rainfall * 0.001 * config.CATCHMENT_AREA_M2 * config.RUNOFF_COEFFICIENT
        storage = min(capacity, storage + inflow)

        efficiency = float(params["irrigation_efficiency"])
        month_of_year = abs_month % 12
        irrig_season = 1.0 + 0.32 * np.sin(2 * np.pi * (month_of_year - 5) / 12.0)
        irrig_need = (crop_water_demand * irrig_season) / max(efficiency, 0.4)
        available = max(0.0, storage - env_flow)

        if params["allocation_priority"] == "essential_first":
            hh_alloc = min(household_water, available)
            live_alloc = min(livestock_water_m3, max(0.0, available - hh_alloc))
            irrig_alloc = min(
                irrig_need, max(0.0, available - hh_alloc - live_alloc)
            )
        else:
            total_need = irrig_need + household_water + livestock_water_m3
            scale = 1.0 if total_need <= 0 else min(1.0, available / total_need)
            irrig_alloc = irrig_need * scale
            hh_alloc = household_water * scale
            live_alloc = livestock_water_m3 * scale

        used = irrig_alloc + hh_alloc + live_alloc + min(env_flow, storage)
        storage = max(0.0, storage - used)
        unmet = max(
            0.0,
            irrig_need + household_water + livestock_water_m3 - (irrig_alloc + hh_alloc + live_alloc),
        )
        stress = min(
            1.0,
            unmet / max(1.0, irrig_need + household_water + livestock_water_m3),
        )
        storage_ratio = storage / max(capacity, 1.0)
        stress = min(1.0, max(stress, 1.0 - storage_ratio) * 0.55 + stress * 0.45)

        water_factor = 1.0 if irrig_need <= 0 else irrig_alloc / irrig_need
        monthly_crop_t = float(
            (crops["area_ha"] * crops["yield_t_ha_year"] / 12.0 * water_factor).sum()
        )
        forage_t = float(
            crops.loc[crops["commodity"].isin(["forage", "paddy_feed"]), "area_ha"].sum()
            * 0.55
            * water_factor
            * (0.85 + 0.15 * np.sin(2 * np.pi * (abs_month % 12) / 12.0))
        )
        feed_need_t = 0.0
        if len(herds):
            feed_need_t = float(
                (herds["n_animals"] * herds["feed_need_kg_month_per_animal"]).sum()
                / 1000.0
            )
        feed_gap = max(0.0, feed_need_t - forage_t)
        herd_factor = (
            1.0
            if feed_need_t <= 0
            else max(0.45, 1.0 - 0.65 * (feed_gap / feed_need_t))
        )

        manure_kg = manure_per_month * herd_factor
        runoff_share = float(params["manure_runoff_base"]) + 0.35 * stress
        nutrient_load = manure_kg * 0.0065 * runoff_share
        river_flow = max(35_000.0, inflow * 0.15 + env_flow * 0.5)
        nutrient_mg_l = nutrient_load * 1000.0 / river_flow
        exceed = int(nutrient_mg_l > config.NUTRIENT_LIMIT_MG_L)

        food_price_index = float(
            np.clip(
                food_price_index * (1.0 + 0.18 * stress - 0.03 * water_factor),
                0.75,
                2.1,
            )
        )
        crop_revenue = 0.0
        for _, crop in crops.iterrows():
            tons = float(crop["area_ha"] * crop["yield_t_ha_year"] / 12.0 * water_factor)
            crop_revenue += tons * _crop_price(crop["commodity"], food_price_index)

        farm_revenue = crop_revenue + livestock_revenue * herd_factor
        opex = (
            float(farms["area_ha"].sum()) * 52.0
            + livestock_water_m3 * 0.55
            + irrig_alloc * 0.32
            + feed_gap * 260.0
            + exceed * 18000.0
            + stress * 25000.0
        )
        subsidy = farm_revenue * float(params["subsidy_rate"]) * (0.4 + stress)
        profit = farm_revenue + subsidy - opex
        subsidy_need = max(0.0, opex - farm_revenue)

        labour_scale = 0.65 + 0.7 * water_factor
        peak = 1.35 if (abs_month % 12) in {10, 11, 0, 1} else 0.75
        seasonal_demand = n_seasonal * labour_scale * peak
        immigrant_demand = n_immigrant * labour_scale * (1.2 if stress < 0.4 else 0.65)

        rows.append(
            {
                "region_id": REGION_ID,
                "month": abs_month + 1,
                "rainfall_mm": rainfall,
                "inflow_m3": inflow,
                "storage_m3": storage,
                "storage_ratio": storage_ratio,
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
                "feed_gap_t": feed_gap,
                "herd_factor": herd_factor,
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
