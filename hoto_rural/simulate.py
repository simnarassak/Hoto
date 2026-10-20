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


def _month_label(abs_month):
    month_of_year = abs_month % 12
    return config.MONTH_NAMES[month_of_year]


def _is_summer(abs_month):
    return int((abs_month % 12) in config.SUMMER_MONTHS)


def _match_labour(supply, demand):
    employed = float(min(supply, demand))
    unmet = float(max(0.0, demand - supply))
    idle = float(max(0.0, supply - demand))
    return employed, unmet, idle


def run_monthly_simulation(
    world,
    scenario="baseline",
    n_months=None,
    seed=None,
    history=False,
    start_storage=None,
    start_food_price_index=None,
    start_retail_stock=None,
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

    population = int(world["meta"]["population"])
    n_perm = int((people["worker_role"] == "permanent_farm").sum())
    n_seasonal = int((people["worker_role"] == "seasonal_farm").sum())
    n_immigrant = int((people["worker_role"] == "immigrant_farm").sum())

    manure_per_month = 0.0
    livestock_water_m3 = 0.0
    livestock_revenue = 0.0
    n_cows = 0.0
    n_sheep = 0.0
    if len(herds):
        for _, h in herds.iterrows():
            manure_per_month += float(h["n_animals"] * h["manure_kg_month_per_animal"])
            spec = SPECIES[h["species"]]
            livestock_water_m3 += float(h["n_animals"] * spec["water_l_day"] * 30.0 / 1000.0)
            livestock_revenue += float(h["n_animals"] * spec["revenue_per_animal_month"])
            if h["species"] == "cow":
                n_cows += float(h["n_animals"])
            elif h["species"] == "sheep":
                n_sheep += float(h["n_animals"])

    crop_water_demand = float((crops["area_ha"] * crops["water_m3_ha_month"]).sum())
    household_water = float(population * config.HOUSEHOLD_WATER_M3_PER_PERSON)
    food_demand_t = float(population * config.FOOD_KG_PER_PERSON_MONTH / 1000.0)
    edible_mask = ~crops["commodity"].isin(["forage", "paddy_feed"])

    rows = []
    food_price_index = 1.0 if start_food_price_index is None else float(start_food_price_index)
    retail_stock = (
        food_demand_t * 0.5 if start_retail_stock is None else float(start_retail_stock)
    )
    for m in range(n_months):
        abs_month = month_offset + m
        month_of_year = abs_month % 12
        is_summer = _is_summer(abs_month)
        season_name = _month_label(abs_month)

        rainfall = _seasonal_rainfall(
            abs_month,
            config.BASE_RAINFALL_MM,
            params["rainfall_factor"],
            params["rainfall_noise"],
            rng,
        )
        inflow = rainfall * 0.001 * config.CATCHMENT_AREA_M2 * config.RUNOFF_COEFFICIENT
        storage_start = storage
        uncapped = storage_start + inflow
        spill = max(0.0, uncapped - capacity)
        storage = min(capacity, uncapped)

        efficiency = float(params["irrigation_efficiency"])
        if is_summer:
            irrig_season = 1.38
        elif month_of_year in {2, 3, 9, 10}:
            irrig_season = 1.08
        else:
            irrig_season = 0.78
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

        env_alloc = min(env_flow, storage)
        withdrawal = irrig_alloc + hh_alloc + live_alloc + env_alloc
        storage = max(0.0, storage - withdrawal)
        storage_end = storage
        water_balance_residual = storage_start + inflow - spill - withdrawal - storage_end
        unmet = max(
            0.0,
            irrig_need + household_water + livestock_water_m3 - (irrig_alloc + hh_alloc + live_alloc),
        )
        stress = min(
            1.0,
            unmet / max(1.0, irrig_need + household_water + livestock_water_m3),
        )
        storage_ratio = storage_end / max(capacity, 1.0)
        stress = min(1.0, max(stress, 1.0 - storage_ratio) * 0.55 + stress * 0.45)

        water_factor = 1.0 if irrig_need <= 0 else irrig_alloc / irrig_need
        monthly_crop_t = float(
            (crops["area_ha"] * crops["yield_t_ha_year"] / 12.0 * water_factor).sum()
        )
        edible_crop_t = float(
            (
                crops.loc[edible_mask, "area_ha"]
                * crops.loc[edible_mask, "yield_t_ha_year"]
                / 12.0
                * water_factor
            ).sum()
        )
        forage_t = float(
            crops.loc[crops["commodity"].isin(["forage", "paddy_feed"]), "area_ha"].sum()
            * 0.55
            * water_factor
            * (0.85 + 0.15 * np.sin(2 * np.pi * month_of_year / 12.0))
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
        livestock_food_t = herd_factor * (n_cows * 0.022 + n_sheep * 0.0025)
        local_food_supply_t = (edible_crop_t + livestock_food_t) * config.LOCAL_FOOD_SHARE

        stock0 = retail_stock
        gross = stock0 + local_food_supply_t
        if gross >= food_demand_t:
            food_import_t = 0.0
            leftover = gross - food_demand_t
            target_stock = food_demand_t * config.FOOD_STOCK_TARGET_MONTHS
            retail_stock = min(leftover, target_stock)
            food_export_t = leftover - retail_stock
        else:
            food_import_t = food_demand_t - gross
            food_export_t = 0.0
            retail_stock = 0.0
        food_gap_t = max(0.0, food_demand_t - local_food_supply_t)
        food_balance_residual = (
            stock0 + local_food_supply_t + food_import_t - food_demand_t - food_export_t - retail_stock
        )

        local_cover = local_food_supply_t / max(food_demand_t, 1e-9)
        food_price_index = float(
            np.clip(
                food_price_index
                * (
                    1.0
                    + 0.1 * stress
                    + 0.16 * max(0.0, 1.0 - local_cover)
                    - 0.04 * (food_export_t / max(food_demand_t, 1e-9))
                    + 0.05 * (food_import_t / max(food_demand_t, 1e-9))
                ),
                0.75,
                2.4,
            )
        )

        manure_kg = manure_per_month * herd_factor
        runoff_share = float(params["manure_runoff_base"]) + 0.35 * stress
        nutrient_load = manure_kg * 0.0065 * runoff_share
        river_flow = max(35_000.0, inflow * 0.15 + env_flow * 0.5)
        nutrient_mg_l = nutrient_load * 1000.0 / river_flow
        exceed = int(nutrient_mg_l > config.NUTRIENT_LIMIT_MG_L)

        crop_revenue = 0.0
        for _, crop in crops.iterrows():
            tons = float(crop["area_ha"] * crop["yield_t_ha_year"] / 12.0 * water_factor)
            crop_revenue += tons * _crop_price(crop["commodity"], food_price_index)

        farm_revenue = crop_revenue + livestock_revenue * herd_factor
        import_cost = food_import_t * 900.0 * food_price_index
        opex = (
            float(farms["area_ha"].sum()) * 52.0
            + livestock_water_m3 * 0.55
            + irrig_alloc * 0.32
            + feed_gap * 260.0
            + exceed * 18000.0
            + stress * 25000.0
            + import_cost * 0.05
        )
        subsidy = farm_revenue * float(params["subsidy_rate"]) * (0.4 + stress)
        profit = farm_revenue + subsidy - opex
        subsidy_need = max(0.0, opex - farm_revenue)

        labour_scale = 0.65 + 0.7 * water_factor
        peak = 1.35 if is_summer or month_of_year in {10} else 0.75
        perm_demand = n_perm * (0.9 + 0.1 * water_factor)
        seasonal_demand = n_seasonal * labour_scale * peak
        immigrant_demand = n_immigrant * labour_scale * (1.25 if (is_summer and stress < 0.45) else 0.7)
        perm_employed, perm_unmet, perm_idle = _match_labour(n_perm, perm_demand)
        seasonal_employed, seasonal_unmet, seasonal_idle = _match_labour(
            n_seasonal, seasonal_demand
        )
        immigrant_employed, immigrant_unmet, immigrant_idle = _match_labour(
            n_immigrant, immigrant_demand
        )
        labour_unmet_total = perm_unmet + seasonal_unmet + immigrant_unmet

        rows.append(
            {
                "region_id": REGION_ID,
                "month": abs_month + 1,
                "calendar_month": season_name,
                "is_summer": is_summer,
                "rainfall_mm": rainfall,
                "inflow_m3": inflow,
                "spill_m3": spill,
                "storage_start_m3": storage_start,
                "storage_m3": storage_end,
                "storage_ratio": storage_ratio,
                "irrigation_use_m3": irrig_alloc,
                "household_water_m3": hh_alloc,
                "livestock_water_m3": live_alloc,
                "environmental_flow_m3": env_alloc,
                "withdrawal_m3": withdrawal,
                "unmet_water_m3": unmet,
                "water_stress_index": stress,
                "water_balance_residual": water_balance_residual,
                "manure_kg": manure_kg,
                "nutrient_load_kg": nutrient_load,
                "river_nutrient_mg_l": nutrient_mg_l,
                "nutrient_exceedance": exceed,
                "crop_supply_t": monthly_crop_t,
                "edible_crop_t": edible_crop_t,
                "livestock_food_t": livestock_food_t,
                "local_food_supply_t": local_food_supply_t,
                "food_demand_t": food_demand_t,
                "food_gap_t": food_gap_t,
                "food_import_t": food_import_t,
                "food_export_t": food_export_t,
                "retail_stock_t": retail_stock,
                "food_balance_residual": food_balance_residual,
                "feed_gap_t": feed_gap,
                "herd_factor": herd_factor,
                "farm_revenue": farm_revenue,
                "farm_opex": opex,
                "farm_profit": profit,
                "labour_supply_permanent": float(n_perm),
                "labour_supply_seasonal": float(n_seasonal),
                "labour_supply_immigrant": float(n_immigrant),
                "labour_demand_permanent": perm_demand,
                "labour_demand_seasonal": seasonal_demand,
                "labour_demand_immigrant": immigrant_demand,
                "labour_employed_permanent": perm_employed,
                "labour_employed_seasonal": seasonal_employed,
                "labour_employed_immigrant": immigrant_employed,
                "labour_unmet_permanent": perm_unmet,
                "labour_unmet_seasonal": seasonal_unmet,
                "labour_unmet_immigrant": immigrant_unmet,
                "labour_idle_permanent": perm_idle,
                "labour_idle_seasonal": seasonal_idle,
                "labour_idle_immigrant": immigrant_idle,
                "labour_unmet_total": labour_unmet_total,
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
            "calendar_month": out["calendar_month"],
            "is_summer": out["is_summer"],
            "crop_supply": out["crop_supply_t"],
            "local_food_supply": out["local_food_supply_t"],
            "food_demand": out["food_demand_t"],
            "food_import": out["food_import_t"],
            "food_export": out["food_export_t"],
            "retail_stock": out["retail_stock_t"],
            "food_price_index": out["food_price_index"],
            "labour_demand_seasonal": out["labour_demand_seasonal"],
            "immigrant_labour_demand": out["labour_demand_immigrant"],
            "labour_unmet_total": out["labour_unmet_total"],
            "water_stress_index": out["water_stress_index"],
            "nutrient_exceedance": out["nutrient_exceedance"],
            "farm_profit": out["farm_profit"],
            "subsidy_need": out["subsidy_need"],
        }
    )


def summer_month_summary(monthly):
    frame = monthly.copy()
    summer = frame.loc[frame["is_summer"] == 1]
    winterish = frame.loc[frame["is_summer"] == 0]
    def _mean(df, col):
        return float(df[col].mean()) if len(df) else float("nan")
    return {
        "n_summer_months": int(len(summer)),
        "summer_irrig_m3": _mean(summer, "irrigation_use_m3"),
        "other_irrig_m3": _mean(winterish, "irrigation_use_m3"),
        "summer_food_import_t": _mean(summer, "food_import_t"),
        "other_food_import_t": _mean(winterish, "food_import_t"),
        "summer_labour_demand": _mean(summer, "labour_demand_seasonal"),
        "other_labour_demand": _mean(winterish, "labour_demand_seasonal"),
        "summer_water_stress": _mean(summer, "water_stress_index"),
        "other_water_stress": _mean(winterish, "water_stress_index"),
    }
