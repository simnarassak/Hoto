import numpy as np
import pandas as pd

from hoto_rural import config
from hoto_rural.simulate import run_monthly_simulation


def run_history(world, seed=None):
    return run_monthly_simulation(
        world,
        scenario="baseline",
        n_months=config.HISTORY_MONTHS,
        seed=seed if seed is not None else config.RANDOM_SEED,
        history=True,
        month_offset=0,
    )


def run_monte_carlo(
    world,
    scenario="drought",
    n_paths=None,
    forecast_months=None,
    seed=None,
    start_storage=None,
    start_food_price_index=None,
    month_offset=None,
):
    n_paths = config.N_MONTE_CARLO if n_paths is None else n_paths
    forecast_months = config.FORECAST_MONTHS if forecast_months is None else forecast_months
    seed = config.RANDOM_SEED if seed is None else seed
    month_offset = config.HISTORY_MONTHS if month_offset is None else month_offset
    paths = []
    for i in range(n_paths):
        frame = run_monthly_simulation(
            world,
            scenario=scenario,
            n_months=forecast_months,
            seed=seed + 1000 + i * 17,
            history=False,
            start_storage=start_storage,
            start_food_price_index=start_food_price_index,
            month_offset=month_offset,
        )
        frame = frame.copy()
        frame["path_id"] = i
        paths.append(frame)
    return pd.concat(paths, ignore_index=True)


def summarise_paths(paths, value_col="storage_m3"):
    rows = []
    for month, part in paths.groupby("month"):
        vals = part[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "month": int(month),
                "p10": float(np.quantile(vals, 0.1)),
                "median": float(np.median(vals)),
                "p90": float(np.quantile(vals, 0.9)),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_forecast_bundle(world, scenario="drought", value_col="storage_m3", n_paths=None, seed=None):
    history = run_history(world, seed=seed)
    start_storage = float(history.iloc[-1]["storage_m3"])
    start_price = float(history.iloc[-1]["food_price_index"])
    paths = run_monte_carlo(
        world,
        scenario=scenario,
        n_paths=n_paths,
        seed=seed,
        start_storage=start_storage,
        start_food_price_index=start_price,
        month_offset=int(history["month"].max()),
    )
    summary = summarise_paths(paths, value_col=value_col)
    last = paths["month"].max()
    median_by_month = paths.groupby("month", as_index=False).agg(
        crop_supply=("crop_supply_t", "median"),
        food_price_index=("food_price_index", "median"),
        labour_demand_seasonal=("labour_demand_seasonal", "median"),
        immigrant_labour_demand=("labour_demand_immigrant", "median"),
        water_stress_index=("water_stress_index", "median"),
        nutrient_exceedance=("nutrient_exceedance", "mean"),
        farm_profit=("farm_profit", "median"),
        subsidy_need=("subsidy_need", "median"),
    )
    median_by_month.insert(0, "region_id", "Rural")
    threshold_prob = float(
        (paths.groupby("path_id")["storage_m3"].min() < config.BASE_STORAGE_M3 * 0.25).mean()
    )
    exceed_prob = float(paths.groupby("path_id")["nutrient_exceedance"].max().mean())
    return {
        "history": history,
        "paths": paths,
        "summary": summary,
        "exports": median_by_month,
        "now_month": int(history["month"].max()),
        "metrics": {
            "prob_storage_below_25pct": threshold_prob,
            "mean_path_nutrient_exceedance": exceed_prob,
            "median_final_profit": float(
                paths.loc[paths["month"] == last, "farm_profit"].median()
            ),
            "start_storage_m3": start_storage,
        },
    }
