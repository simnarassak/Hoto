from pathlib import Path

from hoto.config import (
    AGE_BANDS,
    EMPLOYMENT_DISTRIBUTION,
    IMMIGRANT_STATUS_DISTRIBUTION,
    POPULATION_GROUPS,
    REGIONS,
    WORLD,
)

RURAL_SHARE = float(REGIONS["Rural"])
RURAL_POPULATION = int(round(WORLD["population"] * RURAL_SHARE))
RANDOM_SEED = int(WORLD["random_seed"]) + 17
HISTORY_MONTHS = 36
FORECAST_MONTHS = 24
N_FARMS = 120
N_MONTE_CARLO = 80
NUTRIENT_LIMIT_MG_L = 1.8
BASE_STORAGE_M3 = 11_000_000.0
BASE_RAINFALL_MM = 84.0
CATCHMENT_AREA_M2 = 88_000_000.0
RUNOFF_COEFFICIENT = 0.26
HOUSEHOLD_WATER_M3_PER_PERSON = 0.35
DATA_DIR = Path("data") / "rural"

SCENARIO_PARAMS = {
    "baseline": {
        "rainfall_factor": 1.0,
        "rainfall_noise": 0.15,
        "irrigation_efficiency": 0.68,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.05,
        "manure_runoff_base": 0.16,
    },
    "drought": {
        "rainfall_factor": 0.72,
        "rainfall_noise": 0.2,
        "irrigation_efficiency": 0.68,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.05,
        "manure_runoff_base": 0.22,
    },
    "efficiency": {
        "rainfall_factor": 0.72,
        "rainfall_noise": 0.18,
        "irrigation_efficiency": 0.86,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.08,
        "manure_runoff_base": 0.14,
    },
    "allocation": {
        "rainfall_factor": 0.72,
        "rainfall_noise": 0.18,
        "irrigation_efficiency": 0.68,
        "allocation_priority": "essential_first",
        "subsidy_rate": 0.1,
        "manure_runoff_base": 0.2,
    },
    "combined": {
        "rainfall_factor": 0.72,
        "rainfall_noise": 0.17,
        "irrigation_efficiency": 0.86,
        "allocation_priority": "essential_first",
        "subsidy_rate": 0.12,
        "manure_runoff_base": 0.12,
    },
}

COMMODITY_PRICE = {
    "milk": 0.9,
    "meat_wool": 6.5,
    "avocado": 3200.0,
    "kiwi": 2800.0,
    "apple": 1800.0,
    "grapes": 2200.0,
    "berries": 4500.0,
    "carrot": 900.0,
    "cucumber": 1100.0,
    "paddy_feed": 350.0,
    "forage": 180.0,
}

POPULATION_GROUPS = POPULATION_GROUPS
AGE_BANDS = AGE_BANDS
IMMIGRANT_STATUS_DISTRIBUTION = IMMIGRANT_STATUS_DISTRIBUTION
EMPLOYMENT_DISTRIBUTION = EMPLOYMENT_DISTRIBUTION
