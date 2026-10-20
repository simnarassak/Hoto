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
NUTRIENT_LIMIT_MG_L = 2.5
BASE_STORAGE_M3 = 25_000_000.0
BASE_RAINFALL_MM = 85.0
DATA_DIR = Path("data") / "rural"

SCENARIO_PARAMS = {
    "baseline": {
        "rainfall_factor": 1.0,
        "irrigation_efficiency": 0.7,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.05,
    },
    "drought": {
        "rainfall_factor": 0.55,
        "irrigation_efficiency": 0.7,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.05,
    },
    "efficiency": {
        "rainfall_factor": 0.55,
        "irrigation_efficiency": 0.85,
        "allocation_priority": "balanced",
        "subsidy_rate": 0.08,
    },
    "allocation": {
        "rainfall_factor": 0.55,
        "irrigation_efficiency": 0.7,
        "allocation_priority": "essential_first",
        "subsidy_rate": 0.1,
    },
    "combined": {
        "rainfall_factor": 0.55,
        "irrigation_efficiency": 0.85,
        "allocation_priority": "essential_first",
        "subsidy_rate": 0.12,
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
