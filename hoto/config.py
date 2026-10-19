"""Central configuration for the Hoto synthetic claims platform."""

from pathlib import Path

WORLD = {
    "country_name": "Hoto",
    "population": 100_000,
    "region_count": 5,
    "historical_years": 3,
    "random_seed": 42,
}

POPULATION_GROUPS = {
    "European": 0.60,
    "Indigenous": 0.20,
    "Immigrant": 0.20,
}

REGIONS = {
    "Capital": 0.25,
    "Metro": 0.25,
    "Rural": 0.20,
    "Coastal": 0.15,
    "Industrial": 0.15,
}

AGE_BANDS = {
    "0-17": 0.22,
    "18-34": 0.25,
    "35-49": 0.23,
    "50-64": 0.18,
    "65+": 0.12,
}

IMMIGRANT_STATUSES = [
    "Citizen",
    "Permanent Resident",
    "Resident Visa",
    "Student Visa",
    "Work Visa",
]

IMMIGRANT_STATUS_DISTRIBUTION = {
    "Citizen": 0.20,
    "Permanent Resident": 0.25,
    "Resident Visa": 0.20,
    "Student Visa": 0.15,
    "Work Visa": 0.20,
}

EMPLOYMENT_DISTRIBUTION = {
    "Employed": 0.58,
    "Self-employed": 0.12,
    "Student": 0.12,
    "Unemployed": 0.08,
    "Retired": 0.10,
}

N_PROVIDERS = 500
N_CLAIMS = 25_000

PROVIDER_TYPES = [
    "Medical",
    "Rehabilitation",
    "Transport",
    "Pharmacy",
    "Other Support",
]

PROVIDER_TYPE_PROBS = [0.35, 0.25, 0.15, 0.15, 0.10]

CLAIM_TYPES = [
    "Injury Compensation",
    "Rehabilitation",
    "Medical Expenses",
    "Travel Support",
    "Income Support",
]

CLAIM_TYPE_PROBS = [0.20, 0.20, 0.25, 0.15, 0.20]

CLAIM_AMOUNT_RANGES = {
    "Injury Compensation": (500, 15000),
    "Rehabilitation": (100, 6000),
    "Medical Expenses": (50, 3000),
    "Travel Support": (20, 800),
    "Income Support": (300, 10000),
}

CLAIM_STATUSES = ["Approved", "Declined", "Under Review", "Pending"]
CLAIM_STATUS_PROBS = [0.65, 0.15, 0.10, 0.10]

N_DUPLICATE_PAYMENTS = 100
N_DUPLICATE_CLAIMS = 100
N_FREQUENCY_CLAIMS = 5
N_PROVIDER_ANOMALIES = 5

TEST_SIZE = 0.25
CLASSIFIER_THRESHOLD = 0.5
REVIEW_CAPACITIES = (100, 500)

ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,
    "logistic": 0.30,
    "anomaly": 0.20,
}

RULE_PRIOR_CLAIMS_30D = 3
RULE_MATCHING_CLAIM_COUNT = 1
RULE_PAYMENT_TO_APPROVED_RATIO = 0.95
RULE_PROVIDER_AMOUNT_MULTIPLIER = 2.5

DEMO_SCALE = {
    "population": 5_000,
    "n_providers": 100,
    "n_claims": 2_000,
    "n_duplicate_payments": 40,
    "n_duplicate_claims": 40,
    "n_frequency_claims": 5,
    "n_provider_anomalies": 5,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

def ensure_output_dirs(data_dir=None, models_dir=None):
    data_path = Path(data_dir) if data_dir is not None else DATA_DIR
    models_path = Path(models_dir) if models_dir is not None else MODELS_DIR
    data_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    return data_path, models_path
