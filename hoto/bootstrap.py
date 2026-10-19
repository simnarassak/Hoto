"""Ensure demo artifacts exist for cloud deploys (HF Spaces / Streamlit Cloud)."""

import logging
from pathlib import Path

from hoto.config import (
    DATA_DIR,
    DEMO_SCALE,
    MODELS_DIR,
    WORLD,
)

log = logging.getLogger("hoto.bootstrap")

REQUIRED_DATA_FILES = [
    "investigator_view.csv",
    "model_lab_comparison.csv",
    "hoto_scenario_registry.csv",
    "hoto_claim_features.csv",
    "hoto_claim_metadata.csv",
    "hoto_claims_synthetic.csv",
    "hoto_providers.csv",
]


def artifacts_ready(data_dir=None, models_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR

    data_ok = all((data_dir / name).exists() for name in REQUIRED_DATA_FILES)
    from hoto.infer import models_available

    return data_ok and models_available(models_dir)


def apply_demo_scale():
    """Shrink generation sizes for cloud-friendly first boot."""
    import hoto.config as cfg
    import hoto.data_gen as data_gen
    import hoto.scenarios as scenarios

    cfg.WORLD["population"] = DEMO_SCALE["population"]
    cfg.N_PROVIDERS = DEMO_SCALE["n_providers"]
    cfg.N_CLAIMS = DEMO_SCALE["n_claims"]
    cfg.N_DUPLICATE_PAYMENTS = DEMO_SCALE["n_duplicate_payments"]
    cfg.N_DUPLICATE_CLAIMS = DEMO_SCALE["n_duplicate_claims"]
    cfg.N_FREQUENCY_CLAIMS = DEMO_SCALE["n_frequency_claims"]
    cfg.N_PROVIDER_ANOMALIES = DEMO_SCALE["n_provider_anomalies"]

    data_gen.N_PROVIDERS = DEMO_SCALE["n_providers"]
    data_gen.N_CLAIMS = DEMO_SCALE["n_claims"]
    scenarios.N_DUPLICATE_PAYMENTS = DEMO_SCALE["n_duplicate_payments"]
    scenarios.N_DUPLICATE_CLAIMS = DEMO_SCALE["n_duplicate_claims"]
    scenarios.N_FREQUENCY_CLAIMS = DEMO_SCALE["n_frequency_claims"]
    scenarios.N_PROVIDER_ANOMALIES = DEMO_SCALE["n_provider_anomalies"]


def ensure_artifacts(data_dir=None, models_dir=None, force_demo=False):
    """
    If artifacts are missing, train a compact demo world in-place.

    Full 100k training is too heavy for free HF Spaces / Streamlit Cloud
    cold starts, so demo scale is used unless artifacts already exist.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR

    if artifacts_ready(data_dir, models_dir) and not force_demo:
        return {"status": "ready", "data_dir": data_dir, "models_dir": models_dir}

    log.info("Artifacts missing - building demo-scale Hoto world for cloud boot")
    apply_demo_scale()

    from hoto.pipeline import configure_logging, run_pipeline

    configure_logging()
    run_pipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        seed=WORLD["random_seed"],
        skip_train=False,
    )
    return {
        "status": "built_demo",
        "data_dir": data_dir,
        "models_dir": models_dir,
        "population": DEMO_SCALE["population"],
        "n_claims": DEMO_SCALE["n_claims"],
    }
