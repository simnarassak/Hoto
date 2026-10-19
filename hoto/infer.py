"""Load saved models and score claim/payment feature rows for the public app."""

from pathlib import Path

import joblib
import pandas as pd

from hoto.config import CLASSIFIER_THRESHOLD, DATA_DIR, MODELS_DIR
from hoto.features import CLAIM_MODEL_CATEGORICAL, CLAIM_MODEL_NUMERIC, PAYMENT_MODEL_FEATURES
from hoto.risk_layers import (
    enrich_scored_claims,
    enrich_scored_payments,
)

PAYMENT_MODEL_FILES = {
    "logistic": "hoto_payment_fraud_logistic_regression.joblib",
    "anomaly": "hoto_payment_isolation_forest.joblib",
    "xgboost": "payment_xgboost_pipeline.joblib",
}

CLAIM_MODEL_FILES = {
    "logistic": "hoto_claim_fraud_logistic_regression.joblib",
    "anomaly": "hoto_claim_isolation_forest.joblib",
    "xgboost": "claim_xgboost_pipeline.joblib",
}

def models_available(models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    required = list(PAYMENT_MODEL_FILES.values()) + list(CLAIM_MODEL_FILES.values())
    return all((models_dir / name).exists() for name in required)

def load_payment_models(models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    return {
        key: joblib.load(models_dir / filename)
        for key, filename in PAYMENT_MODEL_FILES.items()
    }

def load_claim_models(models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    return {
        key: joblib.load(models_dir / filename)
        for key, filename in CLAIM_MODEL_FILES.items()
    }

def _ensure_columns(frame, columns):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame[columns].copy()

def score_payments(
    payment_features,
    models=None,
    models_dir=None,
    threshold=CLASSIFIER_THRESHOLD,
    enrich=True,
):
    """Score a payment feature table. Returns probabilities, flags, optional enrichments."""
    if models is None:
        models = load_payment_models(models_dir)

    X = _ensure_columns(payment_features, PAYMENT_MODEL_FEATURES)
    result = payment_features.copy()

    result["logistic_probability"] = models["logistic"].predict_proba(X)[:, 1]
    result["xgboost_probability"] = models["xgboost"].predict_proba(X)[:, 1]
    result["anomaly_score"] = -models["anomaly"].decision_function(X)
    result["classifier_flag"] = (
        result["logistic_probability"] >= threshold
    ).astype(int)
    result["anomaly_flag"] = (
        models["anomaly"].predict(X) == -1
    ).astype(int)
    result["xgboost_flag"] = (
        result["xgboost_probability"] >= threshold
    ).astype(int)

    if enrich:
        result = enrich_scored_payments(payment_features, result)
        result["ensemble_flag"] = (
            result["ensemble_score"] >= threshold
        ).astype(int)
    return result

def score_claims(
    claim_features,
    models=None,
    models_dir=None,
    threshold=CLASSIFIER_THRESHOLD,
    enrich=True,
    claim_metadata=None,
):
    """Score a claim feature table. Returns probabilities, flags, optional enrichments."""
    if models is None:
        models = load_claim_models(models_dir)

    feature_columns = CLAIM_MODEL_CATEGORICAL + CLAIM_MODEL_NUMERIC
    X = _ensure_columns(claim_features, feature_columns)
    result = claim_features.copy()

    result["logistic_probability"] = models["logistic"].predict_proba(X)[:, 1]
    result["xgboost_probability"] = models["xgboost"].predict_proba(X)[:, 1]
    result["anomaly_score"] = -models["anomaly"].decision_function(X)
    result["classifier_flag"] = (
        result["logistic_probability"] >= threshold
    ).astype(int)
    result["anomaly_flag"] = (
        models["anomaly"].predict(X) == -1
    ).astype(int)
    result["xgboost_flag"] = (
        result["xgboost_probability"] >= threshold
    ).astype(int)

    if enrich:
        features_for_rules = claim_features.copy()
        if claim_metadata is not None and "provider_id" in claim_metadata.columns:
            features_for_rules = features_for_rules.merge(
                claim_metadata[["claim_id", "provider_id"]],
                on="claim_id",
                how="left",
            )
        result = enrich_scored_claims(features_for_rules, result)
        result["ensemble_flag"] = (
            result["ensemble_score"] >= threshold
        ).astype(int)
    return result

def load_artifact_csv(name, data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    path = data_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)

def top_n_queue(scored, score_column="ensemble_score", n=100, threshold=None):
    frame = scored.copy()
    if threshold is not None and score_column in frame.columns:
        frame = frame.loc[frame[score_column] >= threshold]
    if score_column not in frame.columns:
        raise ValueError(f"Unknown score column: {score_column}")
    return frame.sort_values(score_column, ascending=False).head(n).copy()
