"""Risk dimensions: ensemble triage, rule hits, provider risk."""

from pathlib import Path

import numpy as np
import pandas as pd

from hoto.config import (
    ENSEMBLE_WEIGHTS,
    RULE_MATCHING_CLAIM_COUNT,
    RULE_PAYMENT_TO_APPROVED_RATIO,
    RULE_PRIOR_CLAIMS_30D,
    RULE_PROVIDER_AMOUNT_MULTIPLIER,
)


def _normalize_series(values):
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(len(values), dtype=float)
    return (values - lo) / (hi - lo)


def add_ensemble_score(scored, weights=None):
    """Add a blended triage score from XGBoost, logistic, and anomaly."""
    weights = ENSEMBLE_WEIGHTS if weights is None else weights
    result = scored.copy()
    anomaly_norm = _normalize_series(result["anomaly_score"])
    result["anomaly_score_norm"] = anomaly_norm
    result["ensemble_score"] = (
        weights["xgboost"] * result["xgboost_probability"]
        + weights["logistic"] * result["logistic_probability"]
        + weights["anomaly"] * anomaly_norm
    )
    return result


def add_claim_rule_hits(claim_features):
    """Transparent rule signals for claims (complement ML, not replace it)."""
    result = claim_features.copy()
    result["rule_high_velocity_30d"] = (
        result["prior_claims_30d"] >= RULE_PRIOR_CLAIMS_30D
    ).astype(int)
    result["rule_duplicate_signature"] = (
        result["prior_matching_claim_count"] >= RULE_MATCHING_CLAIM_COUNT
    ).astype(int)
    result["rule_provider_amount_spike"] = (
        result["claimed_amount"]
        >= (result["provider_prior_mean_amount"] * RULE_PROVIDER_AMOUNT_MULTIPLIER)
    ).astype(int)
    result["rule_hit_count"] = (
        result["rule_high_velocity_30d"]
        + result["rule_duplicate_signature"]
        + result["rule_provider_amount_spike"]
    )
    return result


def add_payment_rule_hits(payment_features):
    """Transparent rule signals for payments."""
    result = payment_features.copy()
    result["rule_near_full_payout"] = (
        result["payment_to_approved_ratio"] >= RULE_PAYMENT_TO_APPROVED_RATIO
    ).astype(int)
    result["rule_repeat_same_amount"] = (
        result["prior_same_claim_amount_count"] >= 1
    ).astype(int)
    result["rule_hit_count"] = (
        result["rule_near_full_payout"] + result["rule_repeat_same_amount"]
    )
    return result


def build_provider_risk(claim_scored, providers=None, data_dir=None):
    """
    Provider-level risk dimension.

    Aggregates claim ML scores and activity volume so investigators can
    review provider networks, not only single claims or payments.
    """
    required = {
        "provider_id",
        "claim_id",
        "ensemble_score",
        "xgboost_probability",
        "anomaly_score",
        "rule_hit_count",
    }
    missing = required - set(claim_scored.columns)
    if missing:
        raise ValueError(f"claim_scored missing columns for provider risk: {missing}")

    grouped = claim_scored.groupby("provider_id", as_index=False).agg(
        claim_count=("claim_id", "size"),
        mean_ensemble_score=("ensemble_score", "mean"),
        max_ensemble_score=("ensemble_score", "max"),
        mean_xgboost=("xgboost_probability", "mean"),
        mean_anomaly=("anomaly_score", "mean"),
        mean_rule_hits=("rule_hit_count", "mean"),
        max_rule_hits=("rule_hit_count", "max"),
    )
    if "claimed_amount" in claim_scored.columns:
        amounts = (
            claim_scored.groupby("provider_id", as_index=False)["claimed_amount"]
            .sum()
            .rename(columns={"claimed_amount": "total_claimed_amount"})
        )
        grouped = grouped.merge(amounts, on="provider_id", how="left")

    score_norm = _normalize_series(grouped["mean_ensemble_score"])
    volume_norm = _normalize_series(np.log1p(grouped["claim_count"]))
    grouped["provider_risk_index"] = 0.7 * score_norm + 0.3 * volume_norm
    grouped = grouped.sort_values(
        "provider_risk_index", ascending=False
    ).reset_index(drop=True)

    if providers is not None and "provider_id" in providers.columns:
        meta_cols = [
            c for c in ["provider_id", "region", "provider_type", "provider_status"]
            if c in providers.columns
        ]
        grouped = grouped.merge(providers[meta_cols], on="provider_id", how="left")

    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(data_dir / "provider_risk_summary.csv", index=False)

    return grouped


def enrich_scored_claims(claim_features, scored):
    with_rules = add_claim_rule_hits(claim_features)
    extra_cols = [
        "rule_high_velocity_30d",
        "rule_duplicate_signature",
        "rule_provider_amount_spike",
        "rule_hit_count",
    ]
    if "provider_id" in with_rules.columns and "provider_id" not in scored.columns:
        extra_cols.insert(0, "provider_id")

    keep = ["claim_id"] + [c for c in extra_cols if c in with_rules.columns]
    merged = scored.merge(with_rules[keep], on="claim_id", how="left")
    return add_ensemble_score(merged)


def enrich_scored_payments(payment_features, scored):
    with_rules = add_payment_rule_hits(payment_features)
    extra_cols = [
        "rule_near_full_payout",
        "rule_repeat_same_amount",
        "rule_hit_count",
    ]
    keep = ["payment_id"] + [c for c in extra_cols if c in with_rules.columns]
    merged = scored.merge(with_rules[keep], on="payment_id", how="left")
    return add_ensemble_score(merged)
