"""Claim and payment feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd


CLAIM_FEATURE_COLUMNS = [
    "claim_id",
    "claim_type",
    "claimant_region",
    "claimed_amount",
    "log_claimed_amount",
    "incident_to_submission_days",
    "submission_month",
    "submission_year",
    "prior_claim_count",
    "prior_claims_30d",
    "prior_matching_claim_count",
    "provider_prior_claim_count",
    "provider_prior_mean_amount",
    "provider_has_history",
]

CLAIM_MODEL_CATEGORICAL = ["claim_type", "claimant_region"]
CLAIM_MODEL_NUMERIC = [
    "claimed_amount",
    "log_claimed_amount",
    "incident_to_submission_days",
    "submission_month",
    "submission_year",
    "prior_claim_count",
    "prior_claims_30d",
    "prior_matching_claim_count",
    "provider_prior_claim_count",
    "provider_prior_mean_amount",
    "provider_has_history",
]

PAYMENT_FEATURE_COLUMNS = [
    "payment_id",
    "claim_type",
    "payment_date",
    "days_since_submission",
    "absolute_payment_amount",
    "log_payment_amount",
    "payment_to_approved_ratio",
    "prior_payment_count",
    "prior_same_claim_amount_count",
    "prior_net_paid_amount",
    "prior_net_paid_to_approved_ratio",
]

PAYMENT_MODEL_FEATURES = [
    "claim_type",
    "days_since_submission",
    "absolute_payment_amount",
    "log_payment_amount",
    "payment_to_approved_ratio",
    "prior_payment_count",
    "prior_same_claim_amount_count",
    "prior_net_paid_amount",
    "prior_net_paid_to_approved_ratio",
]


def _prior_claims_30d(claim_data):
    claim_data = claim_data.copy()
    claim_data["prior_claims_30d"] = 0

    for _, group in claim_data.groupby("person_id", sort=False):
        positions = group.index.to_numpy()
        dates = group["submission_date"].to_numpy(dtype="datetime64[ns]")
        counts = np.zeros(len(group), dtype=int)

        for i, current_date in enumerate(dates):
            window_start = current_date - np.timedelta64(30, "D")
            left = np.searchsorted(dates[:i], window_start, side="left")
            counts[i] = i - left

        claim_data.loc[positions, "prior_claims_30d"] = counts

    return claim_data


def build_claim_features(claims_synthetic, fraud_labels, amount_median=None):
    claim_data = claims_synthetic.copy()
    claim_data["submission_date"] = pd.to_datetime(claim_data["submission_date"])
    claim_data["incident_date"] = pd.to_datetime(claim_data["incident_date"])
    claim_data = claim_data.sort_values(
        ["submission_date", "claim_id"]
    ).reset_index(drop=True)

    claim_data["incident_to_submission_days"] = (
        claim_data["submission_date"] - claim_data["incident_date"]
    ).dt.days
    claim_data["log_claimed_amount"] = np.log1p(
        claim_data["claimed_amount"].clip(lower=0)
    )
    claim_data["submission_month"] = claim_data["submission_date"].dt.month
    claim_data["submission_year"] = claim_data["submission_date"].dt.year

    claim_data["prior_claim_count"] = claim_data.groupby("person_id").cumcount()
    claim_data = _prior_claims_30d(claim_data)

    claim_signature = [
        "person_id", "incident_date", "claim_type", "claimed_amount"
    ]
    claim_data["prior_matching_claim_count"] = claim_data.groupby(
        claim_signature, dropna=False
    ).cumcount()

    claim_data["provider_prior_claim_count"] = (
        claim_data.groupby("provider_id").cumcount()
    )
    claim_data["provider_prior_mean_amount"] = (
        claim_data.groupby("provider_id")["claimed_amount"]
        .transform(lambda values: values.shift().expanding().mean())
    )
    claim_data["provider_has_history"] = (
        claim_data["provider_prior_claim_count"] > 0
    ).astype(int)

    # Prefer a caller-supplied train-only median to avoid leakage.
    if amount_median is None:
        amount_median = float(claim_data["claimed_amount"].median())
    claim_data["provider_prior_mean_amount"] = (
        claim_data["provider_prior_mean_amount"].fillna(amount_median)
    )

    claim_features = claim_data[CLAIM_FEATURE_COLUMNS].copy()
    claim_metadata = claim_data[
        [
            "claim_id",
            "person_id",
            "provider_id",
            "incident_date",
            "submission_date",
        ]
    ].copy()

    claim_targets = fraud_labels.loc[
        fraud_labels["entity_type"].eq("claim"),
        ["entity_id", "fraud_label", "scenario_id", "label_source"],
    ].copy()
    claim_targets = claim_targets.rename(columns={"entity_id": "claim_id"})

    assert claim_features["claim_id"].is_unique
    assert claim_targets["claim_id"].is_unique
    assert set(claim_features["claim_id"]) == set(claim_targets["claim_id"])
    assert "fraud_label" not in claim_features.columns
    assert "scenario_id" not in claim_features.columns
    assert "decision_date" not in claim_features.columns
    assert "approved_amount" not in claim_features.columns

    return {
        "claim_features": claim_features,
        "claim_metadata": claim_metadata,
        "claim_targets": claim_targets,
        "amount_median": amount_median,
    }


def build_payment_features(payments_synthetic, claims_synthetic, fraud_labels):
    payment_data = payments_synthetic.copy()
    payment_data["payment_date"] = pd.to_datetime(payment_data["payment_date"])
    payment_data = payment_data.sort_values(
        ["payment_date", "payment_id"]
    ).reset_index(drop=True)

    claim_lookup = claims_synthetic[
        ["claim_id", "submission_date", "approved_amount", "claim_type"]
    ].copy()
    payment_data = payment_data.merge(
        claim_lookup, on="claim_id", how="left", validate="many_to_one"
    )
    assert payment_data["approved_amount"].notna().all()

    payment_data["days_since_submission"] = (
        payment_data["payment_date"]
        - pd.to_datetime(payment_data["submission_date"])
    ).dt.days
    payment_data["absolute_payment_amount"] = payment_data["payment_amount"].abs()
    payment_data["log_payment_amount"] = np.log1p(
        payment_data["absolute_payment_amount"]
    )
    payment_data["payment_to_approved_ratio"] = (
        payment_data["absolute_payment_amount"]
        / payment_data["approved_amount"]
    )

    payment_data["prior_payment_count"] = (
        payment_data.groupby("claim_id").cumcount()
    )
    payment_data["prior_same_claim_amount_count"] = payment_data.groupby(
        ["claim_id", "absolute_payment_amount"], dropna=False
    ).cumcount()

    payment_data["cash_flow_amount"] = np.where(
        payment_data["payment_status"].eq("Paid")
        | payment_data["payment_status"].eq("Reversal"),
        payment_data["payment_amount"],
        0.0,
    )
    payment_data["prior_net_paid_amount"] = (
        payment_data.groupby("claim_id")["cash_flow_amount"]
        .transform(lambda values: values.cumsum().shift(1).fillna(0))
    )
    payment_data["prior_net_paid_to_approved_ratio"] = (
        payment_data["prior_net_paid_amount"]
        / payment_data["approved_amount"]
    )

    payment_features = payment_data[PAYMENT_FEATURE_COLUMNS].copy()
    payment_metadata = payment_data[
        [
            "payment_id",
            "claim_id",
            "payment_date",
            "payment_status",
            "reversal_reference",
        ]
    ].copy()

    payment_targets = fraud_labels.loc[
        fraud_labels["entity_type"].eq("payment"),
        ["entity_id", "fraud_label", "scenario_id", "label_source"],
    ].copy()
    payment_targets = payment_targets.rename(columns={"entity_id": "payment_id"})

    assert payment_features["payment_id"].is_unique
    assert payment_targets["payment_id"].is_unique
    assert set(payment_features["payment_id"]) == set(
        payment_targets["payment_id"]
    )
    assert "fraud_label" not in payment_features.columns
    assert "scenario_id" not in payment_features.columns
    assert "payment_status" not in payment_features.columns
    assert "reversal_reference" not in payment_features.columns

    return {
        "payment_features": payment_features,
        "payment_metadata": payment_metadata,
        "payment_targets": payment_targets,
    }


def validate_and_save_features(claim_bundle, payment_bundle, data_dir):
    claim_features = claim_bundle["claim_features"]
    payment_features = payment_bundle["payment_features"]

    claim_numeric_columns = [
        "claimed_amount",
        "log_claimed_amount",
        "incident_to_submission_days",
        "prior_claim_count",
        "prior_claims_30d",
        "prior_matching_claim_count",
        "provider_prior_claim_count",
        "provider_prior_mean_amount",
    ]
    payment_numeric_columns = [
        "days_since_submission",
        "absolute_payment_amount",
        "log_payment_amount",
        "payment_to_approved_ratio",
        "prior_payment_count",
        "prior_same_claim_amount_count",
        "prior_net_paid_amount",
        "prior_net_paid_to_approved_ratio",
    ]

    assert np.isfinite(
        claim_features[claim_numeric_columns].to_numpy(dtype=float)
    ).all()
    assert np.isfinite(
        payment_features[payment_numeric_columns].to_numpy(dtype=float)
    ).all()
    assert claim_features["prior_claim_count"].ge(0).all()
    assert claim_features["prior_claims_30d"].ge(0).all()
    assert claim_features["prior_matching_claim_count"].ge(0).all()
    assert payment_features["prior_payment_count"].ge(0).all()
    assert payment_features["prior_same_claim_amount_count"].ge(0).all()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    claim_features.to_csv(data_dir / "hoto_claim_features.csv", index=False)
    claim_bundle["claim_targets"].to_csv(
        data_dir / "hoto_claim_targets.csv", index=False
    )
    claim_bundle["claim_metadata"].to_csv(
        data_dir / "hoto_claim_metadata.csv", index=False
    )
    payment_features.to_csv(data_dir / "hoto_payment_features.csv", index=False)
    payment_bundle["payment_targets"].to_csv(
        data_dir / "hoto_payment_targets.csv", index=False
    )
    payment_bundle["payment_metadata"].to_csv(
        data_dir / "hoto_payment_metadata.csv", index=False
    )
