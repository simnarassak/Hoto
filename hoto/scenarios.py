"""Fraud scenario injection and ground-truth label construction."""

from pathlib import Path

import numpy as np
import pandas as pd

from hoto.config import (
    N_DUPLICATE_CLAIMS,
    N_DUPLICATE_PAYMENTS,
    N_FREQUENCY_CLAIMS,
    N_PROVIDER_ANOMALIES,
    WORLD,
)

def build_scenario_registry():
    return pd.DataFrame([
        {
            "scenario_id": "S001",
            "scenario_type": "Duplicate Payment",
            "description": "A second paid transaction repeats an existing payment.",
            "entity_type": "payment",
            "generation_method": "Copy a paid transaction with a new payment ID.",
        },
        {
            "scenario_id": "S002",
            "scenario_type": "Duplicate Claim",
            "description": "A new submission repeats an existing claim's details.",
            "entity_type": "claim",
            "generation_method": "Copy claim details and create a later submission.",
        },
        {
            "scenario_id": "S003",
            "scenario_type": "Abnormal Claim Frequency",
            "description": "Several related claims are clustered within a short period.",
            "entity_type": "claim",
            "generation_method": "Generate repeated claims for one synthetic claimant.",
        },
        {
            "scenario_id": "S004",
            "scenario_type": "Provider Billing Anomaly",
            "description": "Claims associated with a provider have unusually high amounts.",
            "entity_type": "claim",
            "generation_method": "Generate high-value claims associated with one provider.",
        },
    ])

def inject_fraud_scenarios(claims, payments, people, providers, seed=None):
    base_seed = WORLD["random_seed"] if seed is None else seed

    claims_base = claims.copy(deep=True)
    payments_base = payments.copy(deep=True)
    providers_base = providers.copy(deep=True)
    people_base = people.copy(deep=True)

    claims_synthetic = claims_base.copy(deep=True)
    payments_synthetic = payments_base.copy(deep=True)

    paid_candidates = payments_base.loc[
        payments_base["payment_status"].eq("Paid")
    ].copy()
    n_dup_pay = min(N_DUPLICATE_PAYMENTS, len(paid_candidates))
    if n_dup_pay == 0:
        raise ValueError(
            "No paid payments available. Check the baseline payment data."
        )

    selected = paid_candidates.sample(
        n=n_dup_pay, random_state=base_seed + 11
    ).copy()
    duplicate_payments = selected.copy()
    duplicate_payments["payment_id"] = [
        f"FDP{i:08d}" for i in range(1, len(duplicate_payments) + 1)
    ]
    duplicate_payments["payment_date"] = (
        pd.to_datetime(duplicate_payments["payment_date"])
        + pd.Timedelta(days=2)
    )
    duplicate_payments["payment_status"] = "Paid"
    duplicate_payments["reversal_reference"] = None
    payments_synthetic = pd.concat(
        [payments_synthetic, duplicate_payments], ignore_index=True
    )
    duplicate_payment_labels = pd.DataFrame({
        "entity_id": duplicate_payments["payment_id"].to_numpy(),
        "entity_type": "payment",
        "fraud_label": 1,
        "scenario_id": "S001",
        "label_source": "Synthetic scenario injection",
    })

    duplicate_candidates = claims_base.loc[
        claims_base["claim_status"].isin(["Approved", "Declined"])
    ].copy()
    selected_claims = duplicate_candidates.sample(
        n=min(N_DUPLICATE_CLAIMS, len(duplicate_candidates)),
        random_state=base_seed + 12,
    ).copy()
    duplicate_claims = selected_claims.copy()
    duplicate_claims["claim_id"] = [
        f"FDC{i:07d}" for i in range(1, len(duplicate_claims) + 1)
    ]
    duplicate_claims["submission_date"] = (
        pd.to_datetime(duplicate_claims["submission_date"])
        + pd.Timedelta(days=1)
    )
    duplicate_claims["claim_status"] = "Under Review"
    duplicate_claims["decision_date"] = pd.NaT
    duplicate_claims["approved_amount"] = np.nan
    claims_synthetic = pd.concat(
        [claims_synthetic, duplicate_claims], ignore_index=True
    )
    duplicate_claim_labels = pd.DataFrame({
        "entity_id": duplicate_claims["claim_id"].to_numpy(),
        "entity_type": "claim",
        "fraud_label": 1,
        "scenario_id": "S002",
        "label_source": "Synthetic scenario injection",
    })

    base_frequency_claim = claims_base.sample(
        n=1, random_state=base_seed + 13
    ).iloc[0]
    frequency_claims = pd.DataFrame(
        [base_frequency_claim.to_dict() for _ in range(N_FREQUENCY_CLAIMS)]
    )
    frequency_claims["claim_id"] = [
        f"FREQ{i:07d}" for i in range(1, N_FREQUENCY_CLAIMS + 1)
    ]
    cluster_start = pd.Timestamp("2025-06-01")
    frequency_claims["incident_date"] = cluster_start - pd.Timedelta(days=2)
    frequency_claims["submission_date"] = [
        cluster_start + pd.Timedelta(days=i)
        for i in range(N_FREQUENCY_CLAIMS)
    ]
    frequency_claims["claim_status"] = "Under Review"
    frequency_claims["decision_date"] = pd.NaT
    frequency_claims["approved_amount"] = np.nan
    claims_synthetic = pd.concat(
        [claims_synthetic, frequency_claims], ignore_index=True
    )
    frequency_labels = pd.DataFrame({
        "entity_id": frequency_claims["claim_id"].to_numpy(),
        "entity_type": "claim",
        "fraud_label": 1,
        "scenario_id": "S003",
        "label_source": "Synthetic scenario injection",
    })

    provider_claim_counts = claims_base["provider_id"].value_counts()
    eligible_providers = provider_claim_counts.index[provider_claim_counts > 0]
    if len(eligible_providers) == 0:
        raise ValueError("No providers have associated claims.")

    target_provider_id = eligible_providers[
        np.argmax(provider_claim_counts.loc[eligible_providers].to_numpy())
    ]
    provider_base_claim = claims_base.loc[
        claims_base["provider_id"].eq(target_provider_id)
    ].iloc[0]
    provider_anomaly_claims = pd.DataFrame(
        [provider_base_claim.to_dict() for _ in range(N_PROVIDER_ANOMALIES)]
    )
    provider_anomaly_claims["claim_id"] = [
        f"PBILL{i:07d}" for i in range(1, N_PROVIDER_ANOMALIES + 1)
    ]
    provider_anomaly_claims["incident_date"] = pd.Timestamp("2025-08-01")
    provider_anomaly_claims["submission_date"] = [
        pd.Timestamp("2025-08-02") + pd.Timedelta(days=i)
        for i in range(N_PROVIDER_ANOMALIES)
    ]
    provider_anomaly_claims["claimed_amount"] = np.round(
        provider_anomaly_claims["claimed_amount"] * 5, 2
    )
    provider_anomaly_claims["claim_status"] = "Under Review"
    provider_anomaly_claims["decision_date"] = pd.NaT
    provider_anomaly_claims["approved_amount"] = np.nan
    claims_synthetic = pd.concat(
        [claims_synthetic, provider_anomaly_claims], ignore_index=True
    )
    provider_anomaly_labels = pd.DataFrame({
        "entity_id": provider_anomaly_claims["claim_id"].to_numpy(),
        "entity_type": "claim",
        "fraud_label": 1,
        "scenario_id": "S004",
        "label_source": "Synthetic scenario injection",
    })

    baseline_claim_labels = pd.DataFrame({
        "entity_id": claims_base["claim_id"].to_numpy(),
        "entity_type": "claim",
        "fraud_label": 0,
        "scenario_id": "BASELINE",
        "label_source": "Synthetic baseline generation",
    })
    baseline_payment_labels = pd.DataFrame({
        "entity_id": payments_base["payment_id"].to_numpy(),
        "entity_type": "payment",
        "fraud_label": 0,
        "scenario_id": "BASELINE",
        "label_source": "Synthetic baseline generation",
    })

    fraud_labels = pd.concat(
        [
            baseline_claim_labels,
            baseline_payment_labels,
            duplicate_payment_labels,
            duplicate_claim_labels,
            frequency_labels,
            provider_anomaly_labels,
        ],
        ignore_index=True,
    )

    assert not fraud_labels.duplicated(["entity_id", "entity_type"]).any()
    assert fraud_labels["fraud_label"].isin([0, 1]).all()

    _validate_scenarios(
        claims_synthetic,
        payments_synthetic,
        people_base,
        providers_base,
        fraud_labels,
        claims_base,
        payments_base,
        duplicate_claims,
        frequency_claims,
        provider_anomaly_claims,
        duplicate_payments,
    )

    scenario_registry = build_scenario_registry()
    return {
        "claims_synthetic": claims_synthetic,
        "payments_synthetic": payments_synthetic,
        "fraud_labels": fraud_labels,
        "scenario_registry": scenario_registry,
        "people_base": people_base,
        "providers_base": providers_base,
        "claims_base": claims_base,
        "payments_base": payments_base,
        "target_provider_id": target_provider_id,
    }

def _validate_scenarios(
    claims_synthetic,
    payments_synthetic,
    people_base,
    providers_base,
    fraud_labels,
    claims_base,
    payments_base,
    duplicate_claims,
    frequency_claims,
    provider_anomaly_claims,
    duplicate_payments,
):
    assert claims_synthetic["claim_id"].is_unique
    assert payments_synthetic["payment_id"].is_unique
    assert claims_synthetic["person_id"].isin(people_base["person_id"]).all()
    assert claims_synthetic["provider_id"].isin(
        providers_base["provider_id"]
    ).all()
    assert payments_synthetic["claim_id"].isin(
        claims_synthetic["claim_id"]
    ).all()

    injected_claim_ids = set(
        pd.concat(
            [
                duplicate_claims["claim_id"],
                frequency_claims["claim_id"],
                provider_anomaly_claims["claim_id"],
            ],
            ignore_index=True,
        )
    )
    injected_claim_records = claims_synthetic.loc[
        claims_synthetic["claim_id"].isin(injected_claim_ids)
    ]
    assert injected_claim_records["claim_status"].eq("Under Review").all()
    assert injected_claim_records["approved_amount"].isna().all()

    claim_label_ids = set(
        fraud_labels.loc[
            fraud_labels["entity_type"].eq("claim"), "entity_id"
        ]
    )
    payment_label_ids = set(
        fraud_labels.loc[
            fraud_labels["entity_type"].eq("payment"), "entity_id"
        ]
    )
    assert set(claims_synthetic["claim_id"]) == claim_label_ids
    assert set(payments_synthetic["payment_id"]) == payment_label_ids

    assert len(claims_synthetic) == (
        len(claims_base)
        + len(duplicate_claims)
        + len(frequency_claims)
        + len(provider_anomaly_claims)
    )
    assert len(payments_synthetic) == (
        len(payments_base) + len(duplicate_payments)
    )

def save_scenario_datasets(result, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    result["claims_synthetic"].to_csv(
        data_dir / "hoto_claims_synthetic.csv", index=False
    )
    result["payments_synthetic"].to_csv(
        data_dir / "hoto_payments_synthetic.csv", index=False
    )
    result["fraud_labels"].to_csv(
        data_dir / "hoto_fraud_labels.csv", index=False
    )
    result["scenario_registry"].to_csv(
        data_dir / "hoto_scenario_registry.csv", index=False
    )
