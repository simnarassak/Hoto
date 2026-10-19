"""Synthetic population, claims, payments, and event generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from hoto.config import (
    AGE_BANDS,
    CLAIM_AMOUNT_RANGES,
    CLAIM_STATUS_PROBS,
    CLAIM_STATUSES,
    CLAIM_TYPE_PROBS,
    CLAIM_TYPES,
    EMPLOYMENT_DISTRIBUTION,
    IMMIGRANT_STATUS_DISTRIBUTION,
    N_CLAIMS,
    N_PROVIDERS,
    POPULATION_GROUPS,
    PROVIDER_TYPE_PROBS,
    PROVIDER_TYPES,
    REGIONS,
    WORLD,
)

def _exact_category_values(n_people, distribution, rng):
    values = []
    for label, proportion in distribution.items():
        count = round(n_people * proportion)
        values.extend([label] * count)

    if len(values) > n_people:
        values = values[:n_people]
    elif len(values) < n_people:
        values.extend([list(distribution.keys())[-1]] * (n_people - len(values)))

    rng.shuffle(values)
    return values

def generate_population(seed=None):
    seed = WORLD["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    n_people = WORLD["population"]

    people = pd.DataFrame({
        "person_id": [f"HOTO{i:06d}" for i in range(1, n_people + 1)],
    })

    people["population_group"] = _exact_category_values(
        n_people, POPULATION_GROUPS, rng
    )
    people["region"] = _exact_category_values(n_people, REGIONS, rng)
    people["age_band"] = _exact_category_values(n_people, AGE_BANDS, rng)

    people["current_status"] = "Citizen"
    immigrant_mask = people["population_group"].eq("Immigrant")
    immigrant_count = int(immigrant_mask.sum())
    status_values = _exact_category_values(
        immigrant_count, IMMIGRANT_STATUS_DISTRIBUTION, rng
    )
    people.loc[immigrant_mask, "current_status"] = status_values

    people["employment_status"] = _exact_category_values(
        n_people, EMPLOYMENT_DISTRIBUTION, rng
    )

    _validate_population(people)
    return people

def _validate_population(people):
    n_people = WORLD["population"]
    assert len(people) == n_people
    assert people["person_id"].is_unique
    assert people.notna().all().all()

    expected_groups = {
        group: round(n_people * proportion)
        for group, proportion in POPULATION_GROUPS.items()
    }
    actual_groups = people["population_group"].value_counts().to_dict()
    assert sum(actual_groups.values()) == n_people

    expected_regions = {
        region: round(n_people * proportion)
        for region, proportion in REGIONS.items()
    }
    actual_regions = people["region"].value_counts().to_dict()
    assert sum(actual_regions.values()) == n_people

    assert people.loc[
        people["population_group"].eq("Immigrant"),
        "current_status",
    ].isin(IMMIGRANT_STATUS_DISTRIBUTION.keys()).all()
    assert people["region"].isin(REGIONS.keys()).all()
    assert people["age_band"].isin(AGE_BANDS.keys()).all()

    if sum(expected_groups.values()) == n_people:
        assert actual_groups == expected_groups
    if sum(expected_regions.values()) == n_people:
        assert actual_regions == expected_regions

def generate_providers(rng):
    providers = pd.DataFrame({
        "provider_id": [f"PROV{i:05d}" for i in range(1, N_PROVIDERS + 1)],
    })
    providers["region"] = rng.choice(
        list(REGIONS.keys()), size=N_PROVIDERS, p=list(REGIONS.values())
    )
    providers["provider_type"] = rng.choice(
        PROVIDER_TYPES, size=N_PROVIDERS, p=PROVIDER_TYPE_PROBS
    )
    providers["registration_date"] = (
        pd.Timestamp("2020-01-01")
        + pd.to_timedelta(rng.integers(0, 1826, size=N_PROVIDERS), unit="D")
    )
    providers["service_capacity"] = rng.integers(50, 2001, size=N_PROVIDERS)
    providers["provider_status"] = rng.choice(
        ["Active", "Inactive"], size=N_PROVIDERS, p=[0.95, 0.05]
    )
    return providers

def generate_claims(people, providers, rng):

    claimant_ids = rng.choice(
        people["person_id"].to_numpy(), size=N_CLAIMS, replace=True
    )
    claims = pd.DataFrame({
        "claim_id": [f"CLM{i:07d}" for i in range(1, N_CLAIMS + 1)],
        "person_id": claimant_ids,
    })

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    date_range_days = (end_date - start_date).days + 1

    claims["incident_date"] = start_date + pd.to_timedelta(
        rng.integers(0, date_range_days, size=N_CLAIMS), unit="D"
    )
    submission_delay = rng.integers(1, 61, size=N_CLAIMS)
    claims["submission_date"] = claims["incident_date"] + pd.to_timedelta(
        submission_delay, unit="D"
    )
    claims["claim_type"] = rng.choice(
        CLAIM_TYPES, size=N_CLAIMS, p=CLAIM_TYPE_PROBS
    )

    amounts = np.empty(N_CLAIMS)
    for claim_type, (low, high) in CLAIM_AMOUNT_RANGES.items():
        mask = claims["claim_type"].eq(claim_type)
        count = int(mask.sum())
        amounts[mask.to_numpy()] = np.round(rng.uniform(low, high, size=count), 2)
    claims["claimed_amount"] = amounts

    people_lookup = people[["person_id", "region"]].rename(
        columns={"region": "claimant_region"}
    )
    claims = claims.merge(
        people_lookup, on="person_id", how="left", validate="many_to_one"
    )
    assert claims["claimant_region"].notna().all()

    provider_ids_by_region = {
        region: providers.loc[
            providers["region"].eq(region), "provider_id"
        ].to_numpy()
        for region in REGIONS
    }
    all_provider_ids = providers["provider_id"].to_numpy()

    selected_providers = np.empty(N_CLAIMS, dtype=object)
    for region, regional_ids in provider_ids_by_region.items():
        mask = claims["claimant_region"].eq(region).to_numpy()
        count = int(mask.sum())
        if count == 0:
            continue
        pool = regional_ids if len(regional_ids) > 0 else all_provider_ids
        selected_providers[mask] = rng.choice(pool, size=count)
    claims["provider_id"] = selected_providers

    claims["claim_status"] = rng.choice(
        CLAIM_STATUSES, size=N_CLAIMS, p=CLAIM_STATUS_PROBS
    )

    decided_mask = claims["claim_status"].isin(["Approved", "Declined"])
    decision_delay = rng.integers(1, 31, size=int(decided_mask.sum()))
    claims["decision_date"] = pd.NaT
    claims.loc[decided_mask, "decision_date"] = (
        claims.loc[decided_mask, "submission_date"].reset_index(drop=True)
        + pd.to_timedelta(decision_delay, unit="D")
    ).to_numpy()

    claims["submission_year"] = claims["submission_date"].dt.year
    _validate_claims(claims, people, providers)
    return claims

def _validate_claims(claims, people, providers):
    assert claims["claim_id"].is_unique
    assert providers["provider_id"].is_unique
    assert claims["person_id"].isin(people["person_id"]).all()
    assert claims["provider_id"].isin(providers["provider_id"]).all()
    assert claims["claimed_amount"].gt(0).all()
    assert (claims["submission_date"] > claims["incident_date"]).all()

    decided = claims["claim_status"].isin(["Approved", "Declined"])
    assert claims.loc[decided, "decision_date"].notna().all()
    assert claims.loc[~decided, "decision_date"].isna().all()
    assert (
        claims.loc[decided, "decision_date"]
        >= claims.loc[decided, "submission_date"]
    ).all()

def generate_payments(claims, seed=None):
    seed = WORLD["random_seed"] + 2 if seed is None else seed
    payment_rng = np.random.default_rng(seed)

    claims = claims.copy()
    claims["approved_amount"] = np.nan

    approved_mask = claims["claim_status"].eq("Approved")
    approved_count = int(approved_mask.sum())
    approval_factors = payment_rng.uniform(0.70, 1.00, size=approved_count)
    claims.loc[approved_mask, "approved_amount"] = np.round(
        claims.loc[approved_mask, "claimed_amount"].to_numpy() * approval_factors,
        2,
    )

    assert claims.loc[approved_mask, "approved_amount"].notna().all()
    assert claims.loc[approved_mask, "approved_amount"].gt(0).all()
    assert (
        claims.loc[approved_mask, "approved_amount"]
        <= claims.loc[approved_mask, "claimed_amount"]
    ).all()

    payment_rows = []
    payment_counter = 1
    approved_claims = claims.loc[
        claims["claim_status"].eq("Approved"),
        ["claim_id", "approved_amount", "decision_date"],
    ]

    for claim in approved_claims.itertuples(index=False):
        total_amount = round(float(claim.approved_amount), 2)
        n_installments = int(payment_rng.integers(1, 4))

        if n_installments == 1:
            installments = [total_amount]
        else:
            total_cents = int(round(total_amount * 100))
            cut_points = sorted(
                payment_rng.choice(
                    np.arange(1, total_cents),
                    size=n_installments - 1,
                    replace=False,
                )
            )
            boundaries = [0] + cut_points + [total_cents]
            installments = [
                (boundaries[i + 1] - boundaries[i]) / 100
                for i in range(n_installments)
            ]

        for installment_number, amount in enumerate(installments, start=1):
            days_after_decision = int(payment_rng.integers(1, 16))
            payment_date = (
                pd.Timestamp(claim.decision_date)
                + pd.Timedelta(days=days_after_decision)
            )
            payment_rows.append({
                "payment_id": f"PAY{payment_counter:08d}",
                "claim_id": claim.claim_id,
                "installment_number": installment_number,
                "payment_date": payment_date,
                "payment_amount": round(amount, 2),
                "payment_status": payment_rng.choice(
                    ["Paid", "Pending", "Failed"],
                    p=[0.90, 0.07, 0.03],
                ),
                "reversal_reference": None,
            })
            payment_counter += 1

    payments = pd.DataFrame(payment_rows)
    if payments.empty:
        payments = pd.DataFrame(columns=[
            "payment_id", "claim_id", "installment_number",
            "payment_date", "payment_amount", "payment_status",
            "reversal_reference",
        ])
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])

    paid_payments = payments.loc[payments["payment_status"].eq("Paid")]
    n_reversals = int(round(len(paid_payments) * 0.01))
    if n_reversals > 0:
        reversed_originals = paid_payments.sample(
            n=n_reversals,
            random_state=WORLD["random_seed"] + 3,
        )
        reversal_rows = []
        for i, payment in enumerate(
            reversed_originals.itertuples(index=False), start=1
        ):
            reversal_rows.append({
                "payment_id": f"REV{i:08d}",
                "claim_id": payment.claim_id,
                "installment_number": payment.installment_number,
                "payment_date": (
                    pd.Timestamp(payment.payment_date)
                    + pd.Timedelta(days=int(payment_rng.integers(1, 31)))
                ),
                "payment_amount": -round(float(payment.payment_amount), 2),
                "payment_status": "Reversal",
                "reversal_reference": payment.payment_id,
            })
        payments = pd.concat(
            [payments, pd.DataFrame(reversal_rows)],
            ignore_index=True,
        )

    payments["payment_date"] = pd.to_datetime(payments["payment_date"])
    return claims, payments

def generate_claim_events(claims, payments):
    event_rows = []
    event_counter = 1

    for claim in claims.itertuples(index=False):
        event_rows.append({
            "event_id": f"EVT{event_counter:09d}",
            "claim_id": claim.claim_id,
            "event_timestamp": pd.Timestamp(claim.submission_date),
            "event_type": "Claim Submitted",
            "previous_status": None,
            "new_status": "Submitted",
            "event_source": "Synthetic Claims System",
            "payment_id": None,
        })
        event_counter += 1

        assessment_date = (
            pd.Timestamp(claim.submission_date) + pd.Timedelta(days=1)
        )
        event_rows.append({
            "event_id": f"EVT{event_counter:09d}",
            "claim_id": claim.claim_id,
            "event_timestamp": assessment_date,
            "event_type": "Assessment Started",
            "previous_status": "Submitted",
            "new_status": "Under Review",
            "event_source": "Synthetic Claims System",
            "payment_id": None,
        })
        event_counter += 1

        if pd.notna(claim.decision_date):
            event_rows.append({
                "event_id": f"EVT{event_counter:09d}",
                "claim_id": claim.claim_id,
                "event_timestamp": pd.Timestamp(claim.decision_date),
                "event_type": "Claim Decision",
                "previous_status": "Under Review",
                "new_status": claim.claim_status,
                "event_source": "Synthetic Claims System",
                "payment_id": None,
            })
            event_counter += 1

    claim_events = pd.DataFrame(event_rows)
    claim_events["event_timestamp"] = pd.to_datetime(
        claim_events["event_timestamp"]
    )

    payment_event_rows = []
    for i, payment in enumerate(payments.itertuples(index=False), start=1):
        if payment.payment_status == "Reversal":
            event_type, new_status = "Payment Reversed", "Reversed"
        elif payment.payment_status == "Paid":
            event_type, new_status = "Payment Completed", "Paid"
        elif payment.payment_status == "Pending":
            event_type, new_status = "Payment Pending", "Pending"
        else:
            event_type, new_status = "Payment Failed", "Failed"

        payment_event_rows.append({
            "event_id": f"PEV{i:09d}",
            "claim_id": payment.claim_id,
            "event_timestamp": payment.payment_date,
            "event_type": event_type,
            "previous_status": None,
            "new_status": new_status,
            "event_source": "Synthetic Payment System",
            "payment_id": payment.payment_id,
        })

    payment_events = pd.DataFrame(payment_event_rows)
    claim_events = pd.concat(
        [claim_events, payment_events], ignore_index=True
    )
    claim_events["event_timestamp"] = pd.to_datetime(
        claim_events["event_timestamp"]
    )
    claim_events = claim_events.sort_values(
        ["claim_id", "event_timestamp", "event_id"]
    ).reset_index(drop=True)
    return claim_events

def validate_payments_and_events(claims, payments, claim_events):
    assert payments["payment_id"].is_unique
    assert claim_events["event_id"].is_unique
    assert payments["claim_id"].isin(claims["claim_id"]).all()
    assert claim_events["claim_id"].isin(claims["claim_id"]).all()
    assert payments["payment_amount"].ne(0).all()

    assert payments.loc[
        payments["payment_status"].ne("Reversal"), "payment_amount"
    ].gt(0).all()
    assert payments.loc[
        payments["payment_status"].eq("Reversal"), "payment_amount"
    ].lt(0).all()

    reversal_records = payments.loc[payments["payment_status"].eq("Reversal")]
    original_payments = payments.loc[
        payments["payment_status"].ne("Reversal")
    ].set_index("payment_id")

    assert reversal_records["reversal_reference"].notna().all()
    assert reversal_records["reversal_reference"].isin(
        original_payments.index
    ).all()

    for reversal in reversal_records.itertuples(index=False):
        original = original_payments.loc[reversal.reversal_reference]
        assert original["payment_status"] == "Paid"
        assert original["claim_id"] == reversal.claim_id
        assert np.isclose(
            original["payment_amount"], -reversal.payment_amount
        )
        assert reversal.payment_date > original["payment_date"]

    approved_claim_ids = set(
        claims.loc[claims["claim_status"].eq("Approved"), "claim_id"]
    )
    assert set(payments["claim_id"]).issubset(approved_claim_ids)

    net_paid = (
        payments.loc[payments["payment_status"].isin(["Paid", "Reversal"])]
        .groupby("claim_id")["payment_amount"]
        .sum()
    )
    approved_amount_lookup = claims.set_index("claim_id")["approved_amount"]
    for claim_id, net_amount in net_paid.items():
        assert net_amount <= approved_amount_lookup.loc[claim_id] + 0.01

def run_integrated_quality_gate(people, providers, claims, payments, claim_events):
    assert people["person_id"].is_unique
    assert providers["provider_id"].is_unique
    assert claims["claim_id"].is_unique
    assert payments["payment_id"].is_unique
    assert claim_events["event_id"].is_unique

    assert claims["person_id"].isin(people["person_id"]).all()
    assert claims["provider_id"].isin(providers["provider_id"]).all()
    assert payments["claim_id"].isin(claims["claim_id"]).all()
    assert claim_events["claim_id"].isin(claims["claim_id"]).all()

    assert claims["claimed_amount"].gt(0).all()
    assert (claims["submission_date"] > claims["incident_date"]).all()

    approved = claims["claim_status"].eq("Approved")
    assert claims.loc[approved, "approved_amount"].notna().all()
    assert (
        claims.loc[approved, "approved_amount"]
        <= claims.loc[approved, "claimed_amount"]
    ).all()

    reversals = payments.loc[payments["payment_status"].eq("Reversal")]
    originals = payments.loc[
        payments["payment_status"].ne("Reversal")
    ].set_index("payment_id")
    assert reversals["reversal_reference"].notna().all()
    assert reversals["reversal_reference"].isin(originals.index).all()

    for reversal in reversals.itertuples(index=False):
        original = originals.loc[reversal.reversal_reference]
        assert original["claim_id"] == reversal.claim_id
        assert original["payment_status"] == "Paid"
        assert abs(
            original["payment_amount"] + reversal.payment_amount
        ) < 0.01
        assert reversal.payment_date > original["payment_date"]

    approved_ids = set(claims.loc[approved, "claim_id"])
    assert set(payments["claim_id"]).issubset(approved_ids)

    payment_events = claim_events.loc[claim_events["payment_id"].notna()]
    assert payment_events["payment_id"].isin(payments["payment_id"]).all()
    return len(reversals)

def save_baseline_datasets(
    people, providers, claims, payments, claim_events, data_dir
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    people.to_csv(data_dir / "hoto_population.csv", index=False)
    providers.to_csv(data_dir / "hoto_providers.csv", index=False)
    claims.to_csv(data_dir / "hoto_claims.csv", index=False)
    payments.to_csv(data_dir / "hoto_payments.csv", index=False)
    claim_events.to_csv(data_dir / "hoto_claim_events.csv", index=False)

def generate_baseline_world(seed=None, data_dir=None):
    """Generate and persist the clean synthetic world."""
    from hoto.config import DATA_DIR

    base_seed = WORLD["random_seed"] if seed is None else seed
    people = generate_population(seed=base_seed)

    ecosystem_rng = np.random.default_rng(base_seed + 1)
    providers = generate_providers(ecosystem_rng)
    claims = generate_claims(people, providers, ecosystem_rng)

    claims, payments = generate_payments(claims, seed=base_seed + 2)
    claim_events = generate_claim_events(claims, payments)
    validate_payments_and_events(claims, payments, claim_events)
    n_reversals = run_integrated_quality_gate(
        people, providers, claims, payments, claim_events
    )

    out = Path(data_dir) if data_dir is not None else DATA_DIR
    save_baseline_datasets(
        people, providers, claims, payments, claim_events, out
    )
    return {
        "people": people,
        "providers": providers,
        "claims": claims,
        "payments": payments,
        "claim_events": claim_events,
        "n_reversals": n_reversals,
    }
