"""Unified risk layer, investigation queue, and Model Lab artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from hoto.config import CLASSIFIER_THRESHOLD, REVIEW_CAPACITIES
from hoto.explain import explain_review


def build_unified_risk_results(
    payment_model_results,
    claim_model_data,
    claim_test_idx,
    claim_fraud_probability,
    claim_predictions,
    claim_anomaly_scores,
    claim_anomaly_flags,
    data_dir,
):
    payment_results = payment_model_results.copy()
    payment_results["record_type"] = "payment"
    payment_results["record_id"] = payment_results["payment_id"]

    if len(claim_test_idx) != len(claim_fraud_probability):
        raise ValueError(
            f"Claim test indices ({len(claim_test_idx)}) do not match "
            f"claim predictions ({len(claim_fraud_probability)})."
        )
    if not (
        len(claim_predictions)
        == len(claim_anomaly_scores)
        == len(claim_anomaly_flags)
        == len(claim_fraud_probability)
    ):
        raise ValueError("The claim prediction arrays have different lengths.")

    claim_results = claim_model_data.iloc[claim_test_idx][
        ["claim_id", "fraud_label", "scenario_id"]
    ].copy().reset_index(drop=True)

    claim_results["fraud_probability"] = claim_fraud_probability
    claim_results["anomaly_score"] = claim_anomaly_scores
    claim_results["classifier_flag"] = claim_predictions
    claim_results["anomaly_flag"] = claim_anomaly_flags
    claim_results["record_type"] = "claim"
    claim_results["record_id"] = claim_results["claim_id"]

    common_columns = [
        "record_type",
        "record_id",
        "claim_id",
        "fraud_label",
        "scenario_id",
        "fraud_probability",
        "anomaly_score",
        "classifier_flag",
        "anomaly_flag",
    ]
    payment_results = payment_results[common_columns]
    claim_results = claim_results[common_columns]

    unified_risk_results = pd.concat(
        [claim_results, payment_results], ignore_index=True
    )
    unified_risk_results["fraud_probability"] = (
        unified_risk_results["fraud_probability"].astype(float)
    )
    unified_risk_results["classifier_rank"] = (
        unified_risk_results.groupby("record_type")["fraud_probability"]
        .rank(method="first", ascending=False)
    )
    unified_risk_results["anomaly_rank"] = (
        unified_risk_results.groupby("record_type")["anomaly_score"]
        .rank(method="first", ascending=False)
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    unified_risk_results.to_csv(
        data_dir / "unified_risk_results.csv", index=False
    )
    return unified_risk_results


def build_investigation_queue(unified_risk_results, data_dir):
    investigation_queue = unified_risk_results.copy()

    investigation_queue["review_reason"] = investigation_queue.apply(
        lambda row: (
            "Classifier and anomaly detection"
            if row["classifier_flag"] == 1 and row["anomaly_flag"] == 1
            else "Classifier flagged"
            if row["classifier_flag"] == 1
            else "Anomaly detection flagged"
            if row["anomaly_flag"] == 1
            else "Not flagged"
        ),
        axis=1,
    )
    investigation_queue["review_priority"] = investigation_queue.apply(
        lambda row: (
            "High"
            if row["classifier_flag"] == 1 and row["anomaly_flag"] == 1
            else "Medium"
            if row["classifier_flag"] == 1 or row["anomaly_flag"] == 1
            else "Not flagged"
        ),
        axis=1,
    )

    investigation_queue = investigation_queue[
        (investigation_queue["classifier_flag"] == 1)
        | (investigation_queue["anomaly_flag"] == 1)
    ].copy()

    priority_order = {"High": 0, "Medium": 1}
    investigation_queue["priority_order"] = (
        investigation_queue["review_priority"].map(priority_order)
    )
    investigation_queue = investigation_queue.sort_values(
        ["priority_order", "fraud_probability"],
        ascending=[True, False],
    ).drop(columns="priority_order")

    investigation_queue["review_explanation"] = (
        investigation_queue.apply(explain_review, axis=1)
    )

    investigator_view = investigation_queue[
        [
            "record_type",
            "record_id",
            "claim_id",
            "review_priority",
            "review_reason",
            "fraud_probability",
            "anomaly_score",
            "review_explanation",
            "fraud_label",
            "scenario_id",
        ]
    ].copy()
    investigator_view["fraud_probability"] = (
        investigator_view["fraud_probability"].round(4)
    )
    investigator_view["anomaly_score"] = (
        investigator_view["anomaly_score"].round(4)
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    investigation_queue.to_csv(
        data_dir / "investigation_queue.csv", index=False
    )
    investigator_view.to_csv(
        data_dir / "investigator_view.csv", index=False
    )
    return investigation_queue, investigator_view


def _model_comparison_table(model_probabilities, y_true, threshold=CLASSIFIER_THRESHOLD):
    rows = []
    for model_name, probabilities in model_probabilities.items():
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()
        rows.append({
            "Model": model_name,
            "Average Precision": average_precision_score(y_true, probabilities),
            "ROC-AUC": roc_auc_score(y_true, probabilities),
            "Precision": precision_score(y_true, predicted, zero_division=0),
            "Recall": recall_score(y_true, predicted, zero_division=0),
            "F1": f1_score(y_true, predicted, zero_division=0),
            "True Positives": int(tp),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Negatives": int(tn),
            "Flagged Records": int(predicted.sum()),
        })
    return pd.DataFrame(rows)


def _capacity_table(model_probabilities, y_true, capacities=REVIEW_CAPACITIES):
    rows = []
    for model_name, probabilities in model_probabilities.items():
        order = np.argsort(-probabilities)
        for capacity in capacities:
            selected = order[: min(capacity, len(order))]
            detected = int(y_true[selected].sum())
            rows.append({
                "Model": model_name,
                "Review Capacity": capacity,
                "Fraud Cases in Top Records": detected,
                "Precision at Capacity": detected / len(selected),
                "Recall at Capacity": detected / max(int(y_true.sum()), 1),
            })
    return pd.DataFrame(rows)


def build_payment_model_lab(
    payment_model_data,
    X_test,
    y_test,
    fraud_probabilities,
    xgb_probabilities,
    data_dir,
):
    y_true = y_test.to_numpy(dtype=int)
    test_metadata = payment_model_data.loc[X_test.index].copy()
    assert test_metadata.index.equals(X_test.index)
    assert np.array_equal(
        test_metadata["fraud_label"].to_numpy(dtype=int), y_true
    )

    model_probabilities = {
        "Logistic Regression": fraud_probabilities,
        "XGBoost": xgb_probabilities,
    }
    payment_model_predictions = test_metadata[
        ["payment_id", "claim_id", "fraud_label", "scenario_id"]
    ].copy()
    payment_model_predictions["logistic_probability"] = fraud_probabilities
    payment_model_predictions["xgboost_probability"] = xgb_probabilities

    model_comparison = _model_comparison_table(model_probabilities, y_true)
    review_capacity_comparison = _capacity_table(model_probabilities, y_true)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(
        data_dir / "payment_model_comparison.csv", index=False
    )
    review_capacity_comparison.to_csv(
        data_dir / "payment_review_capacity_comparison.csv", index=False
    )
    payment_model_predictions.to_csv(
        data_dir / "payment_model_predictions.csv", index=False
    )
    return {
        "model_comparison": model_comparison,
        "review_capacity_comparison": review_capacity_comparison,
        "payment_model_predictions": payment_model_predictions,
    }


def build_claim_model_lab(
    claim_model_data,
    X_claim_test,
    y_claim_test,
    claim_lr_probabilities,
    claim_xgb_probabilities,
    data_dir,
):
    y_claim_test_array = np.asarray(y_claim_test).astype(int)
    claim_models = {
        "Logistic Regression": claim_lr_probabilities,
        "XGBoost": claim_xgb_probabilities,
    }

    claim_model_comparison = _model_comparison_table(
        claim_models, y_claim_test_array
    )
    claim_capacity = _capacity_table(claim_models, y_claim_test_array)
    claim_capacity = claim_capacity.rename(columns={
        "Fraud Cases in Top Records": "Positive Cases in Top Records",
    })

    claim_test_metadata = claim_model_data.loc[X_claim_test.index]
    assert np.array_equal(
        claim_test_metadata["fraud_label"].to_numpy(dtype=int),
        y_claim_test_array,
    )
    claim_model_predictions = claim_test_metadata[
        ["claim_id", "fraud_label", "scenario_id"]
    ].copy()
    claim_model_predictions["logistic_probability"] = claim_lr_probabilities
    claim_model_predictions["xgboost_probability"] = claim_xgb_probabilities

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    claim_model_comparison.to_csv(
        data_dir / "claim_model_comparison.csv", index=False
    )
    claim_capacity.to_csv(
        data_dir / "claim_review_capacity_comparison.csv", index=False
    )
    claim_model_predictions.to_csv(
        data_dir / "claim_model_predictions.csv", index=False
    )
    return {
        "claim_model_comparison": claim_model_comparison,
        "claim_review_capacity_comparison": claim_capacity,
        "claim_model_predictions": claim_model_predictions,
    }


def consolidate_model_lab(
    payment_lab,
    claim_lab,
    data_dir,
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    payment_comparison = payment_lab["model_comparison"].copy()
    payment_comparison.insert(0, "Record Type", "Payment")
    claim_comparison = claim_lab["claim_model_comparison"].copy()
    claim_comparison.insert(0, "Record Type", "Claim")
    model_lab_comparison = pd.concat(
        [payment_comparison, claim_comparison], ignore_index=True
    )
    model_lab_comparison.to_csv(
        data_dir / "model_lab_comparison.csv", index=False
    )

    payment_capacity = payment_lab["review_capacity_comparison"].copy()
    payment_capacity.insert(0, "Record Type", "Payment")
    claim_capacity = claim_lab["claim_review_capacity_comparison"].copy()
    claim_capacity.insert(0, "Record Type", "Claim")
    model_lab_capacity = pd.concat(
        [payment_capacity, claim_capacity], ignore_index=True
    )
    model_lab_capacity.to_csv(
        data_dir / "model_lab_review_capacity.csv", index=False
    )

    payment_predictions = payment_lab["payment_model_predictions"].copy()
    payment_predictions["record_type"] = "payment"
    payment_predictions["record_id"] = payment_predictions["payment_id"]

    claim_predictions_df = claim_lab["claim_model_predictions"].copy()
    claim_predictions_df["record_type"] = "claim"
    claim_predictions_df["record_id"] = claim_predictions_df["claim_id"]
    claim_predictions_df["payment_id"] = pd.NA

    prediction_columns = [
        "record_type",
        "record_id",
        "claim_id",
        "payment_id",
        "fraud_label",
        "scenario_id",
        "logistic_probability",
        "xgboost_probability",
    ]
    payment_predictions = payment_predictions.reindex(columns=prediction_columns)
    claim_predictions_df = claim_predictions_df.reindex(columns=prediction_columns)
    model_lab_predictions = pd.concat(
        [claim_predictions_df, payment_predictions], ignore_index=True
    )
    model_lab_predictions.to_csv(
        data_dir / "model_lab_predictions.csv", index=False
    )
    return {
        "model_lab_comparison": model_lab_comparison,
        "model_lab_capacity": model_lab_capacity,
        "model_lab_predictions": model_lab_predictions,
    }


def build_curve_artifacts(
    y_test,
    payment_model_predictions,
    y_claim_test,
    claim_model_predictions,
    data_dir,
):
    curve_models = [
        {
            "record_type": "Payment",
            "model": "Logistic Regression",
            "y_true": y_test.to_numpy(dtype=int),
            "scores": payment_model_predictions[
                "logistic_probability"
            ].to_numpy(),
        },
        {
            "record_type": "Payment",
            "model": "XGBoost",
            "y_true": y_test.to_numpy(dtype=int),
            "scores": payment_model_predictions[
                "xgboost_probability"
            ].to_numpy(),
        },
        {
            "record_type": "Claim",
            "model": "Logistic Regression",
            "y_true": y_claim_test.to_numpy(dtype=int),
            "scores": claim_model_predictions[
                "logistic_probability"
            ].to_numpy(),
        },
        {
            "record_type": "Claim",
            "model": "XGBoost",
            "y_true": y_claim_test.to_numpy(dtype=int),
            "scores": claim_model_predictions[
                "xgboost_probability"
            ].to_numpy(),
        },
    ]

    roc_rows = []
    pr_rows = []
    curve_summary_rows = []

    for item in curve_models:
        y_true = item["y_true"]
        scores = item["scores"]
        assert len(y_true) == len(scores)

        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        for fpr_value, tpr_value, threshold_value in zip(
            fpr, tpr, roc_thresholds
        ):
            roc_rows.append({
                "record_type": item["record_type"],
                "model": item["model"],
                "false_positive_rate": float(fpr_value),
                "true_positive_rate": float(tpr_value),
                "threshold": float(threshold_value),
            })

        precision, recall, pr_thresholds = precision_recall_curve(
            y_true, scores
        )
        for index, (precision_value, recall_value) in enumerate(
            zip(precision, recall)
        ):
            threshold_value = (
                float(pr_thresholds[index])
                if index < len(pr_thresholds)
                else np.nan
            )
            pr_rows.append({
                "record_type": item["record_type"],
                "model": item["model"],
                "precision": float(precision_value),
                "recall": float(recall_value),
                "threshold": threshold_value,
            })

        curve_summary_rows.append({
            "record_type": item["record_type"],
            "model": item["model"],
            "roc_auc_from_curve": float(auc(fpr, tpr)),
            "average_precision": float(
                average_precision_score(y_true, scores)
            ),
            "test_records": len(y_true),
            "positive_cases": int(y_true.sum()),
        })

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    roc_curve_data = pd.DataFrame(roc_rows)
    pr_curve_data = pd.DataFrame(pr_rows)
    curve_summary = pd.DataFrame(curve_summary_rows)

    roc_curve_data.to_csv(data_dir / "model_lab_roc_curves.csv", index=False)
    pr_curve_data.to_csv(
        data_dir / "model_lab_precision_recall_curves.csv", index=False
    )
    curve_summary.to_csv(
        data_dir / "model_lab_curve_summary.csv", index=False
    )
    return {
        "roc_curve_data": roc_curve_data,
        "pr_curve_data": pr_curve_data,
        "curve_summary": curve_summary,
    }
