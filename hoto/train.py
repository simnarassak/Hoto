"""Model training: logistic regression, isolation forest, and XGBoost."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from hoto.config import (
    CLASSIFIER_THRESHOLD,
    TEST_SIZE,
    WORLD,
    XGB_PARAMS,
)
from hoto.features import (
    CLAIM_MODEL_CATEGORICAL,
    CLAIM_MODEL_NUMERIC,
    PAYMENT_MODEL_FEATURES,
)

def _build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

def _classification_metrics(y_true, probabilities, threshold=CLASSIFIER_THRESHOLD):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(y_true, predictions),
        "classification_report": classification_report(
            y_true, predictions, zero_division=0
        ),
    }

def prepare_payment_model_data(payment_features, payment_targets, payment_metadata):
    payment_model_data = (
        payment_features
        .merge(
            payment_targets[["payment_id", "fraud_label", "scenario_id"]],
            on="payment_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            payment_metadata[["payment_id", "claim_id"]],
            on="payment_id",
            how="inner",
            validate="one_to_one",
        )
    )
    assert len(payment_model_data) == len(payment_features)
    assert payment_model_data["fraud_label"].notna().all()

    X = payment_model_data[PAYMENT_MODEL_FEATURES].copy()
    y = payment_model_data["fraud_label"].astype(int)
    groups = payment_model_data["claim_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=WORLD["random_seed"],
    )
    payment_train_idx, payment_test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[payment_train_idx]
    X_test = X.iloc[payment_test_idx]
    y_train = y.iloc[payment_train_idx]
    y_test = y.iloc[payment_test_idx]

    assert set(groups.iloc[payment_train_idx]).isdisjoint(
        set(groups.iloc[payment_test_idx])
    )
    assert y_train.nunique() == 2
    assert y_test.nunique() == 2

    return {
        "payment_model_data": payment_model_data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "payment_train_idx": payment_train_idx,
        "payment_test_idx": payment_test_idx,
    }

def train_payment_models(payment_split, models_dir, data_dir):
    X_train = payment_split["X_train"]
    X_test = payment_split["X_test"]
    y_train = payment_split["y_train"]
    y_test = payment_split["y_test"]
    payment_test_idx = payment_split["payment_test_idx"]
    payment_model_data = payment_split["payment_model_data"]

    numeric_features = [c for c in PAYMENT_MODEL_FEATURES if c != "claim_type"]
    categorical_features = ["claim_type"]
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    fraud_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=WORLD["random_seed"],
        )),
    ])
    fraud_model.fit(X_train, y_train)

    fraud_probabilities = fraud_model.predict_proba(X_test)[:, 1]
    lr_metrics = _classification_metrics(y_test, fraud_probabilities)

    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(X_train, y_train)
    dummy_probabilities = dummy_model.predict_proba(X_test)[:, 1]
    dummy_ap = average_precision_score(y_test, dummy_probabilities)

    X_train_normal = X_train.loc[y_train.eq(0)]
    anomaly_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=WORLD["random_seed"],
            n_jobs=-1,
        )),
    ])
    anomaly_model.fit(X_train_normal)
    anomaly_scores = -anomaly_model.decision_function(X_test)
    anomaly_predictions = (anomaly_model.predict(X_test) == -1).astype(int)

    payment_positive = int(y_train.sum())
    payment_negative = len(y_train) - payment_positive
    if payment_positive == 0:
        raise ValueError("No positive payment training examples were found.")

    xgb_params = {**XGB_PARAMS, "random_state": WORLD["random_seed"]}
    xgb_params["scale_pos_weight"] = payment_negative / payment_positive
    xgb_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", XGBClassifier(**xgb_params)),
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_probabilities = xgb_pipeline.predict_proba(X_test)[:, 1]

    model_results = payment_model_data.iloc[payment_test_idx][
        ["payment_id", "claim_id", "fraud_label", "scenario_id"]
    ].copy()
    model_results["fraud_probability"] = fraud_probabilities
    model_results["anomaly_score"] = anomaly_scores
    model_results["classifier_flag"] = lr_metrics["predictions"]
    model_results["anomaly_flag"] = anomaly_predictions
    model_results = model_results.sort_values(
        "anomaly_score", ascending=False
    ).reset_index(drop=True)

    coefficient_names = (
        fraud_model.named_steps["preprocessor"].get_feature_names_out()
    )
    coefficient_table = pd.DataFrame({
        "feature": coefficient_names,
        "coefficient": fraud_model.named_steps["classifier"].coef_[0],
    })
    coefficient_table["absolute_coefficient"] = (
        coefficient_table["coefficient"].abs()
    )
    coefficient_table = coefficient_table.sort_values(
        "absolute_coefficient", ascending=False
    )

    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        fraud_model,
        models_dir / "hoto_payment_fraud_logistic_regression.joblib",
    )
    joblib.dump(
        anomaly_model,
        models_dir / "hoto_payment_isolation_forest.joblib",
    )
    joblib.dump(xgb_pipeline, models_dir / "payment_xgboost_pipeline.joblib")
    model_results.to_csv(
        data_dir / "hoto_payment_model_predictions.csv", index=False
    )
    coefficient_table.to_csv(
        data_dir / "hoto_payment_model_coefficients.csv", index=False
    )

    return {
        "fraud_model": fraud_model,
        "anomaly_model": anomaly_model,
        "xgb_pipeline": xgb_pipeline,
        "fraud_probabilities": fraud_probabilities,
        "xgb_probabilities": xgb_probabilities,
        "anomaly_scores": anomaly_scores,
        "anomaly_predictions": anomaly_predictions,
        "lr_metrics": lr_metrics,
        "dummy_ap": dummy_ap,
        "model_results": model_results,
        "coefficient_table": coefficient_table,
        "preprocessor": preprocessor,
    }

def prepare_claim_model_data(claim_features, claim_targets, claim_metadata):
    claim_model_data = (
        claim_features
        .merge(
            claim_targets[["claim_id", "fraud_label", "scenario_id"]],
            on="claim_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            claim_metadata[
                ["claim_id", "person_id", "provider_id", "submission_date"]
            ],
            on="claim_id",
            how="inner",
            validate="one_to_one",
        )
    )
    claim_model_data["fraud_label"] = claim_model_data["fraud_label"].astype(int)

    claim_feature_columns = CLAIM_MODEL_CATEGORICAL + CLAIM_MODEL_NUMERIC
    X_claim = claim_model_data[claim_feature_columns].copy()
    y_claim = claim_model_data["fraud_label"].copy()
    claim_groups = claim_model_data["person_id"].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=WORLD["random_seed"],
    )
    claim_train_idx, claim_test_idx = next(
        splitter.split(X_claim, y_claim, groups=claim_groups)
    )

    X_claim_train = X_claim.iloc[claim_train_idx]
    X_claim_test = X_claim.iloc[claim_test_idx]
    y_claim_train = y_claim.iloc[claim_train_idx]
    y_claim_test = y_claim.iloc[claim_test_idx]

    assert set(claim_groups.iloc[claim_train_idx]).isdisjoint(
        set(claim_groups.iloc[claim_test_idx])
    )
    assert y_claim_train.nunique() == 2
    assert y_claim_test.nunique() == 2

    return {
        "claim_model_data": claim_model_data,
        "X_claim_train": X_claim_train,
        "X_claim_test": X_claim_test,
        "y_claim_train": y_claim_train,
        "y_claim_test": y_claim_test,
        "claim_train_idx": claim_train_idx,
        "claim_test_idx": claim_test_idx,
    }

def train_claim_models(claim_split, models_dir, data_dir):
    X_claim_train = claim_split["X_claim_train"]
    X_claim_test = claim_split["X_claim_test"]
    y_claim_train = claim_split["y_claim_train"]
    y_claim_test = claim_split["y_claim_test"]
    claim_test_idx = claim_split["claim_test_idx"]
    claim_model_data = claim_split["claim_model_data"]

    claim_preprocessor = _build_preprocessor(
        CLAIM_MODEL_NUMERIC, CLAIM_MODEL_CATEGORICAL
    )
    claim_fraud_model = Pipeline([
        ("preprocessor", claim_preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=WORLD["random_seed"],
        )),
    ])
    claim_fraud_model.fit(X_claim_train, y_claim_train)

    claim_fraud_probability = claim_fraud_model.predict_proba(X_claim_test)[:, 1]
    claim_lr_metrics = _classification_metrics(
        y_claim_test, claim_fraud_probability
    )

    claim_dummy = DummyClassifier(strategy="prior")
    claim_dummy.fit(X_claim_train, y_claim_train)
    claim_dummy_probability = claim_dummy.predict_proba(X_claim_test)[:, 1]
    claim_dummy_ap = average_precision_score(
        y_claim_test, claim_dummy_probability
    )

    X_claim_train_normal = X_claim_train.loc[y_claim_train.eq(0)].copy()
    claim_anomaly_model = Pipeline([
        ("preprocessor", clone(claim_preprocessor)),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=WORLD["random_seed"],
            n_jobs=-1,
        )),
    ])
    claim_anomaly_model.fit(X_claim_train_normal)
    claim_anomaly_scores = -claim_anomaly_model.decision_function(X_claim_test)
    claim_anomaly_flags = (
        claim_anomaly_model.predict(X_claim_test) == -1
    ).astype(int)

    y_claim_train_array = np.asarray(y_claim_train).astype(int)
    positive_count = int(y_claim_train_array.sum())
    negative_count = len(y_claim_train_array) - positive_count
    if positive_count == 0:
        raise ValueError("No positive claim training examples were found.")

    xgb_params = {**XGB_PARAMS, "random_state": WORLD["random_seed"]}
    xgb_params["scale_pos_weight"] = negative_count / positive_count
    claim_xgb_pipeline = Pipeline([
        ("preprocessor", clone(claim_preprocessor)),
        ("classifier", XGBClassifier(**xgb_params)),
    ])
    claim_xgb_pipeline.fit(X_claim_train, y_claim_train_array)
    claim_xgb_probabilities = claim_xgb_pipeline.predict_proba(X_claim_test)[:, 1]

    claim_results = claim_model_data.iloc[claim_test_idx][
        ["claim_id", "person_id", "provider_id", "fraud_label", "scenario_id"]
    ].copy()
    claim_results["fraud_probability"] = claim_fraud_probability
    claim_results["model_flag"] = claim_lr_metrics["predictions"]
    claim_results = claim_results.sort_values(
        "fraud_probability", ascending=False
    )

    claim_anomaly_results = claim_model_data.iloc[claim_test_idx][
        ["claim_id", "person_id", "provider_id", "fraud_label", "scenario_id"]
    ].copy()
    claim_anomaly_results["anomaly_score"] = claim_anomaly_scores
    claim_anomaly_results["anomaly_flag"] = claim_anomaly_flags
    claim_anomaly_results = claim_anomaly_results.sort_values(
        "anomaly_score", ascending=False
    )

    claim_model_summary = pd.DataFrame([{
        "model": "Claim Logistic Regression",
        "average_precision": claim_lr_metrics["average_precision"],
        "baseline_average_precision": claim_dummy_ap,
        "roc_auc": claim_lr_metrics["roc_auc"],
        "precision_at_0_5": claim_lr_metrics["precision"],
        "recall_at_0_5": claim_lr_metrics["recall"],
        "f1_at_0_5": claim_lr_metrics["f1"],
    }])

    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    claim_results.to_csv(data_dir / "hoto_claim_model_predictions.csv", index=False)
    claim_anomaly_results.to_csv(
        data_dir / "hoto_claim_anomaly_predictions.csv", index=False
    )
    claim_model_summary.to_csv(
        data_dir / "hoto_claim_model_metrics.csv", index=False
    )
    joblib.dump(
        claim_fraud_model,
        models_dir / "hoto_claim_fraud_logistic_regression.joblib",
    )
    joblib.dump(
        claim_anomaly_model,
        models_dir / "hoto_claim_isolation_forest.joblib",
    )
    joblib.dump(
        claim_xgb_pipeline,
        models_dir / "claim_xgboost_pipeline.joblib",
    )

    return {
        "claim_fraud_model": claim_fraud_model,
        "claim_anomaly_model": claim_anomaly_model,
        "claim_xgb_pipeline": claim_xgb_pipeline,
        "claim_fraud_probability": claim_fraud_probability,
        "claim_xgb_probabilities": claim_xgb_probabilities,
        "claim_predictions": claim_lr_metrics["predictions"],
        "claim_anomaly_scores": claim_anomaly_scores,
        "claim_anomaly_flags": claim_anomaly_flags,
        "claim_lr_metrics": claim_lr_metrics,
        "claim_dummy_ap": claim_dummy_ap,
        "claim_results": claim_results,
        "claim_anomaly_results": claim_anomaly_results,
        "claim_model_summary": claim_model_summary,
    }
