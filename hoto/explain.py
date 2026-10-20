import numpy as np
import pandas as pd
import xgboost as xgb


def explain_review(row):
    explanations = []
    if row["classifier_flag"] == 1:
        explanations.append("The classification model assigned a high fraud-risk score.")
    if row["anomaly_flag"] == 1:
        explanations.append("The anomaly detector identified an unusual pattern.")
    if not explanations:
        return "No detection signal recorded."
    return " ".join(explanations)


def explain_from_features(row, top_features=None):
    base = explain_review(row)
    if not top_features:
        return base
    highlights = ", ".join((str(item) for item in top_features[:3]))
    return f"{base} Key drivers: {highlights}."


def shap_feature_contributions(pipeline, feature_row, top_k=10):
    if hasattr(feature_row, "to_frame"):
        X = feature_row.to_frame().T if feature_row.ndim == 1 else feature_row
    else:
        X = pd.DataFrame(feature_row)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    dmat = xgb.DMatrix(np.asarray(X_transformed), feature_names=feature_names)
    contribs = np.asarray(classifier.get_booster().predict(dmat, pred_contribs=True))
    if contribs.ndim == 1:
        contribs = contribs.reshape(1, -1)
    values = contribs[0, :-1]
    contributions = pd.DataFrame({"feature": feature_names, "shap_value": values})
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = (
        contributions.sort_values("abs_shap", ascending=False)
        .head(top_k)
        .drop(columns="abs_shap")
    )
    return contributions.reset_index(drop=True)


def create_shap_force_plot(
    pipeline,
    predictions,
    X_test_data,
    selection="Highest risk",
    max_records=1000,
    random_state=42,
):
    if not isinstance(max_records, int) or max_records < 1:
        raise ValueError("max_records must be a positive integer.")
    if selection not in {"Highest risk", "Random sample"}:
        raise ValueError("selection must be 'Highest risk' or 'Random sample'.")
    n_records = min(max_records, len(predictions))
    if selection == "Highest risk":
        selected = (
            predictions.sort_values("xgboost_probability", ascending=False)
            .head(n_records)
            .copy()
        )
    else:
        selected = predictions.sample(n=n_records, random_state=random_state).copy()
    X_selected = X_test_data.loc[selected.index].copy()
    if not X_selected.index.equals(selected.index):
        raise AssertionError("Selected features and prediction rows are misaligned.")
    preprocessor_step = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor_step.transform(X_selected)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = [str(name) for name in preprocessor_step.get_feature_names_out()]
    dmat = xgb.DMatrix(np.asarray(X_transformed), feature_names=feature_names)
    contribs = np.asarray(classifier.get_booster().predict(dmat, pred_contribs=True))
    if contribs.ndim == 1:
        contribs = contribs.reshape(1, -1)
    shap_values = contribs[:, :-1]
    if shap_values.shape != (len(X_selected), len(feature_names)):
        raise AssertionError("Contribution values do not align with selected rows/features.")
    return (selected, shap_values)
