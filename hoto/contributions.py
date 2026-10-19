"""Option C: contribution numbers + custom Plotly force/waterfall chart."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _as_feature_frame(feature_row):
    if hasattr(feature_row, "to_frame"):
        return feature_row.to_frame().T if getattr(feature_row, "ndim", 1) == 1 else feature_row.copy()
    return pd.DataFrame(feature_row)


def shap_contributions(pipeline, feature_row, top_k=12):
    """XGBoost TreeExplainer contributions (often log-odds)."""
    import shap

    X = _as_feature_frame(feature_row)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    values = shap_values[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[-1])
    else:
        base_value = float(base_value)

    table = pd.DataFrame({
        "feature": feature_names,
        "contribution": values,
    })
    table["abs_contribution"] = table["contribution"].abs()
    table = table.sort_values("abs_contribution", ascending=False).head(top_k)
    final_value = base_value + float(values.sum())
    return {
        "method": "Model contributions (SHAP)",
        "base_value": base_value,
        "final_value": final_value,
        "unit_note": "Values are typically model log-odds contributions, not probabilities.",
        "contributions": table.drop(columns="abs_contribution").reset_index(drop=True),
    }


def linear_contributions(pipeline, feature_row, top_k=12):
    """Simple linear contributions from a logistic regression pipeline."""
    X = _as_feature_frame(feature_row)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    x = np.asarray(X_transformed)[0]
    feature_names = list(preprocessor.get_feature_names_out())
    coef = np.asarray(classifier.coef_[0], dtype=float)
    intercept = float(classifier.intercept_[0])
    values = coef * x

    table = pd.DataFrame({
        "feature": feature_names,
        "contribution": values,
    })
    table["abs_contribution"] = table["contribution"].abs()
    table = table.sort_values("abs_contribution", ascending=False).head(top_k)
    final_value = intercept + float(values.sum())
    return {
        "method": "Simple linear contributions",
        "base_value": intercept,
        "final_value": final_value,
        "unit_note": (
            "Approximate log-odds style contributions from logistic "
            "coefficients x transformed features."
        ),
        "contributions": table.drop(columns="abs_contribution").reset_index(drop=True),
    }


def explain_claim_contributions(
    feature_row,
    method="Model contributions (SHAP)",
    xgb_pipeline=None,
    logistic_pipeline=None,
    top_k=12,
):
    if method == "Model contributions (SHAP)":
        if xgb_pipeline is None:
            raise ValueError("xgb_pipeline is required for SHAP contributions")
        return shap_contributions(xgb_pipeline, feature_row, top_k=top_k)
    if method == "Simple linear contributions":
        if logistic_pipeline is None:
            raise ValueError("logistic_pipeline is required for linear contributions")
        return linear_contributions(logistic_pipeline, feature_row, top_k=top_k)
    raise ValueError(f"Unknown method: {method}")


def force_waterfall_figure(explanation, title=None):
    """
    Custom force-like Plotly waterfall.

    Shows base value, top feature pushes, and final score.
    """
    contrib = explanation["contributions"].copy()
    if contrib.empty:
        fig = go.Figure()
        fig.add_annotation(text="No contributions available", showarrow=False)
        return fig

    # Keep signed order as returned (already top by abs), but waterfall
    # readability is better sorting by contribution descending.
    contrib = contrib.sort_values("contribution", ascending=False)

    labels = ["Base"] + contrib["feature"].tolist() + ["Final"]
    measures = ["absolute"] + ["relative"] * len(contrib) + ["total"]
    values = (
        [explanation["base_value"]]
        + contrib["contribution"].tolist()
        + [0.0]
    )

    fig = go.Figure(
        go.Waterfall(
            name="contribution",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"{v:+.3f}" for v in values[:-1]] + [f"{explanation['final_value']:.3f}"],
            textposition="outside",
            connector={"line": {"color": "#8A8A8A"}},
            increasing={"marker": {"color": "#C45C26"}},
            decreasing={"marker": {"color": "#2F6F8F"}},
            totals={"marker": {"color": "#3D3D3D"}},
        )
    )
    fig.update_layout(
        title=title or explanation["method"],
        showlegend=False,
        waterfallgap=0.25,
        margin=dict(l=20, r=20, t=50, b=120),
        height=460,
        xaxis=dict(tickangle=-35),
        yaxis_title="Contribution",
        annotations=[
            dict(
                text=explanation["unit_note"],
                xref="paper",
                yref="paper",
                x=0,
                y=-0.28,
                showarrow=False,
                align="left",
                font=dict(size=11, color="#555555"),
            )
        ],
    )
    return fig
