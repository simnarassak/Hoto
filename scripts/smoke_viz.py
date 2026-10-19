from hoto import charts
from hoto.app_data import load_live_claim_scores
from hoto.contributions import explain_claim_contributions, force_waterfall_figure
from hoto.features import CLAIM_MODEL_CATEGORICAL, CLAIM_MODEL_NUMERIC
from hoto.infer import load_artifact_csv, load_claim_models


def main():
    pop = load_artifact_csv("hoto_population.csv", "data")
    claims = load_artifact_csv("hoto_claims_synthetic.csv", "data")
    prov = load_artifact_csv("hoto_providers.csv", "data")

    assert charts.population_by_region(pop) is not None
    assert charts.age_by_group_heatmap(pop) is not None
    assert charts.employment_mix(pop) is not None
    assert charts.claims_over_time(claims) is not None
    assert charts.claim_type_mix(claims) is not None
    assert charts.claims_by_region(claims) is not None
    assert charts.providers_by_type_region(prov) is not None
    assert charts.approved_amount_year_on_year(claims) is not None

    models = load_claim_models("models")
    scored, _ = load_live_claim_scores(
        data_dir="data", models_dir="models", models=models
    )
    cols = CLAIM_MODEL_CATEGORICAL + CLAIM_MODEL_NUMERIC
    row = scored.sort_values("ensemble_score", ascending=False).iloc[0][cols]

    for method in [
        "Model contributions (SHAP)",
        "Simple linear contributions",
    ]:
        explanation = explain_claim_contributions(
            row,
            method=method,
            xgb_pipeline=models["xgboost"],
            logistic_pipeline=models["logistic"],
            top_k=8,
        )
        fig = force_waterfall_figure(explanation)
        assert fig is not None
        print(
            method,
            "base",
            round(explanation["base_value"], 3),
            "final",
            round(explanation["final_value"], 3),
        )

    print("BUILD_OK")


if __name__ == "__main__":
    main()
