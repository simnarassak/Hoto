"""Orchestrate the full Hoto training pipeline."""

import argparse
import logging
import sys

from hoto.config import DATA_DIR, MODELS_DIR, WORLD, ensure_output_dirs
from hoto.data_gen import generate_baseline_world
from hoto.features import (
    build_claim_features,
    build_payment_features,
    validate_and_save_features,
)
from hoto.model_lab import (
    build_claim_model_lab,
    build_curve_artifacts,
    build_investigation_queue,
    build_payment_model_lab,
    build_unified_risk_results,
    consolidate_model_lab,
)
from hoto.infer import score_claims
from hoto.risk_layers import build_provider_risk
from hoto.scenarios import inject_fraud_scenarios, save_scenario_datasets
from hoto.train import (
    prepare_claim_model_data,
    prepare_payment_model_data,
    train_claim_models,
    train_payment_models,
)


def configure_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(data_dir=None, models_dir=None, seed=None, skip_train=False):
    data_dir, models_dir = ensure_output_dirs(data_dir, models_dir)
    seed = WORLD["random_seed"] if seed is None else seed
    log = logging.getLogger("hoto.pipeline")

    log.info(
        "Starting Hoto pipeline (country=%s, population=%s, seed=%s)",
        WORLD["country_name"],
        f"{WORLD['population']:,}",
        seed,
    )

    log.info("Stage 1/6 - generating baseline synthetic world")
    baseline = generate_baseline_world(seed=seed, data_dir=data_dir)
    log.info(
        "Baseline ready: people=%s providers=%s claims=%s payments=%s events=%s",
        f"{len(baseline['people']):,}",
        f"{len(baseline['providers']):,}",
        f"{len(baseline['claims']):,}",
        f"{len(baseline['payments']):,}",
        f"{len(baseline['claim_events']):,}",
    )

    log.info("Stage 2/6 - injecting fraud scenarios")
    scenarios = inject_fraud_scenarios(
        baseline["claims"],
        baseline["payments"],
        baseline["people"],
        baseline["providers"],
        seed=seed,
    )
    save_scenario_datasets(scenarios, data_dir)
    log.info(
        "Scenarios ready: modelling claims=%s payments=%s labels=%s",
        f"{len(scenarios['claims_synthetic']):,}",
        f"{len(scenarios['payments_synthetic']):,}",
        f"{len(scenarios['fraud_labels']):,}",
    )

    log.info("Stage 3/6 - engineering features")
    claim_bundle = build_claim_features(
        scenarios["claims_synthetic"],
        scenarios["fraud_labels"],
    )
    payment_bundle = build_payment_features(
        scenarios["payments_synthetic"],
        scenarios["claims_synthetic"],
        scenarios["fraud_labels"],
    )
    validate_and_save_features(claim_bundle, payment_bundle, data_dir)

    if skip_train:
        log.info("skip_train=True - stopping before model training")
        return {
            "baseline": baseline,
            "scenarios": scenarios,
            "claim_bundle": claim_bundle,
            "payment_bundle": payment_bundle,
        }

    log.info("Stage 4/6 - training payment models")
    payment_split = prepare_payment_model_data(
        payment_bundle["payment_features"],
        payment_bundle["payment_targets"],
        payment_bundle["payment_metadata"],
    )
    payment_models = train_payment_models(
        payment_split, models_dir=models_dir, data_dir=data_dir
    )

    log.info("Stage 5/6 - training claim models")
    claim_split = prepare_claim_model_data(
        claim_bundle["claim_features"],
        claim_bundle["claim_targets"],
        claim_bundle["claim_metadata"],
    )
    claim_models = train_claim_models(
        claim_split, models_dir=models_dir, data_dir=data_dir
    )

    log.info("Stage 6/6 - building investigation queue and Model Lab artifacts")
    unified = build_unified_risk_results(
        payment_models["model_results"],
        claim_split["claim_model_data"],
        claim_split["claim_test_idx"],
        claim_models["claim_fraud_probability"],
        claim_models["claim_predictions"],
        claim_models["claim_anomaly_scores"],
        claim_models["claim_anomaly_flags"],
        data_dir=data_dir,
    )
    build_investigation_queue(unified, data_dir=data_dir)

    payment_lab = build_payment_model_lab(
        payment_split["payment_model_data"],
        payment_split["X_test"],
        payment_split["y_test"],
        payment_models["fraud_probabilities"],
        payment_models["xgb_probabilities"],
        data_dir=data_dir,
    )
    claim_lab = build_claim_model_lab(
        claim_split["claim_model_data"],
        claim_split["X_claim_test"],
        claim_split["y_claim_test"],
        claim_models["claim_fraud_probability"],
        claim_models["claim_xgb_probabilities"],
        data_dir=data_dir,
    )
    consolidate_model_lab(payment_lab, claim_lab, data_dir=data_dir)
    build_curve_artifacts(
        payment_split["y_test"],
        payment_lab["payment_model_predictions"],
        claim_split["y_claim_test"],
        claim_lab["claim_model_predictions"],
        data_dir=data_dir,
    )

    log.info("Building provider-risk dimension and enriched claim scores")
    claim_scored = score_claims(
        claim_bundle["claim_features"],
        models={
            "logistic": claim_models["claim_fraud_model"],
            "anomaly": claim_models["claim_anomaly_model"],
            "xgboost": claim_models["claim_xgb_pipeline"],
        },
        enrich=True,
        claim_metadata=claim_bundle["claim_metadata"],
    )
    # Ensure provider_id is present for aggregation.
    if "provider_id" not in claim_scored.columns:
        claim_scored = claim_scored.merge(
            claim_bundle["claim_metadata"][["claim_id", "provider_id"]],
            on="claim_id",
            how="left",
        )
    provider_risk = build_provider_risk(
        claim_scored,
        providers=baseline["providers"],
        data_dir=data_dir,
    )
    claim_scored.to_csv(data_dir / "claim_scores_enriched.csv", index=False)

    log.info("Hoto pipeline completed.")
    log.info("Artifacts written to %s and %s", data_dir, models_dir)
    return {
        "baseline": baseline,
        "scenarios": scenarios,
        "claim_bundle": claim_bundle,
        "payment_bundle": payment_bundle,
        "payment_split": payment_split,
        "claim_split": claim_split,
        "payment_models": payment_models,
        "claim_models": claim_models,
        "unified_risk_results": unified,
        "provider_risk": provider_risk,
        "claim_scored": claim_scored,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Hoto synthetic claims risk pipeline. "
            "Regenerates data, trains models, and writes Model Lab artifacts."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory for CSV artifacts (default: data)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Directory for fitted models (default: models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=WORLD["random_seed"],
        help="Random seed for generation and training",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Generate data and features only; do not train models",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use compact demo-scale population/claims (cloud-friendly)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    if args.demo:
        from hoto.bootstrap import apply_demo_scale

        apply_demo_scale()
    run_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        seed=args.seed,
        skip_train=args.skip_train,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
