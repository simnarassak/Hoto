from pathlib import Path
import pandas as pd
from hoto.config import CLASSIFIER_THRESHOLD, DATA_DIR, MODELS_DIR
from hoto.infer import load_artifact_csv, load_claim_models, score_claims, top_n_queue
from hoto.risk_layers import build_provider_risk

def project_paths(root=None):
    root = Path(root) if root is not None else Path.cwd()
    return (root / 'data', root / 'models')

def ensure_app_ready(data_dir=None, models_dir=None):
    from hoto.bootstrap import ensure_artifacts
    return ensure_artifacts(data_dir=data_dir, models_dir=models_dir)

def load_live_claim_scores(data_dir=None, models_dir=None, threshold=CLASSIFIER_THRESHOLD, models=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    features = load_artifact_csv('hoto_claim_features.csv', data_dir)
    metadata = load_artifact_csv('hoto_claim_metadata.csv', data_dir)
    if features is None:
        raise FileNotFoundError('hoto_claim_features.csv not found')
    if models is None:
        models = load_claim_models(models_dir)
    scored = score_claims(features, models=models, threshold=threshold, enrich=True, claim_metadata=metadata)
    if 'provider_id' not in scored.columns and metadata is not None:
        scored = scored.merge(metadata[['claim_id', 'provider_id']], on='claim_id', how='left')
    return (scored, models)

def load_provider_risk_table(scored_claims=None, data_dir=None, models_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    cached = load_artifact_csv('provider_risk_summary.csv', data_dir)
    if cached is not None:
        return cached
    if scored_claims is None:
        scored_claims, _ = load_live_claim_scores(data_dir=data_dir, models_dir=models_dir)
    providers = load_artifact_csv('hoto_providers.csv', data_dir)
    return build_provider_risk(scored_claims, providers=providers, data_dir=data_dir)

def apply_threshold_flags(scored, threshold=CLASSIFIER_THRESHOLD):
    result = scored.copy()
    if 'logistic_probability' in result.columns:
        result['classifier_flag'] = (result['logistic_probability'] >= threshold).astype(int)
    if 'xgboost_probability' in result.columns:
        result['xgboost_flag'] = (result['xgboost_probability'] >= threshold).astype(int)
    if 'ensemble_score' in result.columns:
        result['ensemble_flag'] = (result['ensemble_score'] >= threshold).astype(int)
    return result

def build_investigation_view(scored_claims, score_column='ensemble_score', top_n=100, threshold=0.0):
    scored_claims = apply_threshold_flags(scored_claims, threshold=threshold)
    queue = top_n_queue(scored_claims, score_column=score_column, n=top_n, threshold=threshold if score_column != 'rule_hit_count' else None)
    display_cols = [c for c in ['claim_id', 'provider_id', 'claim_type', 'claimed_amount', 'ensemble_score', 'xgboost_probability', 'logistic_probability', 'anomaly_score', 'rule_hit_count', 'rule_high_velocity_30d', 'rule_duplicate_signature', 'rule_provider_amount_spike', 'ensemble_flag', 'xgboost_flag', 'classifier_flag', 'anomaly_flag'] if c in queue.columns]
    return queue[display_cols]
