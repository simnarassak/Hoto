import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoto.app_data import build_investigation_view, load_live_claim_scores, load_provider_risk_table
from hoto.bootstrap import artifacts_ready

def main():
    data_dir = ROOT / 'data'
    models_dir = ROOT / 'models'
    print('artifacts_ready', artifacts_ready(data_dir, models_dir))
    scored, _ = load_live_claim_scores(data_dir=data_dir, models_dir=models_dir)
    print('scored_rows', len(scored))
    print('has_ensemble', 'ensemble_score' in scored.columns)
    print('has_rules', 'rule_hit_count' in scored.columns)
    view = build_investigation_view(scored, top_n=20, threshold=0.35)
    print('queue_rows', len(view))
    providers = load_provider_risk_table(scored_claims=scored, data_dir=data_dir, models_dir=models_dir)
    print('provider_rows', len(providers))
    print('top_provider', providers.iloc[0]['provider_id'])
    print('APP_LAYERS_OK')
if __name__ == '__main__':
    main()
