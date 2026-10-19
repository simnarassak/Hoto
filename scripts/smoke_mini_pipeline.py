from pathlib import Path
import tempfile
import pandas as pd
import hoto.config as cfg
import hoto.data_gen as data_gen
import hoto.scenarios as scenarios
cfg.WORLD['population'] = 2000
cfg.N_PROVIDERS = 50
cfg.N_CLAIMS = 500
cfg.N_DUPLICATE_PAYMENTS = 10
cfg.N_DUPLICATE_CLAIMS = 10
cfg.N_FREQUENCY_CLAIMS = 3
cfg.N_PROVIDER_ANOMALIES = 3
data_gen.N_PROVIDERS = 50
data_gen.N_CLAIMS = 500
scenarios.N_DUPLICATE_PAYMENTS = 10
scenarios.N_DUPLICATE_CLAIMS = 10
scenarios.N_FREQUENCY_CLAIMS = 3
scenarios.N_PROVIDER_ANOMALIES = 3
from hoto.pipeline import configure_logging, run_pipeline
from hoto.infer import load_claim_models, models_available, score_claims

def main():
    configure_logging()
    out = Path(tempfile.mkdtemp(prefix='hoto_test_'))
    print('OUT', out)
    run_pipeline(data_dir=out / 'data', models_dir=out / 'models', seed=42)
    assert models_available(out / 'models')
    features = pd.read_csv(out / 'data' / 'hoto_claim_features.csv')
    scored = score_claims(features.head(20), models=load_claim_models(out / 'models'))
    print('scored_cols', [c for c in scored.columns if 'probability' in c or 'flag' in c])
    print('MINI_PIPELINE_OK')
if __name__ == '__main__':
    main()
