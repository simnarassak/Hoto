# Hoto pipeline — testing guide

## What this project does
The `hoto` package rebuilds the synthetic Hoto datasets, engineers claim/payment
features, trains classifiers and anomaly detectors (including XGBoost), and
writes Model Lab CSV artifacts plus fitted models.

Outputs (relative to the directory from which you run the pipeline):
- `data/`
- `models/`

**Important:** training and serving are separated.
- Train with `python -m hoto` (or `python hoto_pipeline.py`)
- Serve the public demo with `streamlit run app/streamlit_app.py`
- App code should import `hoto.infer`, not re-run the training pipeline

## Test in a clean folder

1. Put this project folder on disk (or clone it).
2. Create and activate a virtual environment.
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Run training:
   `python -m hoto`
5. Confirm the run ends with `Hoto pipeline completed.` and that `data/` and
   `models/` have been created.
6. Launch the public app:
   `streamlit run app/streamlit_app.py`

XGBoost metrics can vary slightly across library versions.

## Package map

- `hoto/config.py` — seeds, sizes, paths
- `hoto/data_gen.py` — synthetic world
- `hoto/scenarios.py` — fraud injection
- `hoto/features.py` — feature tables
- `hoto/train.py` — model fitting
- `hoto/model_lab.py` — queue + evaluation artifacts
- `hoto/infer.py` — score saved models (app-safe)
- `app/streamlit_app.py` — public UI (no auth)

## Static security note

The original notebook conversion was scanned for network requests, shell
execution, dynamic `eval`/`exec`, and encoded payload decoding. No matches were
found in executable source. This is a static review, not a commercial antivirus
guarantee. Run your organisation's endpoint scanner before sharing or executing
files in a work environment.
