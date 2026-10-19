---
title: Hoto Claims Risk Demo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Synthetic claims risk triage demo (ensemble, rules, provider risk)
---

# Hoto — Synthetic Claims Risk Platform

Public sandbox for exploring synthetic claims risk scoring. Data and labels are
fully synthetic. Model scores are for triage demos and learning — not proof of
fraud or real claim decisions.

## App structure

1. **Home** — what this is + Hoto world narrative (people → claims → providers → YoY approvals)  
2. **Risk lab** — ensemble queue, threshold, provider risk charts  
3. **Case explain** — timeline and what influence the model to predict a specific outcome
4. **Model lab** — metrics and scenarios  
5. **Simulation** — 2D regional sandbox: economic stress, flood, earthquake, pandemic, storm, tsunami, all-region stress, plus Normal/Tight/Very tight capacity modes ( this a rule based simulation, not ML)  

## What is the product / modeling dimensions

1. **Ensemble triage score** — blends XGBoost + logistic + normalized anomaly  
2. **Transparent rule hits** — velocity, duplicate signature, provider amount spike  
3. **Provider-risk dimension** — aggregates claim risk to provider networks  
4. **Live threshold / top-N** — investigate by capacity, not a fixed 0.5 cut  
5. **Custom force-style explanations** — Plotly waterfall; toggle SHAP vs simple linear contributions

## Project layout

```
Hoto/
  hoto/                 # core package
  app/
    streamlit_app.py    # Streamlit Community Cloud entry
  app.py                # Hugging Face Spaces (Gradio) entry (Not published yet)
  hoto_pipeline.py      # thin training CLI wrapper
  requirements.txt
  data/                 # generated artifacts
  models/               # fitted models
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Train

Full world (heavier):

```bash
python -m hoto
```

Cloud-friendly compact world:

```bash
python -m hoto --demo
```

### Run apps locally

Streamlit:

```bash
streamlit run app/streamlit_app.py
```

Gradio (optional / deferred — not required for Streamlit Cloud):

```bash
python -m pip install gradio==4.44.1
python app.py
```

If `data/` / `models/` are missing, apps auto-build a **demo-scale** world on first boot.

## Deploy (Streamlit Community Cloud)

1. Main file path: `app/streamlit_app.py`
2. In **Advanced settings**, set **Python version to 3.12** (not 3.13/3.14)
3. App URL example: `hoto-sims.streamlit.app`

If the app was already deployed on the wrong Python version: delete it, then
redeploy with the same subdomain and choose **3.12** again.

## Inference API

```python
from hoto.infer import load_claim_models, score_claims
import pandas as pd

models = load_claim_models()
features = pd.read_csv("data/hoto_claim_features.csv")
metadata = pd.read_csv("data/hoto_claim_metadata.csv")
scored = score_claims(features, models=models, claim_metadata=metadata)
```

## Design notes

- Sensitive demographics stay out of model features.
- Grouped train/test splits reduce person/claim leakage.
- Prefer review-capacity and average precision over a hard 0.5 threshold.
- Labels come from injected scenarios (S001–S004), not real fraud cases.
