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
fraud or real claim decisions. **No authentication.**

## App structure

1. **Home** — what this is + Hoto world narrative (people → claims → providers → YoY approvals)  
2. **Risk lab** — ensemble queue, threshold, provider risk charts  
3. **Case explain** — timeline + Option C contribution toggle (SHAP vs linear) with custom Plotly force/waterfall  
4. **Model lab** — metrics and scenarios  
5. **Simulation** — 2D regional sandbox: economic stress, flood, earthquake, pandemic, storm, tsunami, all-region stress, plus Normal/Tight/Very tight capacity modes (rules-based, not ML)  

## What is new (product / modeling dimensions)

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
  app.py                # Hugging Face Spaces (Gradio) entry
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

Gradio (same as HF Spaces):

```bash
python app.py
```

If `data/` / `models/` are missing, apps auto-build a **demo-scale** world on first boot.

## Deploy from your personal GitHub

### 1) Hugging Face Spaces (Gradio)

1. Push this repo to GitHub.
2. Create a Space → SDK **Gradio** → link the GitHub repo.
3. HF reads `README.md` frontmatter (`sdk: gradio`, `app_file: app.py`).
4. Free-tier tip: either commit demo artifacts from `python -m hoto --demo`,
   or let Spaces cold-start call `ensure_artifacts()` (demo scale).

### 2) Streamlit Community Cloud

1. Push the same GitHub repo.
2. New app → select repo → Main file path: `app/streamlit_app.py`.
3. Python version 3.10+ recommended.
4. Same artifact guidance as above.

### Artifact size tip

Full 100k-population outputs can be large for git. Prefer:

- commit **demo** artifacts (`python -m hoto --demo`), or
- omit `data/` + `models/` from git and let cloud boot build demo scale.

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
