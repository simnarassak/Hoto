
# Hoto — Synthetic Claims Risk Platform

**Tech stack:** Python, Streamlit, XGBoost, scikit-learn, SHAP, Plotly

Hoto is an interactive sandbox for exploring synthetic claims risk scoring, model explainability, claims analytics, and scenario-based decision support.

The platform brings together claims analytics, risk triage, provider-risk aggregation, model interpretation, and regional impact simulation in an interactive application.

> **Important:** All claims data and labels are synthetic. Model scores are intended for demonstration, experimentation, and learning. They are not proof of fraud and must not be used to make real-world claim decisions.

**Live demo:** [Hoto Claims Risk Platform](https://hoto-sims.streamlit.app/)

## Screenshots

### Hoto — Main Analytics Dashboard

![Hoto Main Dashboard](images/Page1.png)

The dashboard explores synthetic population, demographic, employment, and claims data through interactive visualisations.

### Hoto — Regional Scenario Simulation

![Hoto Scenario Simulation](images/Page2.png)

The simulation lab allows users to explore hypothetical regional demand changes, capacity constraints, and associated cost pressures under configurable scenarios.

## Key Features

### 1. Claims Risk Triage

- Ensemble risk scoring using XGBoost, logistic regression, and anomaly detection.
- Configurable risk thresholds and top-N review selection.
- Risk prioritisation to support investigation workflows.
- Provider-risk aggregation and visualisation.

### 2. Transparent Rule-Based Indicators

Rule-based indicators highlight potentially unusual patterns, including:

- High-frequency claims activity.
- Duplicate-like claim signatures.
- Unusual provider claim amounts.

These indicators support exploration and review. They do not establish that a claim is fraudulent.

### 3. Model Explainability

- Individual claim contribution visualisations.
- SHAP-based explanations where implemented.
- Feature-level investigation of model outputs.
- Interactive exploration of selected claims and risk scores.

Explanations describe how model features contribute to a prediction; they do not establish causation or prove fraudulent behaviour.

### 4. Model Evaluation

The model lab supports exploration of model performance and risk-scoring scenarios.

Evaluation should consider metrics appropriate to imbalanced classification, including average precision, alongside the operational capacity available for reviewing flagged cases.

### 5. Claims and Population Analytics

Interactive dashboards explore synthetic population characteristics, claims activity over time, claim types, and demographic distributions.

### 6. Regional Scenario Simulation

The simulation lab provides a rules-based sandbox for exploring hypothetical changes in regional claims demand and operational capacity.

Users can configure scenario parameters such as:

- Hazard or stress scenario.
- Stress shock.
- Capacity mode.
- Simulation horizon.
- Stress-to-demand relationship.
- Prevention budget adjustment.
- Time-series metric and region of interest.

Outputs include regional demand changes and estimated cost pressures relative to a baseline.

**Simulation limitation:** This is a synthetic, rules-based scenario model, not a validated machine learning forecast or a prediction of actual claims expenditure.

## Application Structure

The application is organised into the following areas:

1. **Home** — Hoto overview and synthetic claims environment.
2. **Risk Lab** — Ensemble risk scoring, review thresholds, and provider-risk analysis.
3. **Case Explain** — Claim-level investigation, timelines, and contribution visualisations.
4. **Model Lab** — Model evaluation and risk-scoring experiments.
5. **Simulation** — Regional hazard, demand, and capacity scenarios.

The exact navigation and available features depend on the currently deployed application version.

## Hoto Rural — Independent MVP

Hoto Rural is a separate prototype exploring regional microeconomic and freshwater scenarios.

Its modules include:

- Economics.
- Freshwater.
- Scenario Lab with Monte Carlo simulation.

The conceptual feedback loop connects:

**Water → Crops and livestock → Manure and nutrients → Profit and labour**

This module explores relationships between environmental resources and agricultural activity through scenario-based modelling. Its outputs should be interpreted as exploratory simulations rather than validated forecasts.

Run the rural module using the commands below, provided the corresponding module and file paths exist in your checkout:

```bash
python -m hoto_rural
streamlit run app/rural_streamlit_app.py
```

## Project Layout

```text
Hoto/
├── hoto/
├── hoto_rural/
├── app/
│   ├── streamlit_app.py
│   └── rural_streamlit_app.py
├── app.py
├── hoto_pipeline.py
├── requirements.txt
├── data/
│   └── rural/
└── models/
```

This layout describes the main project components. Additional files, datasets, and model artefacts may exist in the repository.

## Getting Started

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`.
- Git.
- A local copy of the repository.

### 1. Clone the repository

```bash
git clone https://github.com/simnarassak/Hoto.git
cd Hoto
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it using the command appropriate for your operating system.

**Windows — Command Prompt**

```bat
.venv\Scripts\activate.bat
```

**Windows — PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

If `app/streamlit_app.py` is the main application entry point:

```bash
streamlit run app/streamlit_app.py
```

If your project uses `app.py` as its entry point instead, run:

```bash
streamlit run app.py
```

Use the entry point that matches the current deployed version.

## Model Training

If the repository's training module is available, train the models using:

```bash
python -m hoto
```

For a smaller demonstration dataset, if supported by the current implementation:

```bash
python -m hoto --demo
```

Verify the expected data and model paths before running training. The commands above assume the relevant command-line options are implemented in the project.

## Inference API

The following example illustrates the project's Python inference interface:

```python
from hoto.infer import load_claim_models, score_claims
import pandas as pd

models = load_claim_models()

features = pd.read_csv("data/hoto_claim_features.csv")
metadata = pd.read_csv("data/hoto_claim_metadata.csv")

scored = score_claims(
    features,
    models=models,
    claim_metadata=metadata,
)

print(scored.head())
```

The example assumes the referenced module, files, and function signatures are available in the repository. Input data must follow the feature schema expected by the inference pipeline.

## Responsible AI and Design Considerations

Hoto is a demonstration platform. Its design notes include the following principles:

- **Synthetic data:** Claims and labels are generated for experimentation, not obtained from actual fraud investigations.
- **Sensitive attributes:** Sensitive demographic attributes are kept out of model features.
- **Leakage prevention:** Grouped train/test splits are intended to reduce person- or claim-level data leakage.
- **Appropriate evaluation:** Average precision and review capacity are considered alongside classification metrics and thresholds.
- **Human review:** Risk scores and rule hits should inform further examination, not replace investigation or establish guilt.
- **Explainability:** Feature contributions help users examine model behaviour but do not establish causality.
- **Scenario transparency:** Rules-based simulations are distinguished from machine learning forecasts.

These design choices do not, by themselves, establish that the models are unbiased, production-ready, or suitable for real claims decisions. Any real-world deployment would require further validation, privacy and security assessment, governance, monitoring, and domain-expert review.

## Deployment

The public demonstration is hosted on Streamlit Community Cloud:

**Application:** [https://hoto-sims.streamlit.app/](https://hoto-sims.streamlit.app/)

For a Streamlit Community Cloud deployment:

1. Connect the GitHub repository.
2. Select the branch to deploy.
3. Set the main application file to the actual Streamlit entry point.
4. Configure any required secrets and dependencies.
5. Deploy and verify the application.

The deployed version may differ from the current local development version.

## Purpose

Hoto is a personal project exploring how machine learning, explainability, interactive analytics, and scenario simulation can be combined in a single application.

It demonstrates practical development across data processing, model inference, visualisation, risk analysis, and interactive decision-support interfaces.

The platform uses synthetic data and is intended for learning, technical demonstration, and portfolio purposes.
