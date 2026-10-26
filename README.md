
# Hoto — Synthetic Claims Risk Platform

**Tech stack:** Python, Streamlit, XGBoost, scikit-learn, SHAP, Plotly

**Live demo:** [Hoto-Sims](https://hoto-sims.streamlit.app/)

Hoto-Sims is an interactive application for exploring synthetic claims risk scoring, model explainability, claims analytics, and risk-based triage workflows.

The platform combines machine learning, transparent rule-based indicators, provider-risk analysis, and interactive visualisations to demonstrate how analytical tools can support risk assessment and review prioritisation.

> **Important:** All claims data and labels are synthetic. Model scores are intended for demonstration and learning. They are not proof of fraud and must not be used to make real-world claims decisions.

## Screenshots

### Hoto — Main Dashboard

![Hoto Main Dashboard](images/Page1.png)

Explore synthetic population, demographic, employment, and claims data through interactive visualisations.

### Hoto — Simulation Lab

![Hoto Simulation Lab](images/Page2.png)

Explore hypothetical regional demand, capacity constraints, and cost pressures using configurable scenario parameters.

## Key Features

### 1. Ensemble Claims Risk Scoring

- Combines XGBoost, logistic regression, and anomaly detection.
- Supports risk-based prioritisation of claims for review.
- Enables exploration of configurable risk thresholds and top-N review queues.
- Provides a foundation for examining patterns in synthetic claims data.

### 2. Transparent Rule-Based Indicators

Highlights potentially unusual patterns through rules such as:

- High-frequency claims activity.
- Duplicate-like claim signatures.
- Unusual provider claim amounts.

Rule hits are indicators for further examination, not proof of fraudulent activity.

### 3. Provider-Risk Analysis

Aggregates claims-related risk indicators at provider level to support exploration of patterns across providers and claims.

### 4. Model Explainability

- Individual claim contribution visualisations.
- SHAP-based explanations where supported by the selected model.
- Feature-level investigation of model outputs.

Explanations help users understand model behaviour but do not establish causation or prove fraud.

### 5. Model Evaluation and Review Capacity

Supports exploration of model performance, risk thresholds, and the number of records selected for review.

For imbalanced classification, evaluation should consider metrics such as average precision alongside the practical capacity available to investigate flagged records.

### 6. Interactive Analytics Dashboard

Provides visual exploration of synthetic claims, population characteristics, demographic distributions, and trends.

### 7. Scenario Simulation

The Simulation Lab provides a synthetic, rules-based environment for examining hypothetical regional demand and capacity changes.

Users can explore scenario parameters and compare estimated outcomes against a baseline.

**Simulation limitation:** Scenario outputs are exploratory and rules-based. They are not validated forecasts of actual claims demand or expenditure.

## Application Structure

The application includes the following areas:

1. **Home** — Overview of the Hoto environment and its synthetic data.
2. **Risk Lab** — Ensemble risk scoring, review thresholds, and provider-risk analysis.
3. **Case Explain** — Claim timelines and model contribution visualisations.
4. **Model Lab** — Model evaluation and scenario exploration.
5. **Simulation** — Regional hazard, demand, and capacity analysis.

Available features may vary with the deployed application version.

## Project Layout

```text
Hoto/
├── hoto/
├── app/
│   └── streamlit_app.py
├── app.py
├── hoto_pipeline.py
├── requirements.txt
├── data/
├── models/
├── images/
│   ├── Page1.png
│   └── Page2.png
└── README.md
```

This is an illustrative layout based on the files and folders described in the project documentation. Keep it aligned with the actual repository structure.

## Getting Started

### Prerequisites

- Python compatible with the dependencies in `requirements.txt`.
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

Activate the environment using the command for your operating system.

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

If `app/streamlit_app.py` is the main entry point:

```bash
streamlit run app/streamlit_app.py
```

If the deployed application instead uses `app.py`, run:

```bash
streamlit run app.py
```

Use the entry point configured for the current version of Hoto-Sims.

## Model Training

If the training module and command-line interface are available in your checkout, run:

```bash
python -m hoto
```

For a smaller demonstration dataset, if supported:

```bash
python -m hoto --demo
```

Confirm that these commands and options are implemented before using them. Training requirements may differ from those needed to run the deployed application.

## Inference API

The project provides a Python inference interface for loading claim models and scoring feature data.

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

This example assumes the module, functions, and CSV files match the current repository implementation. Input columns must conform to the schema expected by the inference pipeline.

## Responsible AI and Design Considerations

Hoto-Sims is a synthetic demonstration platform. Its documented design considerations include:

- **Synthetic data:** Claims and labels are generated for demonstration and experimentation.
- **Sensitive attributes:** Sensitive demographic attributes are excluded from model features.
- **Leakage prevention:** Grouped train/test splits are used to reduce person- or claim-level data leakage.
- **Appropriate evaluation:** Average precision and review capacity are considered alongside model thresholds.
- **Human review:** Risk scores and rule hits are intended to support further examination rather than replace investigation.
- **Explainability:** Feature contributions help users investigate model behaviour but do not establish causality.
- **Scenario transparency:** Rules-based scenario simulation is distinguished from machine learning forecasting.

These design considerations do not, by themselves, establish that a model is unbiased or suitable for real-world deployment. Real claims applications would require additional validation, privacy and security assessments, governance, monitoring, and domain-expert review.

## Deployment

Hoto-Sims is available through Streamlit Community Cloud:

**Live application:** [https://hoto-sims.streamlit.app/](https://hoto-sims.streamlit.app/)

For deployment or maintenance:

1. Connect the GitHub repository to Streamlit Community Cloud.
2. Select the appropriate branch.
3. Configure the correct main application file.
4. Ensure the required dependencies are specified in `requirements.txt`.
5. Configure any required secrets.
6. Deploy and verify the application.

The deployed application may differ from the current local development version.

## Project Purpose

Hoto-Sims is a personal AI and data science project exploring the integration of machine learning, explainability, risk analytics, and interactive decision-support visualisations.

It demonstrates practical development across data processing, model inference, analytical dashboards, risk prioritisation, and scenario exploration.

The platform uses synthetic data and is intended for learning, technical demonstration, and portfolio purposes.
