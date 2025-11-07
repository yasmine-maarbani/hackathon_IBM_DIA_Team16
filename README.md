# 🌱 LLM Energy & CO₂e Estimator — Team 16

Estimate and simulate the environmental impact of Large Language Model (LLM) inference.  
This project provides:
- Two predictive models:
  - Per-request (kWh/request)
  - Per-token (kWh/token), scaled by total tokens
- CO₂e conversion with selectable grid intensities
- An interactive Streamlit app for single requests, conversations, batch analysis, and pipeline comparison
- Optional uncertainty bands and feature importance

Demo and materials:
- [Demo video](https://github.com/yasmine-maarbani/hackathon_IBM_DIA_Team16/blob/main/demo_video.mp4)
- [Pitch deck](https://github.com/yasmine-maarbani/hackathon_IBM_DIA_Team16/blob/main/pitchdeck.pdf)

---

## Table of contents
- [Project structure](#project-structure)
- [Formulas](#formulas)
- [Data, features, and leakage control](#data-features-and-leakage-control)
- [Models and preprocessing](#models-and-preprocessing)
- [Results and evaluation](#results-and-evaluation)
- [Web interface](#web-interface)
- [Quickstart](#quickstart)
- [Notes and limitations](#notes-and-limitations)
- [References](#references)

---

## Project structure
```
hackathon_IBM_DIA_Team16/
├─ artifacts/                  # datasets, trained models
├─ certification/              # competition docs or certificates
├─ source/                     # Streamlit app and utilities
├─ requirements.txt
├─ demo_video.mp4
└─ pitchdeck.pdf
```

- Requirements: [requirements.txt](https://github.com/yasmine-maarbani/hackathon_IBM_DIA_Team16/blob/main/requirements.txt)
- App code: browse the [source/](https://github.com/yasmine-maarbani/hackathon_IBM_DIA_Team16/tree/main/source) folder.

---

## Formulas

Let:
- $X$: feature vector (tokens, durations, model\_name, type, …)
- $T$: total tokens
- $I_{\text{grid}}$: grid carbon intensity (kg CO$_2$e per kWh)

<img width="992" height="561" alt="image" src="https://github.com/user-attachments/assets/fcfb1b83-e230-4c8b-8357-3fbfa1670d74" />


Units:
- $\hat{E}_{\text{request}}$: kWh/request
- $\hat{E}_{\text{token}}$: kWh/token
- $I_{\text{grid}}$: kg CO$_2$e/kWh
- $\widehat{\mathrm{CO_{2}e}}$: kg

---

## Data, features, and leakage control

- Core columns (observed or engineered):
  - Tokens: `prompt_token_length`, `response_token_length`, `total_tokens`
  - Timings: `total_duration_s`, `prompt_duration_s`, `response_duration_s`, `load_duration_s` (normalized to seconds)
  - Context: `model_name`, `type`
- Target(s):
  - Per-request: `energy_consumption_llm_total` (kWh/request)
  - Per-token: `energy_per_token_kWh = energy_consumption_llm_total / total_tokens` (when `total_tokens > 0`)
- Leakage guard (excluded from features): `energy_*` components and `power_draw_*`.

---

## Models and preprocessing

- Preprocessor (ColumnTransformer):
  - Numeric: `StandardScaler(with_mean=False)`
  - Categorical: `OneHotEncoder(handle_unknown="ignore")`
- Candidate regressors:
  - `LinearRegression` (compact baseline)
  - `RandomForestRegressor` (nonlinear baseline)
- Pipelines: `Pipeline([("pre", pre), ("model", regressor)])`

Rationale:
- LinearRegression provides tiny artifacts and fast inference with interpretable behavior.
- RandomForest captures nonlinearity and interactions with minimal tuning.

---

## Results and evaluation

Split: 80/20 train/validation (random_state=42).  
Metrics reported: MAE, RMSE, MAPE, R², and a tercile (3-bin) ranking sanity check.

Per-request (kWh/request)
- LinearRegression
  - MAE = 6.012e−05 kWh (0.06012 Wh)
  - RMSE = 2.5289e−04 kWh
  - MAPE = 109.53% (inflated by near-zero true values)
  - R² = 0.643
  - Acc/Recall (terciles) ≈ 0.642
- RandomForest
  - MAE = 2.92e−06 kWh (0.00292 Wh ≈ 10.5 J)
  - RMSE = 4.863e−05 kWh
  - MAPE = 1.90%
  - R² = 0.987
  - Acc/Recall (terciles) ≈ 0.993

Per-token (kWh/token)
- LinearRegression
  - MAE = 5.13e−08 kWh/token
  - RMSE = 2.28e−07 kWh/token
  - MAPE = 12.45%
  - R² = 0.944
  - Acc/Recall (terciles) ≈ 0.879
- RandomForest
  - MAE = 1.59e−08 kWh/token (≈ 0.057 J/token)
  - RMSE = 1.048e−07 kWh/token
  - MAPE = 3.54%
  - R² = 0.988
  - Acc/Recall (terciles) ≈ 0.962

Interpretation:
- RandomForest is the preferred model for both targets (best MAE/RMSE/MAPE and high R²).
- Per-request LinReg’s high MAPE stems from very small denominators; MAE in kWh is more informative.
- The per-token RF is ideal for “what‑if” control over response length; the per-request RF gives a robust end‑to‑end estimate.

---

## Web interface

Built with Streamlit + Plotly (see `source/`).

Tabs and features:
- Single Request
  - Prompt token estimation (tiktoken or heuristic)
  - Response token control (auto median or manual slider)
  - Two estimates: per-request (kWh) and per-token×tokens (kWh)
  - CO₂e via grid presets (Global, EU, US, India, Renewable, Custom)
  - Optional conformal intervals (residual bands)
  - CO₂ budget optimizer (suggested token cap)
  - Heatmap: CO₂e vs (response tokens × grid intensity)
  - Feature-row preview (exact model inputs with imputed values)
- Conversation
  - Multi-turn simulation with optional context carry
  - Per-turn and cumulative energy/CO₂ curves
- Batch Simulate
  - CSV upload: `prompt,response_tokens,[model_name,type]`
  - Per-row energy/CO₂ + download
- Compare Pipelines
  - Side-by-side: per-request vs per-token×tokens
- Explain & Validate
  - MAE/R² on a dataset sample
  - Permutation feature importance

---

## Quickstart

Create and activate a virtual environment, install dependencies, and launch the app.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run source\web-interface.py
```

macOS/Linux (bash/zsh):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run source/web-interface.py
```

Tips:
- Keep large `.joblib` artifacts on local disk (not cloud-only) to avoid lazy-file issues.
- If you see binary wheels errors (NumPy/SciPy), ensure you run Streamlit with the same Python that installed the packages: `python -m streamlit ...`.

---

## References
- Strubell, E., Ganesh, A., McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP.  
- Patterson, D. et al. (2021). Carbon Emissions and Large Neural Network Training.  
- Lacoste, A. et al. (2019). Quantifying the Carbon Emissions of Machine Learning.  
- Lannelongue, L., Grealey, J., Inouye, M. (2021). Green Algorithms: Quantifying the carbon footprint of computational biology.
