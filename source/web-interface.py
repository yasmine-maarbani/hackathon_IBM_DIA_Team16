import io
import os
import pickle
import textwrap
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------- Config -------------
st.set_page_config(page_title="LLM Query CO₂e Estimator", page_icon="🌱", layout="wide")
DEFAULT_DATA_PATH = "artifacts/processed.csv"  # You can change this
CARBON_PRESETS = {
    "Global avg (~0.475)": 0.475,
    "EU avg (~0.25)": 0.25,
    "US avg (~0.40)": 0.40,
    "India (~0.70)": 0.70,
    "Renewable-heavy (~0.05)": 0.05,
    "Custom": None,
}

# ------------- Utilities -------------
@st.cache_data(show_spinner=False)
def load_data(uploaded: Optional[io.BytesIO], fallback_path: str) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if not os.path.exists(fallback_path):
            return pd.DataFrame()
        df = pd.read_csv(fallback_path)
    # standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def to_wh(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return s
    # if non-negative and < 1.0, likely kWh -> convert to Wh
    if float(finite.min()) >= 0 and float(finite.max()) < 1.0:
        return s * 1000.0
    return s

def ensure_energy_per_token(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "total_tokens" not in out.columns:
        # compute total tokens if present
        if {"prompt_token_length", "response_token_length"}.issubset(out.columns):
            out["total_tokens"] = (
                pd.to_numeric(out["prompt_token_length"], errors="coerce").fillna(0).astype(float)
                + pd.to_numeric(out["response_token_length"], errors="coerce").fillna(0).astype(float)
            )
        else:
            out["total_tokens"] = np.nan
    if "energy_per_token_wh" not in out.columns:
        if "energy_consumption_llm_total" in out.columns:
            out["energy_wh_total"] = to_wh(out["energy_consumption_llm_total"])
        elif "energy_wh_total" in out.columns:
            out["energy_wh_total"] = pd.to_numeric(out["energy_wh_total"], errors="coerce")
        else:
            out["energy_wh_total"] = np.nan
        out["energy_per_token_wh"] = np.where(
            (out["total_tokens"] > 0) & out["total_tokens"].notna(),
            out["energy_wh_total"] / out["total_tokens"],
            np.nan
        )
    return out

@st.cache_data(show_spinner=False)
def build_per_model_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    # Derive per-model median energy/token and IQR as uncertainty
    work = ensure_energy_per_token(df)
    if "model_name" not in work.columns:
        work["model_name"] = "unknown"
    work = work.replace([np.inf, -np.inf], np.nan)
    # robust filtering
    work = work[(work["energy_per_token_wh"].notna()) & (work["energy_per_token_wh"] >= 0)]
    if work.empty:
        return work, {}
    stats = work.groupby("model_name")["energy_per_token_wh"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
    ).reset_index()
    baseline_map = {
        row["model_name"]: {
            "median": float(row["median"]),
            "q25": float(row["q25"]),
            "q75": float(row["q75"]),
        }
        for _, row in stats.iterrows()
    }
    return stats, baseline_map

def estimate_tokens(text: str) -> int:
    text = text or ""
    # Try tiktoken if available
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass
    # Fallback heuristic (~4 chars/token)
    return max(1, int(len(text) / 4))

def predict_energy_wh_baseline(total_tokens: int, model_name: str, baseline_map: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
    if not baseline_map:
        return np.nan, np.nan, np.nan
    if model_name in baseline_map:
        b = baseline_map[model_name]
    else:
        # fallback to global median of medians
        medians = [v["median"] for v in baseline_map.values() if np.isfinite(v["median"])]
        if not medians:
            return np.nan, np.nan, np.nan
        global_med = float(np.median(medians))
        # construct pseudo band
        b = {"median": global_med, "q25": global_med*0.8, "q75": global_med*1.2}
    e_median = b["median"] * total_tokens
    e_low    = b["q25"]   * total_tokens
    e_high   = b["q75"]   * total_tokens
    return e_median, e_low, e_high

def to_co2e_kg(energy_wh: float, kg_per_kwh: float) -> float:
    return (energy_wh / 1000.0) * kg_per_kwh

# Optional: load user-provided model
def load_pickle(file) -> Optional[object]:
    try:
        return pickle.load(file)
    except Exception:
        return None

def ml_predict_energy_wh(
    prompt_tokens: int,
    response_tokens: int,
    df_example: pd.DataFrame,
    model: object,
    preprocessor: object
) -> Optional[float]:
    # Minimal feature parity with earlier prep: adjust as per your training
    total_tokens = prompt_tokens + response_tokens
    row = {
        "prompt_token_length": float(prompt_tokens),
        "response_token_length": float(response_tokens),
        "total_tokens": float(total_tokens),
    }
    X = pd.DataFrame([row])
    try:
        Xp = preprocessor.transform(X)
        y = model.predict(Xp)
        return float(y[0])
    except Exception:
        return None

# ------------- Sidebar -------------
with st.sidebar:
    st.header("Data & Model")
    uploaded_csv = st.file_uploader("Upload processed dataset CSV", type=["csv"], help="If not provided, app looks for artifacts/processed.csv")
    df = load_data(uploaded_csv, DEFAULT_DATA_PATH)

    if df.empty:
        st.warning("No dataset loaded. Upload a processed CSV or place one at artifacts/processed.csv")
    else:
        st.success(f"Loaded dataset with {len(df):,} rows.")

    # Optional ML model upload
    use_ml = st.checkbox("Use uploaded ML model (optional)", value=False,
                         help="If checked, provide both preprocessor.pkl and model.pkl trained to predict energy (Wh).")
    model_obj = None
    preproc_obj = None
    if use_ml:
        preproc_file = st.file_uploader("Upload preprocessor.pkl", type=["pkl"])
        model_file   = st.file_uploader("Upload model.pkl", type=["pkl"])
        if preproc_file and model_file:
            preproc_obj = load_pickle(preproc_file)
            model_obj   = load_pickle(model_file)
            if preproc_obj and model_obj:
                st.success("Model and preprocessor loaded.")
            else:
                st.error("Failed to load one or both pickle files.")

    st.header("Carbon intensity")
    grid_choice = st.selectbox("Select grid CO₂ intensity", list(CARBON_PRESETS.keys()), index=0)
    if grid_choice == "Custom":
        kg_per_kwh = st.number_input("Custom kgCO₂e per kWh", min_value=0.0, value=0.475, step=0.01)
    else:
        kg_per_kwh = CARBON_PRESETS[grid_choice]

    st.caption(f"Using {kg_per_kwh:.3f} kgCO₂e/kWh")

# ------------- Top matters -------------
st.title("🌱 LLM Query CO₂e Estimator")
st.write("Estimate the energy (Wh) and CO₂e (kg) impact of LLM inference for single turns or conversations. Choose models, enter your prompt, and simulate.")

# Prepare per-model baseline
stats_df, baseline_map = build_per_model_baseline(df) if not df.empty else (pd.DataFrame(), {})

if not df.empty and stats_df.empty:
    st.warning("Could not compute per-model baselines (energy_per_token_wh missing and not derivable).")

# Available models
model_list = sorted(df["model_name"].dropna().unique().tolist()) if ("model_name" in df.columns and not df.empty) else ["unknown"]

# ------------- Tabs -------------
tab1, tab2, tab3 = st.tabs(["Single Request", "Conversation", "Compare Models"])

with tab1:
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Single request")
        prompt = st.text_area("Prompt", placeholder="Paste your prompt here...")
        prompt_tokens = estimate_tokens(prompt) if prompt else 0
        st.caption(f"Estimated prompt tokens: {prompt_tokens}")

        response_tokens = st.slider("Expected response tokens", min_value=0, max_value=4096, value=300, step=10)
        models_selected = st.multiselect("Models to evaluate", model_list, default=model_list[:3])

        st.write("---")
        st.write("Estimates")
        rows = []
        for m in models_selected:
            total_tokens = (prompt_tokens or 0) + response_tokens
            if use_ml and model_obj and preproc_obj:
                y_pred = ml_predict_energy_wh(prompt_tokens, response_tokens, df, model_obj, preproc_obj)
                if y_pred is not None and np.isfinite(y_pred):
                    e_median = y_pred
                    e_low, e_high = np.nan, np.nan
                else:
                    e_median, e_low, e_high = predict_energy_wh_baseline(total_tokens, m, baseline_map)
            else:
                e_median, e_low, e_high = predict_energy_wh_baseline(total_tokens, m, baseline_map)

            co2 = to_co2e_kg(e_median, kg_per_kwh) if np.isfinite(e_median) else np.nan
            co2_lo = to_co2e_kg(e_low, kg_per_kwh) if np.isfinite(e_low) else np.nan
            co2_hi = to_co2e_kg(e_high, kg_per_kwh) if np.isfinite(e_high) else np.nan

            rows.append({
                "model_name": m,
                "total_tokens": total_tokens,
                "energy_wh_est": e_median,
                "energy_wh_lo": e_low,
                "energy_wh_hi": e_high,
                "co2e_kg_est": co2,
                "co2e_kg_lo": co2_lo,
                "co2e_kg_hi": co2_hi,
            })
        res_df = pd.DataFrame(rows)

        if not res_df.empty and res_df["energy_wh_est"].notna().any():
            st.dataframe(res_df, use_container_width=True)
            fig = px.bar(
                res_df.sort_values("energy_wh_est", ascending=False),
                x="model_name", y="energy_wh_est",
                error_y=res_df["energy_wh_hi"] - res_df["energy_wh_est"] if res_df["energy_wh_hi"].notna().any() else None,
                error_y_minus=res_df["energy_wh_est"] - res_df["energy_wh_lo"] if res_df["energy_wh_lo"].notna().any() else None,
                labels={"energy_wh_est": "Estimated energy (Wh)"},
                title="Estimated energy by model",
                color="model_name",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                res_df.sort_values("co2e_kg_est", ascending=False),
                x="model_name", y="co2e_kg_est",
                labels={"co2e_kg_est": "Estimated CO₂e (kg)"},
                title="Estimated CO₂e by model",
                color="model_name",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Provide a prompt and choose models to see estimates.")

    with colB:
        st.subheader("Baseline overview")
        if not stats_df.empty:
            st.dataframe(stats_df.rename(columns={
                "model_name": "model_name",
                "median": "median Wh/token",
                "q25": "p25 Wh/token",
                "q75": "p75 Wh/token"
            }), height=380)
        else:
            st.write("No baseline stats available yet.")

with tab2:
    st.subheader("Conversation simulation")
    n_turns = st.number_input("Number of turns", min_value=1, max_value=50, value=5, step=1)
    conv_models = st.multiselect("Model(s) for conversation", model_list, default=[model_list[0]] if model_list else [])
    default_resp = 200
    conv_rows = []
    for i in range(int(n_turns)):
        with st.expander(f"Turn {i+1}", expanded=(i==0)):
            txt = st.text_area(f"User message {i+1}", key=f"turn_txt_{i}")
            ptoks = estimate_tokens(txt) if txt else 0
            st.caption(f"Prompt tokens ≈ {ptoks}")
            rtoks = st.slider(f"Expected response tokens for turn {i+1}", 0, 4096, default_resp, 10, key=f"turn_rtoks_{i}")
            conv_rows.append({"turn": i+1, "prompt_tokens": ptoks, "response_tokens": rtoks})

    if conv_models:
        out_rows = []
        for m in conv_models:
            cum_wh = 0.0
            for r in conv_rows:
                tt = (r["prompt_tokens"] or 0) + (r["response_tokens"] or 0)
                e_mid, e_lo, e_hi = predict_energy_wh_baseline(tt, m, baseline_map)
                cum_wh += (e_mid or 0)
                out_rows.append({
                    "model_name": m,
                    "turn": r["turn"],
                    "turn_tokens": tt,
                    "turn_energy_wh": e_mid,
                    "cumulative_energy_wh": cum_wh,
                    "cumulative_co2e_kg": to_co2e_kg(cum_wh, kg_per_kwh),
                })
        conv_df = pd.DataFrame(out_rows)
        if not conv_df.empty:
            st.dataframe(conv_df, use_container_width=True)
            fig = px.line(
                conv_df, x="turn", y="cumulative_energy_wh", color="model_name",
                markers=True, labels={"cumulative_energy_wh": "Cumulative energy (Wh)", "turn": "Turn"},
                title="Cumulative energy over the conversation"
            )
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(
                conv_df, x="turn", y="cumulative_co2e_kg", color="model_name",
                markers=True, labels={"cumulative_co2e_kg": "Cumulative CO₂e (kg)", "turn": "Turn"},
                title="Cumulative CO₂e over the conversation"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.download_button("Download CSV", conv_df.to_csv(index=False).encode("utf-8"), "conversation_estimates.csv", "text/csv")
    else:
        st.info("Select at least one model to simulate a conversation.")

with tab3:
    st.subheader("Compare models")
    cmp_prompt = st.text_area("Prompt for comparison", placeholder="Optional. Leave blank to only use response tokens.", key="cmp_prompt")
    cmp_ptoks = estimate_tokens(cmp_prompt) if cmp_prompt else 0
    st.caption(f"Estimated prompt tokens: {cmp_ptoks}")
    cmp_rtoks = st.slider("Expected response tokens", 0, 4096, 300, 10, key="cmp_rtoks")
    cmp_models = st.multiselect("Models to compare", model_list, default=model_list[: min(8, len(model_list))])

    if cmp_models:
        rows = []
        for m in cmp_models:
            total_tokens = (cmp_ptoks or 0) + cmp_rtoks
            e_mid, e_lo, e_hi = predict_energy_wh_baseline(total_tokens, m, baseline_map)
            rows.append({
                "model_name": m,
                "total_tokens": total_tokens,
                "energy_wh_est": e_mid,
                "co2e_kg_est": to_co2e_kg(e_mid, kg_per_kwh) if np.isfinite(e_mid) else np.nan
            })
        cmp_df = pd.DataFrame(rows).sort_values("energy_wh_est", ascending=False)
        st.dataframe(cmp_df, use_container_width=True)
        fig = px.bar(
            cmp_df, x="model_name", y="energy_wh_est", color="model_name",
            labels={"energy_wh_est": "Estimated energy (Wh)"},
            title="Model comparison — energy per request"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select models to compare.")

# ------------- Footer -------------
st.caption("Baseline estimates use per-model median energy-per-token derived from your dataset. Upload an ML model to override.")