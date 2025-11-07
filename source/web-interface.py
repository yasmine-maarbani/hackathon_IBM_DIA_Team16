# LLM Energy & CO₂e Estimator (ML Pipelines)
# - Default loads models from repo paths under artifacts/models/
# - upload (.joblib/.pkl) for ad-hoc testing
# - Smart response-token slider (auto median per model, dynamic q95 bounds)
# - Builds exact feature rows from your pipelines' ColumnTransformer ("pre" step)

import os
import io
from typing import Optional, Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Estimation and Simulation of the CO₂Impact of LLM Queries", page_icon="🌱", layout="wide")

# --- Config (cross-platform, repo-relative) ---
from pathlib import Path

# This file lives in <repo>/source/web-interface.py -> repo root is parent of 'source'
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATHS = [
    str(REPO_ROOT / "artifacts" / "processed.csv"),                      # repo default
    str(REPO_ROOT / "llm_inference_energy_consumption_final.csv"),       # optional alt name at repo root
]

# Models inside repo
DEFAULT_MODEL_REQ = str(REPO_ROOT / "artifacts" / "models" / "model_energy_per_request_linreg.joblib")  # kWh/request
DEFAULT_MODEL_TOK = str(REPO_ROOT / "artifacts" / "models" / "model_energy_per_token_linreg.joblib")    # kWh/token

CARBON_PRESETS = {
    "Global avg (~0.475)": 0.475,
    "EU avg (~0.25)": 0.25,
    "US avg (~0.40)": 0.40,
    "India (~0.70)": 0.70,
    "Renewable-heavy (~0.05)": 0.05,
    "Custom": None,
}

# ================= Utils =================
@st.cache_data(show_spinner=False)
def load_csv(uploaded: Optional[io.BytesIO], default_paths: List[str]) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame()
        for p in default_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
    if not df.empty:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

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

def to_co2e_kg(energy_kwh: float, kg_per_kwh: float) -> float:
    return energy_kwh * kg_per_kwh

def ensure_seconds_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_seconds(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return s
        med = np.nanmedian(s)
        return s/1e9 if (np.isfinite(med) and med > 1e6) else s
    work = df.copy()
    for c in ["total_duration", "load_duration", "prompt_duration", "response_duration"]:
        if c in work.columns and (c + "_s") not in work.columns:
            work[c + "_s"] = to_seconds(work[c])
    return work

def extract_expected_columns(pipe) -> Tuple[List[str], List[str]]:
    """
    Read numeric/categorical columns from the ColumnTransformer ('pre' step) inside your Pipeline.
    Assumes transformers named 'num' and 'cat'.
    """
    try:
        pre = pipe.named_steps["pre"]
    except Exception:
        return [], []
    expected_num, expected_cat = [], []
    for name, transformer, cols in getattr(pre, "transformers_", []):
        if name == "num":
            expected_num = list(cols)
        elif name == "cat":
            expected_cat = list(cols)
    return expected_num, expected_cat

@st.cache_data(show_spinner=False)
def compute_fill_stats(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[Dict, Dict]:
    med_num = {}
    for c in num_cols:
        if c in df.columns:
            med_num[c] = pd.to_numeric(df[c], errors="coerce").median()
        else:
            med_num[c] = np.nan
    mode_cat = {}
    for c in cat_cols:
        if c in df.columns and df[c].notna().any():
            mode_cat[c] = df[c].mode()[0]
        else:
            mode_cat[c] = None
    return med_num, mode_cat

def default_rtoks_for_model(df: pd.DataFrame, model_name: Optional[str], fallback: int = 300) -> int:
    if df.empty or "response_token_length" not in df.columns:
        return fallback
    try:
        s = None
        if model_name and "model_name" in df.columns:
            s = pd.to_numeric(df.loc[df["model_name"] == model_name, "response_token_length"], errors="coerce").dropna()
            if s.empty:
                s = None
        if s is None:
            s = pd.to_numeric(df["response_token_length"], errors="coerce").dropna()
        med = float(s.median()) if len(s) else fallback
        return int(np.clip(med if np.isfinite(med) else fallback, 10, 2000))
    except Exception:
        return fallback

def get_response_slider_bounds(df: pd.DataFrame,
                               model_name: Optional[str],
                               fallback_default: int = 300,
                               hard_min: int = 0,
                               hard_max: int = 4096) -> Tuple[int, int, int]:
    if df.empty or "response_token_length" not in df.columns:
        return hard_min, hard_max, fallback_default
    try:
        s = None
        if model_name and "model_name" in df.columns:
            s = pd.to_numeric(df.loc[df["model_name"] == model_name, "response_token_length"], errors="coerce").dropna()
            if s.empty:
                s = None
        if s is None:
            s = pd.to_numeric(df["response_token_length"], errors="coerce").dropna()

        med = float(s.median()) if len(s) else fallback_default
        q95 = float(s.quantile(0.95)) if len(s) else fallback_default * 3

        default_val = int(np.clip(med if np.isfinite(med) else fallback_default, 10, hard_max))
        max_val = int(np.clip(q95 if np.isfinite(q95) else fallback_default * 3, 50, hard_max))

        if default_val > max_val:
            default_val = max_val
        return hard_min, max_val, default_val
    except Exception:
        return hard_min, hard_max, fallback_default

def build_feature_row(prompt: str,
                      response_tokens: int,
                      model_name_feature: Optional[str],
                      type_feature: Optional[str],
                      num_expected: List[str],
                      cat_expected: List[str],
                      med_num: Dict,
                      mode_cat: Dict) -> Tuple[pd.DataFrame, int, int, int]:
    pt = int(estimate_tokens(prompt))
    rt = int(response_tokens) if "response_token_length" in num_expected else 0
    tt = pt + rt

    row = {}
    # Fill numeric
    for c in num_expected:
        if c == "prompt_token_length":
            row[c] = float(pt)
        elif c == "response_token_length":
            row[c] = float(rt)
        elif c == "total_tokens":
            row[c] = float(max(1, tt))
        else:
            val = med_num.get(c, 0.0)
            if pd.isna(val):
                val = 0.0
            row[c] = float(val)
    # Fill categorical
    for c in cat_expected:
        if c == "model_name":
            row[c] = model_name_feature or mode_cat.get(c, "unknown")
        elif c == "type":
            row[c] = type_feature or mode_cat.get(c, "unknown")
        else:
            row[c] = mode_cat.get(c, "unknown")

    X = pd.DataFrame([[row.get(c) for c in (num_expected + cat_expected)]],
                     columns=(num_expected + cat_expected))
    return X, pt, rt, tt

# ================= Sidebar: Data & Models =================
with st.sidebar:
    st.header("Data & Models")

    up_csv = st.file_uploader("Upload dataset CSV (optional)", type=["csv"])
    df = load_csv(up_csv, DEFAULT_DATA_PATHS)
    if not df.empty:
        df = ensure_seconds_columns(df)

    if df.empty:
        st.warning("No dataset loaded. Upload a CSV or place one at artifacts/processed.csv")
    else:
        st.success(f"Dataset loaded: {len(df):,} rows")

    st.subheader("Pipelines")
    source = st.radio(
        "Model source",
        options=["Repo path", "Upload"],
        index=0,
        help="Prefer Repo path for reproducibility; use Upload for ad-hoc models."
    )

    mdl_req, mdl_tok = None, None

    @st.cache_resource(show_spinner=False)
    def load_model_from_path(path: str):
        return joblib.load(path)

    @st.cache_resource(show_spinner=False)
    def load_model_from_upload(uploaded):
        data = uploaded.read()
        return joblib.load(io.BytesIO(data))

    if source == "Repo path":
        req_path = st.text_input("Per-request model path", DEFAULT_MODEL_REQ)
        tok_path = st.text_input("Per-token model path", DEFAULT_MODEL_TOK)
        if st.button("Load models", type="primary", use_container_width=True):
            try:
                assert os.path.exists(req_path), f"Missing: {req_path}"
                mdl_req = load_model_from_path(req_path)
                st.success(f"Loaded per-request pipeline from {req_path}")
            except Exception as e:
                st.error(f"Per-request load failed: {e}")
            try:
                assert os.path.exists(tok_path), f"Missing: {tok_path}"
                mdl_tok = load_model_from_path(tok_path)
                st.success(f"Loaded per-token pipeline from {tok_path}")
            except Exception as e:
                st.error(f"Per-token load failed: {e}")
    else:
        up_req = st.file_uploader("Upload per-request (.joblib/.pkl)", type=["joblib", "pkl"])
        up_tok = st.file_uploader("Upload per-token (.joblib/.pkl)", type=["joblib", "pkl"])
        if up_req:
            try:
                mdl_req = load_model_from_upload(up_req)
                st.success("Loaded per-request pipeline (upload)")
            except Exception as e:
                st.error(f"Per-request upload load failed: {e}")
        if up_tok:
            try:
                mdl_tok = load_model_from_upload(up_tok)
                st.success("Loaded per-token pipeline (upload)")
            except Exception as e:
                st.error(f"Per-token upload load failed: {e}")

    st.subheader("Carbon intensity")
    grid_choice = st.selectbox("Grid CO₂ intensity (kg/kWh)", list(CARBON_PRESETS.keys()), index=0)
    if grid_choice == "Custom":
        kg_per_kwh = st.number_input("Custom kgCO₂e per kWh", min_value=0.0, value=0.475, step=0.01)
    else:
        kg_per_kwh = CARBON_PRESETS[grid_choice]
    st.caption(f"Using {kg_per_kwh:.3f} kgCO₂e/kWh")

# Persist models if loaded this run
if mdl_req is not None:
    st.session_state["mdl_req"] = mdl_req
if mdl_tok is not None:
    st.session_state["mdl_tok"] = mdl_tok

# Retrieve models from session if available
mdl_req = st.session_state.get("mdl_req")
mdl_tok = st.session_state.get("mdl_tok")

# Derive expected columns and fill stats
expected_num_req, expected_cat_req = extract_expected_columns(mdl_req) if mdl_req else ([], [])
expected_num_tok, expected_cat_tok = extract_expected_columns(mdl_tok) if mdl_tok else ([], [])
expected_num = sorted(list(set(expected_num_req + expected_num_tok)))
expected_cat = sorted(list(set(expected_cat_req + expected_cat_tok)))
med_num, mode_cat = compute_fill_stats(df, expected_num, expected_cat) if not df.empty else ({}, {})

models_available = sorted(df["model_name"].dropna().unique()) if ("model_name" in df.columns) else []
types_available  = sorted(df["type"].dropna().unique())       if ("type" in df.columns) else []

# ================= Main UI =================
st.title("🌱 Estimation and Simulation of the CO₂Impact of LLM Queries")
st.write("Predict energy (kWh) and CO₂e (kg) using our trained per-request and per-token pipelines. Token slider is auto-seeded from our dataset medians.")

tab1, tab2, tab3, tab4 = st.tabs(["Single Request", "Conversation", "Batch Simulate", "Compare Pipelines"])

# ---------- Single Request ----------
with tab1:
    colL, colR = st.columns([2, 1], gap="large")
    with colL:
        st.subheader("Inputs")
        prompt = st.text_area("Prompt", placeholder="Paste your prompt here...", height=140)

        # Optional categorical features used by your pipelines
        model_feat = st.selectbox("model_name (feature)", ["<none>"] + models_available, index=0) if models_available else "<none>"
        type_feat  = st.selectbox("type (feature)", ["<none>"] + types_available, index=0) if types_available else "<none>"
        model_feat = None if model_feat == "<none>" else model_feat
        type_feat  = None if type_feat == "<none>" else type_feat

        # Smart response tokens
        st.markdown("#### Response length")
        auto_len = st.checkbox("Auto (dataset median)", value=True,
                               help="Uses median response_token_length for the selected model (or global median).")
        if auto_len:
            response_tokens = default_rtoks_for_model(df, model_feat, fallback=300)
            st.info(f"Auto response tokens = {response_tokens} (dataset median)")
        else:
            min_rt, max_rt, def_rt = get_response_slider_bounds(df, model_feat, fallback_default=300)
            response_tokens = st.slider("Expected response tokens",
                                        min_value=min_rt, max_value=max_rt, value=def_rt, step=10)

        # Build feature row
        X_row, pt, rt, tt = build_feature_row(
            prompt, response_tokens, model_feat, type_feat,
            expected_num, expected_cat, med_num, mode_cat
        )
        st.caption(f"Estimated prompt tokens: {pt} | Response tokens: {rt} | Total: {tt}")

        results = []
        # Per-request model
        if mdl_req:
            try:
                e_req_kwh = float(mdl_req.predict(X_row)[0])
                results.append({"source": "Per-request pipeline", "energy_kWh": e_req_kwh,
                                "energy_Wh": e_req_kwh * 1000, "co2e_kg": to_co2e_kg(e_req_kwh, kg_per_kwh)})
            except Exception as e:
                st.error(f"Per-request prediction failed: {e}")
        # Per-token model
        if mdl_tok:
            try:
                e_tok_kwh = float(mdl_tok.predict(X_row)[0])
                e_req_from_tok_kwh = e_tok_kwh * tt
                results.append({"source": "Per-token × tokens", "energy_kWh": e_req_from_tok_kwh,
                                "energy_Wh": e_req_from_tok_kwh * 1000, "co2e_kg": to_co2e_kg(e_req_from_tok_kwh, kg_per_kwh)})
            except Exception as e:
                st.error(f"Per-token prediction failed: {e}")

        if results:
            res_df = pd.DataFrame(results)
            st.subheader("Estimates")
            st.dataframe(res_df, use_container_width=True)
            fig = px.bar(res_df, x="source", y="energy_kWh", color="source",
                         labels={"energy_kWh": "Energy (kWh)"}, title="Energy per request")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load at least one pipeline to see results.")

    with colR:
        st.subheader("Feature row (preview)")
        if expected_num or expected_cat:
            st.dataframe(X_row, use_container_width=True)
        else:
            st.info("Load pipelines to see expected features here.")

# ---------- Conversation ----------
with tab2:
    st.subheader("Simulate a conversation")
    turns = st.number_input("Number of turns", min_value=1, max_value=30, value=5, step=1)
    carry_context = st.checkbox("Carry context (each turn's prompt includes prior text)", value=True)

    conv_rows = []
    prior_text = ""
    for i in range(int(turns)):
        with st.expander(f"Turn {i+1}", expanded=(i == 0)):
            txt = st.text_area(f"User message {i+1}", key=f"txt_{i}")
            prompt_text = (prior_text + "\n" + txt) if (carry_context and prior_text) else txt
            pt_i = estimate_tokens(prompt_text) if prompt_text else 0
            default_rt = default_rtoks_for_model(df, None, fallback=200) if not df.empty else 200
            rt_i = st.slider(f"Response tokens for turn {i+1}", 0, 4096, default_rt, 10, key=f"rt_{i}")
            conv_rows.append((pt_i, rt_i))
            prior_text = (prior_text + "\n" + txt).strip()

    results = []
    cum_kwh = 0.0
    for i, (pt_i, rt_i) in enumerate(conv_rows, start=1):
        X_i, _, _, tt_i = build_feature_row(
            "", rt_i, model_name_feature=None, type_feature=None,
            num_expected=expected_num, cat_expected=expected_cat,
            med_num=med_num, mode_cat=mode_cat
        )
        # Overwrite token columns if present
        if "prompt_token_length" in expected_num: X_i["prompt_token_length"] = float(pt_i)
        if "response_token_length" in expected_num: X_i["response_token_length"] = float(rt_i)
        if "total_tokens" in expected_num: X_i["total_tokens"] = float(max(1, pt_i + rt_i))
        tt_i = pt_i + rt_i

        e_turns = []
        if mdl_req:
            try:
                e_turns.append(float(mdl_req.predict(X_i)[0]))
            except Exception:
                pass
        if mdl_tok:
            try:
                e_tok = float(mdl_tok.predict(X_i)[0])
                e_turns.append(e_tok * tt_i)
            except Exception:
                pass

        if e_turns:
            e_kwh = float(np.nanmean(e_turns))
            cum_kwh += e_kwh
            results.append({
                "turn": i,
                "turn_tokens": tt_i,
                "turn_energy_kWh": e_kwh,
                "cumulative_energy_kWh": cum_kwh,
                "cumulative_co2e_kg": to_co2e_kg(cum_kwh, kg_per_kwh)
            })

    if results:
        conv_df = pd.DataFrame(results)
        st.dataframe(conv_df, use_container_width=True)
        fig = px.line(conv_df, x="turn", y="cumulative_energy_kWh", markers=True,
                      labels={"cumulative_energy_kWh": "Cumulative energy (kWh)"},
                      title="Cumulative energy over turns")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download CSV", conv_df.to_csv(index=False).encode("utf-8"),
                           "conversation_estimates.csv", "text/csv")
    else:
        st.info("Enter turns and ensure at least one pipeline is loaded.")

# ---------- Batch Simulate ----------
with tab3:
    st.subheader("Batch simulation")
    st.write("Upload a CSV with columns: prompt,response_tokens,[optional: model_name,type].")
    up_batch = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
    if up_batch and (mdl_req or mdl_tok):
        dfb = pd.read_csv(up_batch)
        out = []
        for _, row in dfb.iterrows():
            prompt_b = str(row.get("prompt", ""))
            rtoks_b = int(row.get("response_tokens", 300))
            model_b = row.get("model_name", None)
            type_b = row.get("type", None)
            X_b, pt_b, rt_b, tt_b = build_feature_row(
                prompt_b, rtoks_b, model_b, type_b,
                expected_num, expected_cat, med_num, mode_cat
            )
            # Predict with available pipelines
            if mdl_req:
                try:
                    e_kwh = float(mdl_req.predict(X_b)[0])
                    out.append({"prompt": prompt_b[:64]+"..." if len(prompt_b) > 64 else prompt_b,
                                "response_tokens": rtoks_b, "source": "Per-request",
                                "energy_kWh": e_kwh, "energy_Wh": e_kwh * 1000,
                                "co2e_kg": to_co2e_kg(e_kwh, kg_per_kwh)})
                except Exception:
                    pass
            if mdl_tok:
                try:
                    e_tok = float(mdl_tok.predict(X_b)[0])
                    e_req = e_tok * tt_b
                    out.append({"prompt": prompt_b[:64]+"..." if len(prompt_b) > 64 else prompt_b,
                                "response_tokens": rtoks_b, "source": "Per-token×tokens",
                                "energy_kWh": e_req, "energy_Wh": e_req * 1000,
                                "co2e_kg": to_co2e_kg(e_req, kg_per_kwh)})
                except Exception:
                    pass
        if out:
            ob = pd.DataFrame(out)
            st.dataframe(ob, use_container_width=True)
            st.download_button("Download results CSV", ob.to_csv(index=False).encode("utf-8"),
                               "batch_results.csv", "text/csv")
        else:
            st.warning("No predictions produced. Ensure pipelines are loaded and CSV has required columns.")
    elif up_batch:
        st.info("Load at least one pipeline to run batch simulation.")

# ---------- Compare Pipelines ----------
with tab4:
    st.subheader("Compare pipelines")
    cmp_prompt = st.text_area("Prompt for comparison", placeholder="Short prompt...")
    # Dynamic slider bounds (global, since model selection is optional here)
    min_rt, max_rt, def_rt = get_response_slider_bounds(df, None, fallback_default=300)
    cmp_rtoks = st.slider("Expected response tokens", min_value=min_rt, max_value=max_rt, value=def_rt, step=10)
    X_cmp, pt_c, rt_c, tt_c = build_feature_row(
        cmp_prompt, cmp_rtoks, None, None,
        expected_num, expected_cat, med_num, mode_cat
    )
    rows = []
    if mdl_req:
        try:
            e_req = float(mdl_req.predict(X_cmp)[0])
            rows.append({"pipeline": "Per-request", "energy_kWh": e_req})
        except Exception:
            pass
    if mdl_tok:
        try:
            e_tok = float(mdl_tok.predict(X_cmp)[0])
            rows.append({"pipeline": "Per-token×tokens", "energy_kWh": e_tok * tt_c})
        except Exception:
            pass
    if rows:
        comp = pd.DataFrame(rows)
        comp["co2e_kg"] = comp["energy_kWh"].apply(lambda x: to_co2e_kg(x, kg_per_kwh))
        st.dataframe(comp, use_container_width=True)
        fig = px.bar(comp, x="pipeline", y="energy_kWh", color="pipeline",
                     labels={"energy_kWh": "Energy (kWh)"}, title="Pipeline comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load pipelines and enter a prompt to compare.")

st.caption("Models predict energy in kWh. The app shows kWh/Wh and converts to CO₂e using the selected grid intensity.")