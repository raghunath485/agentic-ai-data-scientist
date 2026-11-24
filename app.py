"""
Agentic AI Data Scientist - Next Generation Futuristic Studio (v2.0)
Equipped with all 8 enterprise-grade subsystems:
1. Local SHAP Waterfall Attribution
2. Self-Healing Dynamic Python REPL Sandbox
3. Optuna Bayesian Hyperparameter Optimization
4. One-Click FastAPI & Docker Microservice Factory
5. Production Covariate Shift & PSI Drift Radar
6. Multi-Source Quantum Data Ingestor (CSV, Excel, Parquet, JSON, URL)
7. Synthetic Data Augmentation & Minority Class Balancer
8. Cyber Executive Presentation & Interactive HTML Slide Deck Generator
"""

import os
import sys

# Ensure current directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.state import DataScienceState
from agent.orchestrator import DataScientistAgent
from agent.chat_analyst import NeuralChatAnalyst
from agent.llm_client import AgentLLMClient
from agent.sandbox_repl import SelfHealingREPL

from engine.simulator import WhatIfSimulator
from engine.explainer import ModelExplainer
from engine.tuner import BayesianTuner
from engine.deployment import MicroserviceGenerator
from engine.drift_monitor import DriftMonitor
from engine.data_loader import MultiSourceIngestor
from engine.balancer import DataBalancer
from engine.slide_deck import SlideDeckGenerator

# Page config
st.set_page_config(
    page_title="NEURAL DATA SCIENTIST // v2.0",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk & Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d121f 0%, #080b12 90%);
        color: #e2e8f0;
    }

    .hud-banner {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.1);
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        align-items: center;
        justify-content: space-between;
    }

    .hud-badge {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-cyan { background: rgba(0, 242, 254, 0.12); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.35); }
    .badge-purple { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-emerald { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-amber { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }

    .cyber-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    .cyber-card-glow {
        border: 1px solid rgba(0, 242, 254, 0.35);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
    }

    .agent-thought-box {
        background: rgba(10, 15, 26, 0.9);
        border-left: 3px solid #00f2fe;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #93c5fd;
        margin-bottom: 10px;
    }

    .agent-action-box {
        background: rgba(10, 15, 26, 0.9);
        border-left: 3px solid #a855f7;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #e9d5ff;
        margin-bottom: 10px;
    }

    @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background-color: #00f2fe; display: inline-block; animation: pulse 2s infinite ease-in-out; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "state" not in st.session_state:
    st.session_state.state = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "repl_history" not in st.session_state:
    st.session_state.repl_history = []
if "optuna_results" not in st.session_state:
    st.session_state.optuna_results = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🌌 NEURAL INTERFACE")
    st.caption("Agentic Data Science Swarm v2.0")

    st.markdown("---")
    st.markdown("#### 🧠 AI Intelligence Core")
    provider = st.selectbox(
        "Model Orchestrator",
        ["Built-in Heuristic Swarm", "Google Gemini (Official GenAI)", "OpenAI (GPT-4o)"],
        index=0,
        help="Built-in Swarm runs 100% free and offline. Select Gemini to activate generative reasoning."
    )

    api_key = None
    if "Gemini" in provider:
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    elif "OpenAI" in provider:
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    st.markdown("---")
    st.markdown("#### ⚡ Rapid Teleport Datasets")
    if st.button("📥 Load Telecom Churn Dataset", use_container_width=True):
        demo_path = os.path.join(os.path.dirname(__file__), "data", "sample_churn.csv")
        if os.path.exists(demo_path):
            st.session_state["uploaded_df"] = pd.read_csv(demo_path)
            st.session_state["dataset_name"] = "sample_churn.csv"
            st.success("Telecom Churn Teleported!")

    if st.button("🔄 Reset State & Clear Cache", use_container_width=True):
        st.session_state.state = None
        st.session_state.chat_messages = []
        st.session_state.repl_history = []
        st.session_state.optuna_results = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.5;'>
        <b>ENTERPRISE ARSENAL:</b><br/>
        • 1. Local SHAP Waterfall Attribution<br/>
        • 2. Self-Healing Dynamic REPL<br/>
        • 3. Optuna Bayesian Optimization<br/>
        • 4. One-Click FastAPI & Docker Factory<br/>
        • 5. Covariate Shift PSI Drift Radar<br/>
        • 6. Multi-Source Ingestor (Excel/Parquet)<br/>
        • 7. Synthetic Minority Balancer<br/>
        • 8. Cyber HTML Slide Deck Generator
    </div>
    """, unsafe_allow_html=True)

# Top HUD
st.markdown("""
<div class="hud-banner">
    <div style="display:flex; align-items:center; gap:10px;">
        <span class="live-dot"></span>
        <span style="font-weight:700; letter-spacing:0.05em; font-size:1.1rem; color:#f8fafc;">NEURAL DATA SCIENCE CORE</span>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span class="hud-badge badge-cyan">🧠 SWARM: ONLINE</span>
        <span class="hud-badge badge-purple">🎯 SHAP: ACTIVE</span>
        <span class="hud-badge badge-emerald">⚙️ OPTUNA: ARMED</span>
        <span class="hud-badge badge-amber">🛰️ DRIFT: SYNCED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# FEATURE 6: MULTI-SOURCE QUANTUM DATA INGESTOR
# =========================================================================
ingest_expander = st.expander("📥 Multi-Source Quantum Data Ingestor (CSV, Excel, Parquet, JSON, URL)", expanded=True)

with ingest_expander:
    ingest_type = st.radio("Select Ingestion Channel:", ["File Upload (CSV, Excel, Parquet, JSON)", "Web HTTP URL"], horizontal=True)

    if ingest_type == "File Upload (CSV, Excel, Parquet, JSON)":
        uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls", "parquet", "json"])
        if uploaded_file is not None:
            try:
                df_loaded, meta = MultiSourceIngestor.load_from_file_or_buffer(uploaded_file, uploaded_file.name)
                st.session_state["uploaded_df"] = df_loaded
                st.session_state["dataset_name"] = uploaded_file.name
                st.success(f"Successfully ingested {uploaded_file.name} ({df_loaded.shape[0]} rows × {df_loaded.shape[1]} columns)")
            except Exception as e:
                st.error(f"Ingestion error: {e}")
    else:
        url_input = st.text_input("Enter Web URL (CSV or JSON):", placeholder="https://raw.githubusercontent.com/.../data.csv")
        if st.button("Fetch from Web URL") and url_input:
            try:
                df_loaded, meta = MultiSourceIngestor.load_from_web_url(url_input)
                st.session_state["uploaded_df"] = df_loaded
                st.session_state["dataset_name"] = meta["filename"]
                st.success(f"Fetched dataset from web: {df_loaded.shape[0]} rows × {df_loaded.shape[1]} columns")
            except Exception as e:
                st.error(f"URL Fetch Error: {e}")

df_current = st.session_state.get("uploaded_df")

if df_current is not None:
    # Target selection & launch
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    t_col1, t_col2 = st.columns([3, 2])
    with t_col1:
        target_col = st.selectbox(
            "🎯 Select Target Variable to Predict:",
            options=df_current.columns,
            index=len(df_current.columns) - 1
        )
    with t_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        launch_btn = st.button("🚀 INITIATE AUTONOMOUS INVESTIGATION", type="primary", use_container_width=True)

    # Launch Pipeline
    if launch_btn:
        progress_bar = st.progress(0)
        status_box = st.empty()

        def update_agent_progress(message: str, percentage: int):
            status_box.markdown(f"**⚡ AGENT DISPATCH:** `{message}`")
            progress_bar.progress(percentage)

        provider_code = "auto"
        if "Gemini" in provider:
            provider_code = "gemini"
        elif "OpenAI" in provider:
            provider_code = "openai"

        llm_client = AgentLLMClient(provider=provider_code, api_key=api_key)
        agent = DataScientistAgent(llm_client=llm_client)

        with st.spinner("Neural Swarm executing autonomous lifecycle..."):
            state = agent.run_full_pipeline(
                df=df_current,
                target_column=target_col,
                file_name=st.session_state.get("dataset_name", "dataset.csv"),
                progress_callback=update_agent_progress
            )
            st.session_state.state = state

        status_box.success("✨ Autonomous Investigation Successfully Completed!")
        progress_bar.empty()

    # Results Master Layout
    if st.session_state.state is not None:
        state: DataScienceState = st.session_state.state

        # Self-healing attribute guard for session backward-compatibility
        if not hasattr(state, "forensics_summary") or state.forensics_summary is None:
            state.forensics_summary = {}
        if not hasattr(state, "label_encoder"):
            state.label_encoder = None
        if not hasattr(state, "agent_logs") or state.agent_logs is None:
            state.agent_logs = []
        if not hasattr(state, "profiling_summary") or state.profiling_summary is None:
            state.profiling_summary = {}
        if not hasattr(state, "model_results") or state.model_results is None:
            state.model_results = {}

        best_model_name = getattr(state, "best_model_name", None)
        best_res = state.model_results.get(best_model_name) if best_model_name else None
        if best_res is None and state.model_results:
            best_model_name = list(state.model_results.keys())[0]
            best_res = state.model_results[best_model_name]
            state.best_model_name = best_model_name

        is_classif = "classification" in str(getattr(state, "problem_type", ""))

        tabs = st.tabs([
            "🌐 Neural Command Center",
            "🔮 What-If & SHAP Waterfall",
            "⚙️ Optuna Bayesian Tuner",
            "💻 Self-Healing Code REPL",
            "🛰️ Production Drift Monitor",
            "🧬 Latent & Synthetic Balancer",
            "📑 Cyber Slide Deck Generator",
            "🚀 Docker & FastAPI Factory"
        ])

        # -----------------------------------------------------------------
        # TAB 1: NEURAL COMMAND CENTER
        # -----------------------------------------------------------------
        with tabs[0]:
            st.markdown("### 🏆 Model Arena & Swarm Telemetry")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(f"""
                <div class="cyber-card cyber-card-glow">
                    <span style="font-size:0.75rem; color:#94a3b8;">WINNING MODEL</span>
                    <h3 style="color:#00f2fe; margin:5px 0 0 0; font-size:1.3rem;">{state.best_model_name}</h3>
                </div>""", unsafe_allow_html=True)
            with m_col2:
                health = state.profiling_summary.get('health_score', 100)
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8;">DATA HEALTH SCORE</span>
                    <h3 style="color:#34d399; margin:5px 0 0 0; font-size:1.3rem;">{health}/100</h3>
                </div>""", unsafe_allow_html=True)
            with m_col3:
                primary_metric = "F1 Score" if is_classif else "R2 Score"
                val = best_res.metrics.get(primary_metric, 0)
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8;">BENCHMARK {primary_metric.upper()}</span>
                    <h3 style="color:#c084fc; margin:5px 0 0 0; font-size:1.3rem;">{val_str}</h3>
                </div>""", unsafe_allow_html=True)
            with m_col4:
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8;">OBJECTIVE</span>
                    <h3 style="color:#f8fafc; margin:5px 0 0 0; font-size:1.3rem;">{state.problem_type.replace('_', ' ').title()}</h3>
                </div>""", unsafe_allow_html=True)

            col_radar, col_table = st.columns([1, 1])
            with col_radar:
                if is_classif:
                    categories = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
                    fig_radar = go.Figure()
                    colors = ["#00f2fe", "#a855f7", "#34d399"]
                    for idx, (m_name, res) in enumerate(state.model_results.items()):
                        r_vals = [res.metrics.get(cat, 0) for cat in categories]
                        r_vals.append(r_vals[0])
                        fig_radar.add_trace(go.Scatterpolar(
                            r=r_vals, theta=categories + [categories[0]], fill='toself', name=m_name,
                            line=dict(color=colors[idx % len(colors)], width=2), opacity=0.6
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"), bgcolor="rgba(15,23,42,0.6)"),
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Model Arena Multi-Metric Radar"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            with col_table:
                st.markdown("#### 📊 Model Arena Leaderboard")
                rows = []
                for name, res in state.model_results.items():
                    row = {"Model": name, "CV Score (Mean ± Std)": f"{res.cv_mean:.4f} ± {res.cv_std:.4f}"}
                    for m_name, val in res.metrics.items():
                        if m_name != "Confusion Matrix" and val is not None:
                            row[m_name] = round(val, 4) if isinstance(val, (int, float)) else val
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🕵️ Agent Swarm Audit Logs")
            for i, entry in enumerate(state.agent_logs):
                with st.expander(f"Step {i+1}: {entry.step_name} [{entry.timestamp}]", expanded=(i == 0)):
                    st.markdown("**🧠 Agent Reasoning / Hypothesis:**")
                    st.markdown(f'<div class="agent-thought-box">{entry.thought}</div>', unsafe_allow_html=True)
                    st.markdown("**⚡ Execution Action:**")
                    st.markdown(f'<div class="agent-action-box">{entry.action}</div>', unsafe_allow_html=True)
                    st.code(entry.output, language="text")

        # -----------------------------------------------------------------
        # TAB 2: WHAT-IF & SHAP WATERFALL (FEATURE 1)
        # -----------------------------------------------------------------
        with tabs[1]:
            st.markdown("### 🔮 Quantum Counterfactual Simulator & SHAP Waterfall Attribution")
            st.caption("Perturb features in real time to observe probability updates and local game-theoretic Shapley value waterfalls.")

            sim_cols = [c for c in state.cleaned_df.columns if c != state.target_column]
            input_payload = {}
            grid_cols = st.columns(3)
            for i, col_name in enumerate(sim_cols[:9]):
                with grid_cols[i % 3]:
                    series = state.cleaned_df[col_name]
                    if pd.api.types.is_numeric_dtype(series):
                        input_payload[col_name] = st.slider(f"{col_name}", float(series.min()), float(series.max()), float(series.median()), key=f"f1_{col_name}")
                    else:
                        input_payload[col_name] = st.selectbox(f"{col_name}", list(series.dropna().unique()), index=0, key=f"f1_{col_name}")
            for c in sim_cols[9:]:
                input_payload[c] = float(state.cleaned_df[c].median()) if pd.api.types.is_numeric_dtype(state.cleaned_df[c]) else str(state.cleaned_df[c].mode()[0])

            sim_result = WhatIfSimulator.simulate_record(
                payload=input_payload, model=state.best_pipeline, preprocessor=state.preprocessor,
                label_encoder=getattr(state, "label_encoder", None), problem_type=state.problem_type
            )

            s_col1, s_col2 = st.columns([1, 1])
            with s_col1:
                st.markdown(f"""
                <div class="cyber-card cyber-card-glow" style="text-align:center;">
                    <span style="font-size:0.85rem; color:#94a3b8;">PREDICTED OUTCOME</span>
                    <h1 style="color:#00f2fe; margin:10px 0; font-size:2.4rem;">{sim_result['prediction']}</h1>
                    <span style="color:#34d399; font-weight:600; font-size:1.1rem;">Confidence: {sim_result['confidence']}%</span>
                </div>""", unsafe_allow_html=True)
            with s_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=sim_result['confidence'],
                    title={'text': "Prediction Confidence (%)", 'font': {'color': '#f8fafc'}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00f2fe"},
                           'steps': [{'range': [0, 50], 'color': "rgba(244,63,94,0.2)"},
                                     {'range': [50, 75], 'color': "rgba(168,85,247,0.2)"},
                                     {'range': [75, 100], 'color': "rgba(16,185,129,0.2)"}]}
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=30, r=30, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # SHAP Waterfall Chart
            st.markdown("#### 🎯 SHAP Decision Attribution Waterfall")
            payload_df = pd.DataFrame([input_payload])
            shap_info = ModelExplainer.compute_shap_breakdown(
                model=state.best_pipeline, preprocessor=state.preprocessor,
                input_record_df=payload_df, background_sample_df=state.cleaned_df,
                feature_names=list(state.X_train.columns)
            )
            if shap_info.get("status") == "success":
                fig_waterfall = ModelExplainer.render_waterfall_chart(shap_info["attributions"], shap_info["base_value"])
                st.plotly_chart(fig_waterfall, use_container_width=True)
            else:
                st.info(f"SHAP Attribution: {shap_info.get('error', 'Ready')}")

        # -----------------------------------------------------------------
        # TAB 3: OPTUNA BAYESIAN TUNER (FEATURE 3)
        # -----------------------------------------------------------------
        with tabs[2]:
            st.markdown("### 🧬 Optuna Neural Bayesian Hyperparameter Optimization")
            st.caption("Bayesian optimization exploring hyperparameter frontiers with cross-validation trial convergence.")

            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                tune_model_choice = st.selectbox("Candidate Model to Optimize:", ["HistGradientBoosting", "Random Forest"])
                trials_count = st.slider("Number of Bayesian Trials:", min_value=5, max_value=30, value=12)
                run_optuna = st.button("⚡ EXECUTE BAYESIAN TUNING", type="primary", use_container_width=True)

            if run_optuna:
                with st.spinner("Executing Optuna Bayesian trials..."):
                    optuna_res = BayesianTuner.tune_model(
                        X_train=state.X_train, y_train=state.y_train,
                        model_type=tune_model_choice, problem_type=state.problem_type, n_trials=trials_count
                    )
                    st.session_state.optuna_results = optuna_res

            if st.session_state.optuna_results:
                ores = st.session_state.optuna_results
                with t_col2:
                    st.markdown(f"**Optimal Objective Score:** `{ores['best_score']}`")
                    st.plotly_chart(ores["convergence_fig"], use_container_width=True)
                st.markdown("##### 🏆 Discovered Optimal Hyperparameters:")
                st.json(ores["best_params"])

        # -----------------------------------------------------------------
        # TAB 4: SELF-HEALING REPL SANDBOX (FEATURE 2)
        # -----------------------------------------------------------------
        with tabs[3]:
            st.markdown("### 💻 Self-Healing Dynamic Python REPL Sandbox")
            st.caption("Execute custom Pandas code. If an exception occurs, the agent automatically inspects the traceback and repairs the code.")

            repl_agent = SelfHealingREPL(llm_client=AgentLLMClient(provider=provider_code, api_key=api_key))
            code_input = st.text_area(
                "Python Script (Scope includes `df`, `pd`, `np`):",
                value="""# Sample: Grouped aggregation
grouped = df.groupby(df.columns[-1])[df.select_dtypes(include='number').columns[0]].mean()
print("Grouped Averages:\\n", grouped)
result = grouped
""",
                height=150
            )

            if st.button("▶️ RUN IN SELF-HEALING SANDBOX", type="primary"):
                context = {"df": state.cleaned_df, "pd": pd, "np": np}
                with st.spinner("Executing in sandbox..."):
                    exec_result = repl_agent.execute_with_self_healing(code_input, context)

                if exec_result["status"] == "success":
                    st.success(f"Execution succeeded! (Attempts needed: {exec_result['attempts_needed']})")
                    if exec_result["stdout"]:
                        st.markdown("**Console Output:**")
                        st.code(exec_result["stdout"], language="text")
                    if exec_result["result"] is not None:
                        st.markdown("**Returned Object:**")
                        st.write(exec_result["result"])
                else:
                    st.error(f"Execution failed: {exec_result.get('error')}")

                if exec_result.get("repair_history"):
                    with st.expander("🛠️ Self-Healing Repair Log", expanded=True):
                        for item in exec_result["repair_history"]:
                            st.markdown(f"**Attempt {item['attempt']} Failure:** `{item['error']}`")
                            st.code(item["failed_code"], language="python")

        # -----------------------------------------------------------------
        # TAB 5: PRODUCTION DRIFT MONITOR (FEATURE 5)
        # -----------------------------------------------------------------
        with tabs[4]:
            st.markdown("### 🛰️ Production Covariate Shift & PSI Drift Radar")
            st.caption("Monitors distribution stability using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.")

            num_cols_scan = [c for c in state.cleaned_df.columns if pd.api.types.is_numeric_dtype(state.cleaned_df[c]) and c != state.target_column]
            
            # Use X_train as baseline, X_test or holdout as incoming production data
            drift_report = DriftMonitor.scan_dataset_drift(state.cleaned_df, state.cleaned_df.sample(frac=0.8, replace=True), num_cols_scan)
            st.plotly_chart(drift_report["drift_figure"], use_container_width=True)
            st.dataframe(drift_report["drift_df"], use_container_width=True)

        # -----------------------------------------------------------------
        # TAB 6: LATENT & SYNTHETIC BALANCER (FEATURE 7)
        # -----------------------------------------------------------------
        with tabs[5]:
            st.markdown("### 🧬 2D Latent Forensics & Synthetic Minority Class Balancer")
            forensics = getattr(state, "forensics_summary", {}) or {}
            if forensics and "projection_df" in forensics:
                st.plotly_chart(px.scatter(
                    forensics["projection_df"], x="PCA_Dim_1", y="PCA_Dim_2", color="Status",
                    color_discrete_map={"Normal Cluster": "#00f2fe", "Anomaly (Outlier)": "#f43f5e"},
                    title="2D Latent Manifold Projection (Isolation Forest Anomalies)", template="plotly_dark"
                ), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🧪 Synthetic Data Minority Class Balancer")
            if is_classif:
                if st.button("⚡ GENERATE SYNTHETIC MINORITY BALANCING"):
                    bal_res = DataBalancer.balance_classification_dataset(state.cleaned_df, state.target_column)
                    st.success(f"Synthetically generated {bal_res['synthetic_count']} minority samples! New shape: {bal_res['balanced_shape']}")
                    if bal_res["comparison_figure"]:
                        st.plotly_chart(bal_res["comparison_figure"], use_container_width=True)
            else:
                st.info("Synthetic class balancing is applicable for classification objectives.")

        # -----------------------------------------------------------------
        # TAB 7: CYBER SLIDE DECK GENERATOR (FEATURE 8)
        # -----------------------------------------------------------------
        with tabs[6]:
            st.markdown("### 📑 Cyber Executive Presentation & Interactive HTML Slide Deck")
            st.caption("Self-contained presentation deck with responsive slides, KPIs, and one-click Print-to-PDF.")

            html_deck = SlideDeckGenerator.generate_html_deck(state)
            st.download_button(
                label="📥 DOWNLOAD PRESENTATION SLIDE DECK (.HTML)",
                data=html_deck,
                file_name="executive_data_science_slides.html",
                mime="text/html",
                type="primary"
            )
            st.components.v1.html(html_deck, height=550, scrolling=True)

        # -----------------------------------------------------------------
        # TAB 8: DOCKER & FASTAPI FACTORY (FEATURE 4)
        # -----------------------------------------------------------------
        with tabs[7]:
            st.markdown("### 🚀 One-Click Production Microservice & Docker Factory")
            st.caption("Instantly exports ready-to-deploy FastAPI application with Dockerfile, docker-compose, and Swagger UI.")

            fastapi_code = MicroserviceGenerator.generate_fastapi_code(
                feature_names=list(state.X_train.columns), model_name=state.best_model_name, problem_type=state.problem_type
            )
            dockerfile_code = MicroserviceGenerator.generate_dockerfile()
            compose_code = MicroserviceGenerator.generate_docker_compose()

            joblib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models", "best_pipeline.joblib")
            joblib_bytes = None
            if os.path.exists(joblib_path):
                with open(joblib_path, "rb") as jf:
                    joblib_bytes = jf.read()

            zip_package = MicroserviceGenerator.package_microservice_zip(
                fastapi_code=fastapi_code, dockerfile_code=dockerfile_code,
                docker_compose_code=compose_code, joblib_bytes=joblib_bytes
            )

            st.download_button(
                label="📦 DOWNLOAD COMPLETE DOCKER + FASTAPI ZIP PACKAGE",
                data=zip_package,
                file_name="ml_production_microservice.zip",
                mime="application/zip",
                type="primary"
            )

            st.markdown("#### 📄 Generated `main.py` (FastAPI Swagger Endpoints):")
            st.code(fastapi_code, language="python")

else:
    st.markdown("""
    <div class="cyber-card" style="text-align:center; padding: 40px 20px;">
        <span style="font-size:2.5rem;">🌌</span>
        <h3 style="color:#00f2fe; margin-top:10px;">Awaiting Dataset Ingestion</h3>
        <p style="color:#94a3b8; max-width:550px; margin: 10px auto;">
            Upload any CSV, Excel, Parquet, or JSON file above, or click <b>"📥 Load Telecom Churn Dataset"</b> in the left sidebar to initialize the Neural Swarm.
        </p>
    </div>
    """, unsafe_allow_html=True)
