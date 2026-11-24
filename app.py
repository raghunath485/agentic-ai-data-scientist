"""
Agentic AI Data Scientist - Next Generation Futuristic Studio (v2.0)
High-tech Glassmorphism UI, Neural Swarm Telemetry, 2D Latent Forensics,
Quantum "What-If" Counterfactual Simulator, and Conversational Data Intelligence.
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
from engine.simulator import WhatIfSimulator

# Set futuristic page configuration
st.set_page_config(
    page_title="NEURAL DATA SCIENTIST // v2.0",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk / Futuristic Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background & headers */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d121f 0%, #080b12 90%);
        color: #e2e8f0;
    }

    /* Futuristic HUD banner */
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

    .badge-cyan {
        background: rgba(0, 242, 254, 0.12);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.35);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Glass card container */
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

    /* Agent Thought Terminal */
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

    /* Pulse animation */
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00f2fe;
        display: inline-block;
        animation: pulse 2s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "state" not in st.session_state:
    st.session_state.state = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "what_if_values" not in st.session_state:
    st.session_state.what_if_values = {}

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

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.5;'>
        <b>SWARM SUBSYSTEMS:</b><br/>
        • Autonomous Profiler & Health Scorer<br/>
        • Leak-Free Preprocessor & Scaler<br/>
        • Stratified 5-Fold Model Arena<br/>
        • Isolation Forest Latent Forensics<br/>
        • Quantum What-If Simulator
    </div>
    """, unsafe_allow_html=True)

# Futuristic Top HUD Banner
st.markdown("""
<div class="hud-banner">
    <div style="display:flex; align-items:center; gap:10px;">
        <span class="live-dot"></span>
        <span style="font-weight:700; letter-spacing:0.05em; font-size:1.1rem; color:#f8fafc;">NEURAL DATA SCIENCE CORE</span>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span class="hud-badge badge-cyan">🧠 SWARM: ONLINE</span>
        <span class="hud-badge badge-purple">🔍 FORENSICS: ACTIVE</span>
        <span class="hud-badge badge-emerald">🔮 WHAT-IF: SYNCED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Ingestion Section
col_upload, col_target = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader("📥 Ingest Tabular Dataset (CSV)", type=["csv"], help="Upload any CSV file for autonomous analysis.")
    if uploaded_file is not None:
        try:
            st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
            st.session_state["dataset_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

df_current = st.session_state.get("uploaded_df")

if df_current is not None:
    with col_target:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        target_col = st.selectbox(
            "🎯 Target Variable to Predict:",
            options=df_current.columns,
            index=len(df_current.columns) - 1
        )
        launch_btn = st.button("🚀 INITIATE AUTONOMOUS INVESTIGATION", type="primary", use_container_width=True)

    # Dataset Telemetry Preview
    with st.expander(f"📊 Live Data Manifold Preview: {st.session_state.get('dataset_name', 'Dataset')} ({df_current.shape[0]} Rows × {df_current.shape[1]} Columns)", expanded=False):
        st.dataframe(df_current.head(8), use_container_width=True)

    # Run Autonomous Pipeline
    if launch_btn:
        progress_bar = st.progress(0)
        status_box = st.empty()

        def update_agent_progress(message: str, percentage: int):
            status_box.markdown(f"**⚡ AGENT DISPATCH:** `{message}`")
            progress_bar.progress(percentage)

        # Provider setup
        provider_code = "auto"
        if "Gemini" in provider:
            provider_code = "gemini"
        elif "OpenAI" in provider:
            provider_code = "openai"

        llm_client = AgentLLMClient(provider=provider_code, api_key=api_key)
        agent = DataScientistAgent(llm_client=llm_client)

        with st.spinner("Neural Swarm analyzing, cleaning, and benchmarking models..."):
            state = agent.run_full_pipeline(
                df=df_current,
                target_column=target_col,
                file_name=st.session_state.get("dataset_name", "dataset.csv"),
                progress_callback=update_agent_progress
            )
            st.session_state.state = state

        status_box.success("✨ Autonomous Investigation Successfully Completed!")
        progress_bar.empty()

    # Results Section
    if st.session_state.state is not None:
        state: DataScienceState = st.session_state.state

        tabs = st.tabs([
            "🌐 Neural Command Center",
            "🧬 2D Latent & Anomaly Forensics",
            "🔮 Quantum 'What-If' Simulator",
            "💬 Neural Data Chat",
            "📈 Deep Statistical Visuals",
            "📋 Cyber Executive Report",
            "🚀 Inference API"
        ])

        # -----------------------------------------------------------------
        # TAB 1: NEURAL COMMAND CENTER
        # -----------------------------------------------------------------
        with tabs[0]:
            st.markdown("### 🏆 Model Arena & Telemetry")
            
            # Metric HUD Badges
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            best_res = state.model_results[state.best_model_name]
            is_classif = "classification" in state.problem_type

            with m_col1:
                st.markdown(f"""
                <div class="cyber-card cyber-card-glow">
                    <span style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.05em;">WINNING MODEL</span>
                    <h3 style="color:#00f2fe; margin:5px 0 0 0; font-size:1.3rem;">{state.best_model_name}</h3>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                health = state.profiling_summary.get('health_score', 100)
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.05em;">DATA HEALTH SCORE</span>
                    <h3 style="color:#34d399; margin:5px 0 0 0; font-size:1.3rem;">{health}/100</h3>
                </div>
                """, unsafe_allow_html=True)

            with m_col3:
                primary_metric = "F1 Score" if is_classif else "R2 Score"
                val = best_res.metrics.get(primary_metric, 0)
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.05em;">BENCHMARK {primary_metric.upper()}</span>
                    <h3 style="color:#c084fc; margin:5px 0 0 0; font-size:1.3rem;">{val_str}</h3>
                </div>
                """, unsafe_allow_html=True)

            with m_col4:
                st.markdown(f"""
                <div class="cyber-card">
                    <span style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.05em;">OBJECTIVE</span>
                    <h3 style="color:#f8fafc; margin:5px 0 0 0; font-size:1.3rem;">{state.problem_type.replace('_', ' ').title()}</h3>
                </div>
                """, unsafe_allow_html=True)

            col_radar, col_table = st.columns([1, 1])

            # Radar Chart for Multi-Dimensional Model Comparison
            with col_radar:
                if is_classif:
                    categories = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
                    fig_radar = go.Figure()
                    colors = ["#00f2fe", "#a855f7", "#34d399"]

                    for idx, (m_name, res) in enumerate(state.model_results.items()):
                        r_vals = [res.metrics.get(cat, 0) for cat in categories]
                        r_vals.append(r_vals[0])  # Close the radar loop
                        c_list = categories + [categories[0]]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=r_vals,
                            theta=c_list,
                            fill='toself',
                            name=m_name,
                            line=dict(color=colors[idx % len(colors)], width=2),
                            opacity=0.6
                        ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                            bgcolor="rgba(15,23,42,0.6)"
                        ),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        title="Model Arena Multi-Metric Radar",
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Radar chart optimized for classification benchmarks.")

            # Model Benchmark Leaderboard Table
            with col_table:
                st.markdown("#### 📊 Model Arena Leaderboard")
                rows = []
                for name, res in state.model_results.items():
                    row = {"Model": name, "CV Score (Mean ± Std)": f"{res.cv_mean:.4f} ± {res.cv_std:.4f}"}
                    for m_name, val in res.metrics.items():
                        if m_name != "Confusion Matrix" and val is not None:
                            row[m_name] = round(val, 4) if isinstance(val, (int, float)) else val
                    rows.append(row)
                df_leaderboard = pd.DataFrame(rows).set_index("Model")
                st.dataframe(df_leaderboard, use_container_width=True)

            # Step-by-Step Agent Telemetry Traces
            st.markdown("---")
            st.markdown("#### 🕵️ Agent Swarm Audit Logs")
            for i, entry in enumerate(state.agent_logs):
                with st.expander(f"Step {i+1}: {entry.step_name} [{entry.timestamp}]", expanded=(i == 0)):
                    st.markdown("**🧠 Agent Reasoning / Hypothesis:**")
                    st.markdown(f'<div class="agent-thought-box">{entry.thought}</div>', unsafe_allow_html=True)
                    st.markdown("**⚡ Execution Action:**")
                    st.markdown(f'<div class="agent-action-box">{entry.action}</div>', unsafe_allow_html=True)
                    st.markdown("**📋 Observation / Output:**")
                    st.code(entry.output, language="text")

        # -----------------------------------------------------------------
        # TAB 2: 2D LATENT & ANOMALY FORENSICS
        # -----------------------------------------------------------------
        with tabs[1]:
            st.markdown("### 🧬 Latent Manifold Projection & Unsupervised Anomaly Forensics")
            st.caption("PCA dimensionality reduction to 2D space paired with Isolation Forest unsupervised outlier detection.")

            forensics = state.forensics_summary
            if forensics and "projection_df" in forensics:
                proj_df = forensics["projection_df"]

                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    st.metric("Total Observations Analyzed", f"{forensics['total_records']:,}")
                with f_col2:
                    st.metric("Anomalies Detected (Isolation Forest)", f"{forensics['anomaly_count']} records")
                with f_col3:
                    st.metric("Contamination Ratio", f"{forensics['anomaly_ratio']}%")

                fig_forensic = px.scatter(
                    proj_df,
                    x="PCA_Dim_1",
                    y="PCA_Dim_2",
                    color="Status",
                    color_discrete_map={
                        "Normal Cluster": "#00f2fe",
                        "Anomaly (Outlier)": "#f43f5e"
                    },
                    hover_data=["Anomaly_Score"],
                    title=f"2D Latent Manifold Projection (Explained Variance: {forensics['explained_variance'][0]}% & {forensics['explained_variance'][1]}%)",
                    template="plotly_dark"
                )
                fig_forensic.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)"
                )
                st.plotly_chart(fig_forensic, use_container_width=True)
            else:
                st.info("Forensic manifold projection requires at least two numeric features.")

        # -----------------------------------------------------------------
        # TAB 3: QUANTUM "WHAT-IF" SIMULATOR
        # -----------------------------------------------------------------
        with tabs[2]:
            st.markdown("### 🔮 Quantum Counterfactual 'What-If' Simulation Engine")
            st.caption("Perturb input feature values in real time and watch the model's prediction probability shift instantly.")

            sim_cols = [c for c in state.cleaned_df.columns if c != state.target_column]
            
            # Interactive feature controls
            st.markdown("##### 🎛️ Feature Value Controllers:")
            input_payload = {}
            grid_cols = st.columns(3)

            for i, col_name in enumerate(sim_cols[:9]):
                with grid_cols[i % 3]:
                    series = state.cleaned_df[col_name]
                    if pd.api.types.is_numeric_dtype(series):
                        min_v = float(series.min())
                        max_v = float(series.max())
                        median_v = float(series.median())
                        input_payload[col_name] = st.slider(
                            f"{col_name}",
                            min_value=min_v,
                            max_value=max_v,
                            value=median_v,
                            key=f"sim_{col_name}"
                        )
                    else:
                        unique_vals = list(series.dropna().unique())
                        input_payload[col_name] = st.selectbox(
                            f"{col_name}",
                            options=unique_vals,
                            index=0,
                            key=f"sim_{col_name}"
                        )

            # Fill in unselected columns with defaults
            for c in sim_cols[9:]:
                if pd.api.types.is_numeric_dtype(state.cleaned_df[c]):
                    input_payload[c] = float(state.cleaned_df[c].median())
                else:
                    input_payload[c] = str(state.cleaned_df[c].mode()[0])

            # Run Simulation
            sim_result = WhatIfSimulator.simulate_record(
                payload=input_payload,
                model=state.best_pipeline,
                preprocessor=state.preprocessor,
                label_encoder=state.label_encoder,
                problem_type=state.problem_type
            )

            st.markdown("---")
            sim_out_col1, sim_out_col2 = st.columns([1, 1])

            with sim_out_col1:
                st.markdown("#### 🎯 Live Model Prediction:")
                pred_label = sim_result["prediction"]
                conf_val = sim_result["confidence"]

                st.markdown(f"""
                <div class="cyber-card cyber-card-glow" style="text-align:center;">
                    <span style="font-size:0.85rem; color:#94a3b8;">PREDICTED OUTCOME</span>
                    <h1 style="color:#00f2fe; margin:10px 0; font-size:2.4rem;">{pred_label}</h1>
                    <span style="color:#34d399; font-weight:600; font-size:1.1rem;">Confidence: {conf_val}%</span>
                </div>
                """, unsafe_allow_html=True)

            with sim_out_col2:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf_val,
                    title={'text': "Prediction Confidence Meter (%)", 'font': {'color': '#f8fafc'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#00f2fe"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(244, 63, 94, 0.2)"},
                            {'range': [50, 75], 'color': "rgba(168, 85, 247, 0.2)"},
                            {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#f8fafc"},
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=240
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        # -----------------------------------------------------------------
        # TAB 4: NEURAL DATA CHAT
        # -----------------------------------------------------------------
        with tabs[3]:
            st.markdown("### 💬 Neural Conversational Analyst")
            st.caption("Ask natural language questions about patterns, correlations, and anomalies in your data.")

            analyst = NeuralChatAnalyst(llm_client=AgentLLMClient(provider=provider_code, api_key=api_key))

            # Sample query buttons
            st.markdown("**💡 Quick Prompt Injections:**")
            q_cols = st.columns(3)
            with q_cols[0]:
                if st.button("🔍 Explain why the winning model won"):
                    st.session_state.chat_messages.append({"role": "user", "content": "Explain why the winning model won and summarize its advantages."})
                    resp = analyst.answer_query("Explain why the winning model won and summarize its advantages.", state)
                    st.session_state.chat_messages.append({"role": "assistant", "content": resp})
            with q_cols[1]:
                if st.button("📊 Top 3 factors driving predictions"):
                    st.session_state.chat_messages.append({"role": "user", "content": "What are the top 3 factors driving the predictions and why?"})
                    resp = analyst.answer_query("What are the top 3 factors driving the predictions and why?", state)
                    st.session_state.chat_messages.append({"role": "assistant", "content": resp})
            with q_cols[2]:
                if st.button("⚠️ Production risks & data drift"):
                    st.session_state.chat_messages.append({"role": "user", "content": "What are the biggest data quality and drift risks for this dataset?"})
                    resp = analyst.answer_query("What are the biggest data quality and drift risks for this dataset?", state)
                    st.session_state.chat_messages.append({"role": "assistant", "content": resp})

            # Chat history
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

            user_query = st.chat_input("Ask any data science or dataset question...")
            if user_query:
                st.chat_message("user").write(user_query)
                st.session_state.chat_messages.append({"role": "user", "content": user_query})
                with st.spinner("Neural Swarm deliberating..."):
                    reply = analyst.answer_query(user_query, state)
                st.chat_message("assistant").write(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

        # -----------------------------------------------------------------
        # TAB 5: DEEP STATISTICAL VISUALS
        # -----------------------------------------------------------------
        with tabs[4]:
            st.markdown("### 📈 Deep Statistical Visualizations")
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                # Correlation Heatmap
                corr_dict = state.profiling_summary.get("corr_matrix", {})
                if corr_dict:
                    corr_df = pd.DataFrame(corr_dict)
                    fig_corr = px.imshow(
                        corr_df,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Numerical Feature Correlation Heatmap",
                        template="plotly_dark"
                    )
                    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Insufficient numerical columns for correlation matrix.")

            with v_col2:
                # Top Feature Importance
                best_res = state.model_results[state.best_model_name]
                if best_res.feature_importances:
                    feat_s = pd.Series(best_res.feature_importances).sort_values(ascending=True)
                    fig_imp = px.bar(
                        x=feat_s.values,
                        y=feat_s.index,
                        orientation='h',
                        title=f"Top Influential Features ({state.best_model_name})",
                        labels={"x": "Importance / Relative Weight", "y": "Feature"},
                        template="plotly_dark",
                        color_discrete_sequence=["#00f2fe"]
                    )
                    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_imp, use_container_width=True)

            if is_classif:
                cm = best_res.metrics.get("Confusion Matrix")
                if cm:
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        title=f"Holdout Confusion Matrix ({state.best_model_name})",
                        labels=dict(x="Predicted Class", y="Actual Class"),
                        color_continuous_scale="Purples",
                        template="plotly_dark"
                    )
                    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cm, use_container_width=True)

        # -----------------------------------------------------------------
        # TAB 6: CYBER EXECUTIVE REPORT
        # -----------------------------------------------------------------
        with tabs[5]:
            st.markdown("### 📋 Executive Investigation Briefing")
            if state.executive_report:
                st.markdown(state.executive_report)
                st.download_button(
                    label="💾 Download Cyber Executive Report (.md)",
                    data=state.executive_report,
                    file_name="executive_data_science_report.md",
                    mime="text/markdown"
                )

        # -----------------------------------------------------------------
        # TAB 7: INFERENCE API & ASSETS
        # -----------------------------------------------------------------
        with tabs[6]:
            st.markdown("### 🚀 Production Inference API & Model Bundle")
            st.markdown("Exportable assets for production microservices and Docker containers.")

            st.markdown("#### 📦 Standalone Scoring Script")
            st.code(state.inference_code, language="python")

            joblib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models", "best_pipeline.joblib")
            if os.path.exists(joblib_path):
                with open(joblib_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Serialized Pipeline Bundle (.joblib)",
                        data=f.read(),
                        file_name="best_pipeline.joblib",
                        mime="application/octet-stream"
                    )
else:
    st.markdown("""
    <div class="cyber-card" style="text-align:center; padding: 40px 20px;">
        <span style="font-size:2.5rem;">🌌</span>
        <h3 style="color:#00f2fe; margin-top:10px;">Awaiting Dataset Ingestion</h3>
        <p style="color:#94a3b8; max-width:550px; margin: 10px auto;">
            Upload any CSV dataset above, or click <b>"📥 Load Telecom Churn Dataset"</b> in the left sidebar to launch the Autonomous Neural Swarm.
        </p>
    </div>
    """, unsafe_allow_html=True)
