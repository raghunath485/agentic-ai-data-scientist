"""
Agentic AI Data Scientist - Next Generation Streamlit Application.
Provides autonomous EDA, leak-free preprocessing, multi-model benchmarking,
model explainability, executive reporting, and live inference.
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
from agent.llm_client import AgentLLMClient

# Page configuration
st.set_page_config(
    page_title="Agentic AI Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .agent-thought {
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
        padding: 10px 15px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "state" not in st.session_state:
    st.session_state.state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration
with st.sidebar:
    st.title("🤖 Agent Configuration")
    
    st.markdown("### 🧠 AI Engine")
    provider = st.selectbox(
        "LLM Provider",
        ["Built-in Heuristic Specialist", "Google Gemini", "OpenAI"],
        index=0,
        help="The built-in specialist runs completely offline without any API keys. Select Gemini or OpenAI to enable full LLM dynamic reasoning."
    )

    api_key = None
    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    st.markdown("---")
    st.markdown("### 📁 Quick Demo Dataset")
    if st.button("📥 Load Telecom Churn Dataset"):
        demo_path = os.path.join(os.path.dirname(__file__), "data", "sample_churn.csv")
        if os.path.exists(demo_path):
            st.session_state["uploaded_df"] = pd.read_csv(demo_path)
            st.session_state["dataset_name"] = "sample_churn.csv"
            st.success("Loaded Telecom Churn Demo Dataset!")

    st.markdown("---")
    st.caption("Agentic AI Data Scientist v2.0 | Real-World Standard")

# Header
st.title("🤖 Agentic AI Data Scientist")
st.markdown("Autonomous End-to-End Machine Learning, Statistical Profiling, and Explainable Insights.")

# Dataset Ingestion Section
col_upload, col_settings = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
            st.session_state["dataset_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

df_current = st.session_state.get("uploaded_df")

if df_current is not None:
    with col_settings:
        target_col = st.selectbox(
            "Select Target Column",
            options=df_current.columns,
            index=len(df_current.columns) - 1
        )
        run_btn = st.button("🚀 Run Autonomous Agent", type="primary", use_container_width=True)

    # Preview Dataset
    with st.expander("📄 Dataset Preview & Quick Statistics", expanded=False):
        st.dataframe(df_current.head(10), use_container_width=True)
        st.write(f"**Dimensions:** {df_current.shape[0]} rows × {df_current.shape[1]} columns")

    # Run Agent Pipeline
    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, val: int):
            status_text.info(msg)
            progress_bar.progress(val)

        # Initialize LLM client
        provider_code = "auto"
        if provider == "Google Gemini":
            provider_code = "gemini"
        elif provider == "OpenAI":
            provider_code = "openai"

        llm_client = AgentLLMClient(provider=provider_code, api_key=api_key)
        agent = DataScientistAgent(llm_client=llm_client)

        with st.spinner("Autonomous Agent is executing data science lifecycle..."):
            state = agent.run_full_pipeline(
                df=df_current,
                target_column=target_col,
                file_name=st.session_state.get("dataset_name", "dataset.csv"),
                progress_callback=update_progress
            )
            st.session_state.state = state

        status_text.success("Autonomous Data Science Pipeline Completed Successfully!")
        progress_bar.empty()

    # If state is available, display results tabs
    if st.session_state.state is not None:
        state: DataScienceState = st.session_state.state

        tabs = st.tabs([
            "📊 Dashboard & Model Benchmark",
            "🔍 Agent Execution Traces",
            "📈 Interactive Visualizations",
            "📋 Executive Report",
            "🔮 Live Inference & Export"
        ])

        # TAB 1: Dashboard & Benchmark
        with tabs[0]:
            st.subheader("🏆 Model Benchmarking & Performance")
            
            # KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            best_res = state.model_results[state.best_model_name]
            
            with kpi1:
                st.metric("Winning Model", state.best_model_name)
            with kpi2:
                health = state.profiling_summary.get("health_score", "N/A")
                st.metric("Data Health Score", f"{health}/100")
            with kpi3:
                primary_metric = "F1 Score" if "classification" in state.problem_type else "R2 Score"
                val = best_res.metrics.get(primary_metric, "N/A")
                st.metric(f"Top {primary_metric}", f"{val:.4f}" if isinstance(val, (int, float)) else val)
            with kpi4:
                st.metric("Problem Type", state.problem_type.replace('_', ' ').title())

            st.markdown("---")

            # Benchmark Comparison Table
            st.markdown("### 📊 Benchmark Comparison")
            rows = []
            for name, res in state.model_results.items():
                row_data = {"Model": name, "CV Score (Mean ± Std)": f"{res.cv_mean:.4f} ± {res.cv_std:.4f}"}
                for metric_name, val in res.metrics.items():
                    if metric_name != "Confusion Matrix" and val is not None:
                        row_data[metric_name] = round(val, 4) if isinstance(val, (int, float)) else val
                rows.append(row_data)

            res_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(res_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        # TAB 2: Agent Execution Traces
        with tabs[1]:
            st.subheader("🕵️ Agent Step-by-Step Reasoning & Audit Logs")
            for i, entry in enumerate(state.agent_logs):
                with st.expander(f"Step {i+1}: {entry.step_name} [{entry.timestamp}]", expanded=(i == 0)):
                    st.markdown("**🧠 Agent Reasoning / Hypothesis:**")
                    st.info(entry.thought)
                    st.markdown("**⚡ Action Taken:**")
                    st.code(entry.action, language="text")
                    st.markdown("**📋 Execution Output:**")
                    st.code(entry.output, language="text")

        # TAB 3: Interactive Visualizations
        with tabs[2]:
            st.subheader("📈 Interactive Statistical Visualizations")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                # Correlation Heatmap
                corr_dict = state.profiling_summary.get("corr_matrix", {})
                if corr_dict:
                    corr_df = pd.DataFrame(corr_dict)
                    fig_corr = px.imshow(
                        corr_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Numerical Feature Correlation Heatmap"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Insufficient numerical columns for correlation matrix.")

            with col_v2:
                # Top Feature Importance
                best_res = state.model_results[state.best_model_name]
                if best_res.feature_importances:
                    feat_s = pd.Series(best_res.feature_importances).sort_values(ascending=True)
                    fig_imp = px.bar(
                        x=feat_s.values,
                        y=feat_s.index,
                        orientation='h',
                        title=f"Top Influential Features ({state.best_model_name})",
                        labels={"x": "Importance / Relative Weight", "y": "Feature"}
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance not applicable for current model type.")

            # Confusion Matrix for Classification
            if "classification" in state.problem_type:
                cm = best_res.metrics.get("Confusion Matrix")
                if cm:
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        title=f"Holdout Confusion Matrix ({state.best_model_name})",
                        labels=dict(x="Predicted Class", y="Actual Class"),
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

        # TAB 4: Executive Report
        with tabs[3]:
            st.subheader("📋 Executive Data Science Report")
            if state.executive_report:
                st.markdown(state.executive_report)
                st.download_button(
                    label="💾 Download Full Report (Markdown)",
                    data=state.executive_report,
                    file_name="executive_data_science_report.md",
                    mime="text/markdown"
                )

        # TAB 5: Live Inference & Export
        with tabs[4]:
            st.subheader("🔮 Real-Time Model Inference & Production Assets")
            st.markdown("Test the best trained model on single-record payloads.")

            # Dynamic Input Form
            inference_cols = list(state.X_train.columns)[:10]
            with st.form("inference_form"):
                st.markdown("##### Input Feature Values:")
                input_data = {}
                cols = st.columns(3)
                for i, col_name in enumerate(inference_cols):
                    with cols[i % 3]:
                        mean_val = float(state.X_train[col_name].mean()) if col_name in state.X_train else 0.0
                        input_data[col_name] = st.number_input(col_name, value=round(mean_val, 2))
                
                predict_submitted = st.form_submit_button("Predict with Best Model", type="primary")

            if predict_submitted:
                try:
                    payload_df = pd.DataFrame([input_data])
                    raw_pred = state.best_pipeline.predict(payload_df)[0]
                    st.success(f"**Model Prediction:** `{raw_pred}`")
                except Exception as e:
                    st.warning(f"Inference preview: {e}")

            st.markdown("---")
            st.markdown("### 📦 Standalone Python Inference Code")
            st.code(state.inference_code, language="python")

            joblib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models", "best_pipeline.joblib")
            if os.path.exists(joblib_path):
                with open(joblib_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Trained Pipeline Bundle (.joblib)",
                        data=f.read(),
                        file_name="best_pipeline.joblib",
                        mime="application/octet-stream"
                    )
else:
    st.info("👆 Upload a CSV dataset or click 'Load Telecom Churn Dataset' in the sidebar to begin.")
