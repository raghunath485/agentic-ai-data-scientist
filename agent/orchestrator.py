"""
Orchestrator Agent for the Autonomous Data Scientist.
Coordinates profiling, cleaning, modeling, evaluation, and reporting
while logging thoughts and actions to the shared state.
"""

from typing import Dict, Any, Callable, Optional
import pandas as pd
from core.state import DataScienceState
from engine.profiler import DataProfiler
from engine.cleaner import DataCleaner
from engine.modeler import ModelTrainer
from engine.reporter import ReportGenerator
from agent.llm_client import AgentLLMClient


class DataScientistAgent:
    def __init__(self, llm_client: Optional[AgentLLMClient] = None):
        self.llm = llm_client or AgentLLMClient()

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        file_name: str = "dataset.csv",
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> DataScienceState:
        """
        Executes the autonomous end-to-end data science lifecycle.
        """
        state = DataScienceState(raw_df=df, file_name=file_name, target_column=target_column)

        # -----------------------------------------------------------------
        # STEP 1: Ingestion & Problem Discovery
        # -----------------------------------------------------------------
        if progress_callback:
            progress_callback("Step 1/5: Profiling Dataset & Health Check...", 20)

        thought_prompt = f"Dataset uploaded with {df.shape[0]} rows and {df.shape[1]} columns. Target column is '{target_column}'."
        agent_thought = self.llm.query(
            f"Analyze this context: {thought_prompt}. What should the data profiling strategy be?",
            system_prompt="You are an expert Chief AI Data Scientist planning automated EDA."
        )
        
        profiling_res = DataProfiler.profile_dataset(df, target_column=target_column)
        state.profiling_summary = profiling_res
        state.problem_type = profiling_res.get("problem_type")

        state.log_step(
            step_name="Exploratory Data Analysis & Health Check",
            thought=agent_thought,
            action=f"Profiled {df.shape[1]} columns. Calculated Health Score of {profiling_res['health_score']}/100. Inferred objective: {state.problem_type}.",
            output=f"Target: '{target_column}' | Categorical features: {len(profiling_res['categorical_cols'])} | Numerical features: {len(profiling_res['numerical_cols'])} | Duplicates: {profiling_res['duplicate_rows']}"
        )

        # -----------------------------------------------------------------
        # STEP 2: Leakage-Free Preprocessing & Cleaning
        # -----------------------------------------------------------------
        if progress_callback:
            progress_callback("Step 2/5: Building Leakage-Free Preprocessing Pipeline...", 40)

        agent_thought = self.llm.query(
            f"Data has {profiling_res['health_score']}/100 health score. Target is {target_column}. Missing values present. Plan preprocessing.",
            system_prompt="You are an expert AI Data Scientist planning feature engineering and cleaning."
        )

        clean_res = DataCleaner.prepare_and_clean_data(df, target_column=target_column)
        state.cleaned_df = clean_res["cleaned_df"]
        state.X_train = clean_res["X_train"]
        state.X_test = clean_res["X_test"]
        state.y_train = clean_res["y_train"]
        state.y_test = clean_res["y_test"]
        state.preprocessor = clean_res["preprocessor"]
        state.numerical_cols = clean_res["numerical_features"]
        state.categorical_cols = clean_res["categorical_features"]
        state.dropped_cols = clean_res["dropped_features"]

        state.log_step(
            step_name="Data Cleaning & Feature Engineering",
            thought=agent_thought,
            action="Applied ColumnTransformer (Median/Mode imputation + StandardScaler + OneHotEncoder) strictly on train split.",
            output="\n".join([f"• {act}" for act in clean_res["cleaning_actions"]])
        )

        # -----------------------------------------------------------------
        # STEP 3: Model Benchmarking & Cross-Validation
        # -----------------------------------------------------------------
        if progress_callback:
            progress_callback("Step 3/5: Benchmarking Machine Learning Algorithms with 5-Fold CV...", 65)

        agent_thought = self.llm.query(
            f"Problem type is {state.problem_type}. Benchmarking Linear baseline, Random Forest, and Gradient Boosted trees. Explain metric criteria.",
            system_prompt="You are an expert AI Data Scientist orchestrating model validation."
        )

        train_res = ModelTrainer.benchmark_models(
            X_train=state.X_train,
            y_train=state.y_train,
            X_test=state.X_test,
            y_test=state.y_test,
            problem_type=state.problem_type,
            cv_folds=5
        )
        state.model_results = train_res["model_results"]
        state.best_model_name = train_res["best_model_name"]
        state.best_pipeline = train_res["best_model_obj"]

        best_m = state.model_results[state.best_model_name].metrics
        metric_summary = ", ".join([f"{k}: {v}" for k, v in best_m.items() if k != "Confusion Matrix" and v is not None])

        state.log_step(
            step_name="Model Benchmarking & Validation",
            thought=agent_thought,
            action=f"Benchmarked 3 candidate algorithms using 5-Fold Stratified Cross-Validation. Selected {state.best_model_name}.",
            output=f"Winning Model: {state.best_model_name} | Metrics: {metric_summary}"
        )

        # -----------------------------------------------------------------
        # STEP 4: Export Pipeline & Inference Code
        # -----------------------------------------------------------------
        if progress_callback:
            progress_callback("Step 4/5: Serializing Model Bundle & Generating Inference Code...", 85)

        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, "outputs", "models")

        model_path = ModelTrainer.export_best_pipeline(
            best_model=state.best_pipeline,
            preprocessor=state.preprocessor,
            label_encoder=clean_res["label_encoder"],
            output_dir=output_dir
        )

        inference_code = ModelTrainer.generate_inference_snippet(
            model_name=state.best_model_name,
            problem_type=state.problem_type,
            feature_names=list(state.X_train.columns)
        )
        state.inference_code = inference_code

        state.log_step(
            step_name="Model Serialization & Deployment Assets",
            thought="Production models must be easily deployable with reproducible pipelines.",
            action=f"Serialized complete inference pipeline bundle to '{model_path}'.",
            output="Generated ready-to-run Python scoring code and saved pipeline artifact."
        )

        # -----------------------------------------------------------------
        # STEP 5: Executive Report Synthesis
        # -----------------------------------------------------------------
        if progress_callback:
            progress_callback("Step 5/5: Synthesizing Executive Data Science Report...", 95)

        report = ReportGenerator.generate_markdown_report(state)
        state.executive_report = report

        state.log_step(
            step_name="Synthesis & Executive Reporting",
            thought="Summarizing all statistical findings, model metrics, and production guidance into an executive briefing.",
            action="Compiled comprehensive markdown report.",
            output=f"Generated executive report with data health score {profiling_res['health_score']}/100 and benchmark summary."
        )

        if progress_callback:
            progress_callback("Autonomous Investigation Complete!", 100)

        return state
