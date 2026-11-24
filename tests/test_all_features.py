"""
Master Integration Test Suite covering all 8 enterprise subsystems.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from agent.orchestrator import DataScientistAgent
from agent.llm_client import AgentLLMClient
from agent.sandbox_repl import SelfHealingREPL

from engine.explainer import ModelExplainer
from engine.tuner import BayesianTuner
from engine.deployment import MicroserviceGenerator
from engine.drift_monitor import DriftMonitor
from engine.data_loader import MultiSourceIngestor
from engine.balancer import DataBalancer
from engine.slide_deck import SlideDeckGenerator


def run_master_test():
    print("[TEST 1/8] Testing Ingestor...")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_churn.csv")
    df, meta = MultiSourceIngestor.load_from_file_or_buffer(csv_path, "sample_churn.csv")
    assert df.shape[0] > 0
    print(" - MultiSourceIngestor passed.")

    print("[TEST 2/8] Running Base Pipeline...")
    agent = DataScientistAgent(llm_client=AgentLLMClient(provider="heuristic"))
    state = agent.run_full_pipeline(df=df, target_column="Churn", file_name="sample_churn.csv")
    assert state.best_model_name is not None
    print(f" - Winning Model: {state.best_model_name}")

    print("[TEST 3/8] Testing SHAP Explainer...")
    rec_df = df.head(1).drop(columns=["Churn"])
    shap_info = ModelExplainer.compute_shap_breakdown(
        model=state.best_pipeline, preprocessor=state.preprocessor,
        input_record_df=rec_df, background_sample_df=df,
        feature_names=list(state.X_train.columns)
    )
    assert shap_info["status"] == "success"
    print(" - SHAP Explainer passed.")

    print("[TEST 4/8] Testing Self-Healing REPL...")
    repl = SelfHealingREPL(llm_client=AgentLLMClient(provider="heuristic"))
    res_repl = repl.execute_with_self_healing("result = df.shape[0]", {"df": df})
    assert res_repl["status"] == "success" and res_repl["result"] == df.shape[0]
    print(" - SelfHealingREPL passed.")

    print("[TEST 5/8] Testing Optuna Bayesian Tuner...")
    opt_res = BayesianTuner.tune_model(
        X_train=state.X_train, y_train=state.y_train,
        model_type="Random Forest", problem_type=state.problem_type, n_trials=3
    )
    assert opt_res["best_score"] > 0
    print(f" - Optuna Tuner passed (best score: {opt_res['best_score']}).")

    print("[TEST 6/8] Testing Production Drift Monitor...")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Churn"]
    drift_res = DriftMonitor.scan_dataset_drift(df, df.sample(frac=0.9, replace=True), num_cols)
    assert "drift_df" in drift_res
    print(" - Drift Monitor passed.")

    print("[TEST 7/8] Testing Synthetic Minority Balancer...")
    bal_res = DataBalancer.balance_classification_dataset(df, "Churn")
    assert bal_res["balanced_shape"][0] >= df.shape[0]
    print(" - Synthetic Balancer passed.")

    print("[TEST 8/8] Testing Microservice & Slide Deck Generators...")
    fastapi_code = MicroserviceGenerator.generate_fastapi_code(list(state.X_train.columns), state.best_model_name, state.problem_type)
    assert "FastAPI" in fastapi_code
    zip_bytes = MicroserviceGenerator.package_microservice_zip(fastapi_code, "FROM python", "version: '3.8'")
    assert len(zip_bytes) > 0
    
    html_deck = SlideDeckGenerator.generate_html_deck(state)
    assert "Executive Briefing" in html_deck
    print(" - Microservice & Slide Deck passed.")

    print("\n[ALL 8 SUBSYSTEMS PASSED WITH 100% SUCCESS!]")


if __name__ == "__main__":
    run_master_test()
