"""
Integration test for Agentic AI Data Scientist pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from agent.orchestrator import DataScientistAgent
from agent.llm_client import AgentLLMClient


def test_end_to_end_pipeline():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_churn.csv")
    df = pd.read_csv(csv_path)

    agent = DataScientistAgent(llm_client=AgentLLMClient(provider="heuristic"))
    state = agent.run_full_pipeline(
        df=df,
        target_column="Churn",
        file_name="sample_churn.csv"
    )

    assert state.problem_type == "binary_classification"
    assert state.best_model_name is not None
    assert len(state.model_results) >= 3
    assert state.executive_report is not None
    assert len(state.agent_logs) == 5

    print("[SUCCESS] All pipeline assertions passed successfully!")
    print(f"[WINNER] Winning Model: {state.best_model_name}")
    print(f"[HEALTH] Health Score: {state.profiling_summary['health_score']}/100")


if __name__ == "__main__":
    test_end_to_end_pipeline()
