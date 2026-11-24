"""
Neural Data Chat Agent.
Answers natural language questions about the active dataset, statistical distributions,
and model behavior using contextual dataset summaries and LLM reasoning.
"""

from typing import Dict, Any, Optional
import pandas as pd
from agent.llm_client import AgentLLMClient
from core.state import DataScienceState


class NeuralChatAnalyst:
    def __init__(self, llm_client: Optional[AgentLLMClient] = None):
        self.llm = llm_client or AgentLLMClient()

    def answer_query(self, query: str, state: DataScienceState) -> str:
        """Generates an intelligent context-aware answer for natural language queries."""
        df = state.raw_df
        if df is None:
            return "No dataset has been loaded yet. Please ingest a dataset first."

        # Build context summary
        summary_context = f"""
Dataset Name: {state.file_name or 'Uploaded Dataset'}
Total Rows: {len(df)}
Total Columns: {len(df.columns)}
Target Column: {state.target_column}
Objective: {state.problem_type}
Winning Model: {state.best_model_name}
Data Health Score: {state.profiling_summary.get('health_score', 'N/A')}/100
Numerical Columns: {', '.join(state.numerical_cols[:8])}
Categorical Columns: {', '.join(state.categorical_cols[:8])}
"""
        best_res = state.model_results.get(state.best_model_name)
        if best_res and best_res.feature_importances:
            top_feats = ", ".join([f"{k} ({v})" for k, v in list(best_res.feature_importances.items())[:5]])
            summary_context += f"\nTop Influential Features: {top_feats}"

        prompt = f"""You are the Neural AI Data Scientist Swarm assistant.
Context of the analyzed dataset:
{summary_context}

User question:
"{query}"

Please provide a sharp, executive-level data science answer with actionable observations and specific numbers from the context when available.
"""
        return self.llm.query(
            prompt,
            system_prompt="You are a futuristic Neural AI Data Scientist. Provide precise, analytical, and highly structured insights."
        )
