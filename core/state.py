"""
State management module for Agentic AI Data Scientist.
Maintains state across all stages of data ingestion, profiling,
cleaning, modeling, evaluation, and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass
class AgentLogEntry:
    step_name: str
    thought: str
    action: str
    output: str
    timestamp: str


@dataclass
class ModelResult:
    model_name: str
    model_obj: Any
    metrics: Dict[str, float]
    feature_importances: Optional[Dict[str, float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


@dataclass
class DataScienceState:
    raw_df: Optional[pd.DataFrame] = None
    cleaned_df: Optional[pd.DataFrame] = None
    file_name: Optional[str] = None
    target_column: Optional[str] = None
    problem_type: Optional[str] = None  # 'binary_classification', 'multiclass_classification', 'regression'
    
    # Feature classification
    numerical_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    dropped_cols: List[str] = field(default_factory=list)
    
    # Preprocessor pipeline
    preprocessor: Optional[Any] = None
    label_encoder: Optional[Any] = None
    
    # Processed splits
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    
    # Modeling & Evaluation
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    best_pipeline: Optional[Any] = None
    
    # EDA & Forensics cache
    profiling_summary: Dict[str, Any] = field(default_factory=dict)
    forensics_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Generated outputs
    executive_report: Optional[str] = None
    inference_code: Optional[str] = None
    
    # Audit log
    agent_logs: List[AgentLogEntry] = field(default_factory=list)

    def log_step(self, step_name: str, thought: str, action: str, output: str):
        from datetime import datetime
        entry = AgentLogEntry(
            step_name=step_name,
            thought=thought,
            action=action,
            output=output,
            timestamp=datetime.now().strftime("%H:%M:%S")
        )
        self.agent_logs.append(entry)

    def __getattr__(self, name: str):
        """Fallback for any attribute added across session reloads."""
        return None
