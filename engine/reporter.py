"""
Engine module for generating comprehensive Executive & Technical Data Science Reports.
Synthesizes data health, EDA observations, preprocessing steps, and model benchmark comparisons.
"""

from typing import Dict, Any
from core.state import DataScienceState


class ReportGenerator:
    @staticmethod
    def generate_markdown_report(state: DataScienceState) -> str:
        """Constructs a professional Data Science report in Markdown."""
        profile = state.profiling_summary
        best_name = state.best_model_name
        best_result = state.model_results.get(best_name) if best_name else None

        # Build model comparison markdown table
        metrics_headers = []
        is_classification = state.problem_type in ["binary_classification", "multiclass_classification"]
        
        if is_classification:
            table_header = "| Model Name | CV F1 (Mean ± Std) | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC |\n|---|---|---|---|---|---|---|---|\n"
            rows = []
            for name, res in state.model_results.items():
                m = res.metrics
                auc_str = f"{m.get('ROC-AUC'):.4f}" if m.get('ROC-AUC') is not None else "N/A"
                is_winner = " 🏆" if name == best_name else ""
                row = f"| **{name}{is_winner}** | {res.cv_mean:.4f} ± {res.cv_std:.4f} | {m.get('Accuracy', 0):.4f} | {m.get('Balanced Accuracy', 0):.4f} | {m.get('Precision', 0):.4f} | {m.get('Recall', 0):.4f} | **{m.get('F1 Score', 0):.4f}** | {auc_str} |"
                rows.append(row)
            model_table = table_header + "\n".join(rows)
        else:
            table_header = "| Model Name | CV R² (Mean ± Std) | RMSE | MAE | R² Score |\n|---|---|---|---|---|\n"
            rows = []
            for name, res in state.model_results.items():
                m = res.metrics
                is_winner = " 🏆" if name == best_name else ""
                row = f"| **{name}{is_winner}** | {res.cv_mean:.4f} ± {res.cv_std:.4f} | {m.get('RMSE', 0):.4f} | {m.get('MAE', 0):.4f} | **{m.get('R2 Score', 0):.4f}** |"
                rows.append(row)
            model_table = table_header + "\n".join(rows)

        # Build feature importance list
        feat_imp_text = ""
        if best_result and best_result.feature_importances:
            feat_imp_text = "\n".join([f"- **`{feat}`**: {imp:.4f}" for feat, imp in list(best_result.feature_importances.items())[:10]])
        else:
            feat_imp_text = "*No tree feature importances computed for this model.*"

        # Report content
        report = f"""# 📊 Autonomous Data Science Investigation Report

**Dataset:** `{state.file_name or 'Uploaded Dataset'}`  
**Target Variable:** `{state.target_column}`  
**Problem Objective:** `{state.problem_type.replace('_', ' ').title()}`  
**Data Health Score:** `{profile.get('health_score', 'N/A')}/100`

---

## 1. Executive Summary
An autonomous agentic data science investigation was conducted on the dataset. The primary objective was to model and predict `{state.target_column}` using statistical profiling, leak-free preprocessing, and model benchmarking across multiple algorithms.

- **Recommended Production Model:** **{best_name}**
- **Validation Method:** 5-Fold Stratified / K-Fold Cross-Validation followed by out-of-sample holdout test evaluation.
- **Top Metric Achieved:** {f"F1 Score: **{best_result.metrics.get('F1 Score', 'N/A')}** (Accuracy: {best_result.metrics.get('Accuracy', 'N/A')})" if is_classification else f"R² Score: **{best_result.metrics.get('R2 Score', 'N/A')}** (RMSE: {best_result.metrics.get('RMSE', 'N/A')})"}

---

## 2. Dataset Diagnostics & Health
- **Total Observations:** `{profile.get('total_rows', 0):,}`
- **Total Features:** `{profile.get('total_cols', 0):,}`
- **Duplicate Rows:** `{profile.get('duplicate_rows', 0)} ({profile.get('duplicate_pct', 0)}%)`
- **Numerical Features Analyzed:** `{len(profile.get('numerical_cols', []))}`
- **Categorical Features Analyzed:** `{len(profile.get('categorical_cols', []))}`
- **Excluded Features (High Cardinality / Constant):** `{len(profile.get('constant_cols', [])) + len(profile.get('high_cardinality_cols', []))}`

---

## 3. Preprocessing & Leakage Prevention Strategy
To ensure real-world production robustness and avoid optimistic validation bias:
1. **Partitioning:** Data was split into train and test sets *before* fitting transformations.
2. **Missing Value Imputation:** Numerical features imputed using median strategies; categorical features imputed using mode strategies.
3. **Scaling & Encoding:** Numerical columns normalized via `StandardScaler`; categoricals encoded via `OneHotEncoder(handle_unknown='ignore')`.
4. **Leakage Safeguards:** All estimators and transformers were encapsulated in an end-to-end `ColumnTransformer` pipeline.

---

## 4. Model Benchmark Comparison
{model_table}

---

## 5. Model Explainability: Top Influential Features
The top drivers impacting predictions for **{best_name}** are:

{feat_imp_text}

---

## 6. Production Deployment & Operational Guidance
1. **Model Serialization:** The winning model pipeline has been saved as a self-contained `.joblib` bundle containing preprocessing and trained weights.
2. **Inference Guardrails:** Ensure inference payloads match the original categorical categories and numerical boundaries.
3. **Drift Monitoring:** Monitor feature distributions continuously in production to detect covariate shift.
"""
        return report
