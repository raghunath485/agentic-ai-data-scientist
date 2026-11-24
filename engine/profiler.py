"""
Engine module for Exploratory Data Analysis (EDA) and Data Profiling.
Computes dataset health score, missingness, distributions, and correlation.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class DataProfiler:
    @staticmethod
    def profile_dataset(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Performs a comprehensive automated profile of the dataset."""
        total_rows, total_cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # Column-level diagnostics
        columns_info = []
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / total_rows) * 100

        numerical_cols = []
        categorical_cols = []
        high_cardinality_cols = []
        constant_cols = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = int(missing_counts[col])
            missing_pct = round(float(missing_pcts[col]), 2)
            n_unique = int(df[col].nunique(dropna=True))

            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            # Detect constant columns
            if n_unique <= 1:
                constant_cols.append(col)

            # Detect high cardinality / potential IDs
            if not is_numeric and n_unique > (total_rows * 0.7) and total_rows > 50:
                high_cardinality_cols.append(col)

            if is_numeric and col != target_column:
                numerical_cols.append(col)
            elif not is_numeric and col != target_column:
                categorical_cols.append(col)

            columns_info.append({
                "column": col,
                "dtype": dtype,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_values": n_unique,
                "is_numeric": is_numeric
            })

        # Calculate Data Health Score (0 - 100)
        # Deduct for missing values, duplicates, constant columns
        avg_missing_pct = missing_pcts.mean() if not missing_pcts.empty else 0
        missing_penalty = min(avg_missing_pct * 1.5, 40)
        duplicate_penalty = min(duplicate_pct * 1.2, 25)
        constant_penalty = min(len(constant_cols) * 5, 20)
        health_score = max(0, int(100 - missing_penalty - duplicate_penalty - constant_penalty))

        # Correlation matrix for numeric columns
        corr_matrix = {}
        if len(numerical_cols) >= 2:
            num_df = df[numerical_cols].dropna()
            if not num_df.empty:
                corr = num_df.corr().round(3)
                corr_matrix = corr.to_dict()

        # Target column profile
        target_info = {}
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            is_target_num = pd.api.types.is_numeric_dtype(target_series)
            n_target_unique = target_series.nunique(dropna=True)
            
            # Problem type inference
            if not is_target_num or n_target_unique <= 10:
                if n_target_unique == 2:
                    problem_type = "binary_classification"
                else:
                    problem_type = "multiclass_classification"
                value_counts = target_series.value_counts(dropna=False, normalize=True).round(3).to_dict()
                target_info = {
                    "problem_type": problem_type,
                    "unique_classes": n_target_unique,
                    "class_distribution": {str(k): float(v) for k, v in value_counts.items()},
                    "is_imbalanced": (min(value_counts.values()) < 0.2) if len(value_counts) > 1 else False
                }
            else:
                problem_type = "regression"
                target_info = {
                    "problem_type": problem_type,
                    "mean": round(float(target_series.mean()), 3),
                    "std": round(float(target_series.std()), 3),
                    "min": round(float(target_series.min()), 3),
                    "max": round(float(target_series.max()), 3),
                    "skewness": round(float(target_series.skew()), 3)
                }
        else:
            problem_type = "unknown"

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "health_score": health_score,
            "columns_info": columns_info,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "high_cardinality_cols": high_cardinality_cols,
            "constant_cols": constant_cols,
            "corr_matrix": corr_matrix,
            "problem_type": problem_type,
            "target_info": target_info
        }
