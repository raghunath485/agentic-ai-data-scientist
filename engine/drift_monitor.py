"""
Production Data Drift & Covariate Shift Monitoring Engine.
Computes two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI)
to detect distribution degradation in incoming production data batches.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import plotly.graph_objects as go


class DriftMonitor:
    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between two distributions."""
        try:
            # Create quantile bins based on baseline
            quantiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(baseline, quantiles)
            bin_edges[0] -= 1e-5
            bin_edges[-1] += 1e-5

            base_counts, _ = np.histogram(baseline, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            base_pct = base_counts / len(baseline)
            curr_pct = curr_counts / len(current)

            # Avoid division by zero
            base_pct = np.where(base_pct == 0, 1e-4, base_pct)
            curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)

            psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
            return round(float(psi_val), 4)
        except Exception:
            return 0.0

    @classmethod
    def scan_dataset_drift(
        cls,
        baseline_df: pd.DataFrame,
        incoming_df: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Runs comprehensive KS and PSI scans across all provided features.
        """
        drift_results = []
        overall_drift_count = 0

        for feat in features:
            if feat not in baseline_df.columns or feat not in incoming_df.columns:
                continue

            base_col = baseline_df[feat].dropna().values
            curr_col = incoming_df[feat].dropna().values

            if len(base_col) < 5 or len(curr_col) < 5:
                continue

            # If numeric
            if pd.api.types.is_numeric_dtype(baseline_df[feat]):
                ks_stat, p_val = ks_2samp(base_col, curr_col)
                psi_val = cls.calculate_psi(base_col, curr_col)

                # Determine status
                if psi_val > 0.25 or p_val < 0.01:
                    status = "Critical Drift Alert"
                    color = "#f43f5e"
                    overall_drift_count += 1
                elif psi_val > 0.10 or p_val < 0.05:
                    status = "Moderate Shift"
                    color = "#fbbf24"
                else:
                    status = "Distribution Stable"
                    color = "#34d399"

                drift_results.append({
                    "Feature": feat,
                    "Type": "Numeric",
                    "PSI Metric": psi_val,
                    "KS p-value": round(float(p_val), 4),
                    "Status": status,
                    "Color": color
                })

        drift_df = pd.DataFrame(drift_results)

        # Plotly Drift Gauge
        avg_psi = drift_df["PSI Metric"].mean() if not drift_df.empty else 0.0
        fig_drift = go.Figure(go.Bar(
            x=drift_df["Feature"] if not drift_df.empty else [],
            y=drift_df["PSI Metric"] if not drift_df.empty else [],
            marker_color=drift_df["Color"] if not drift_df.empty else [],
            text=drift_df["Status"] if not drift_df.empty else [],
            textposition="auto"
        ))
        fig_drift.add_hline(y=0.10, line_dash="dot", line_color="#fbbf24", annotation_text="Moderate Shift (0.10)")
        fig_drift.add_hline(y=0.25, line_dash="dash", line_color="#f43f5e", annotation_text="Critical Drift (0.25)")
        fig_drift.update_layout(
            title="Population Stability Index (PSI) Drift Radar",
            xaxis_title="Features Analyzed",
            yaxis_title="PSI Score (Divergence)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=340,
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return {
            "drift_df": drift_df,
            "drift_figure": fig_drift,
            "total_features_scanned": len(drift_results),
            "drifted_features_count": overall_drift_count,
            "mean_psi": round(float(avg_psi), 4)
        }
