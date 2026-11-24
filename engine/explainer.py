"""
Engine module for Game-Theoretic SHAP (SHapley Additive exPlanations).
Computes observation-level Shapley values and renders interactive Plotly Waterfall charts.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ModelExplainer:
    @staticmethod
    def compute_shap_breakdown(
        model: Any,
        preprocessor: Any,
        input_record_df: pd.DataFrame,
        background_sample_df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates feature attribution breakdown for a single input record.
        Uses TreeExplainer for tree models or Linear/Kernel attribution for linear baselines.
        """
        try:
            # Transform input record and background sample
            X_trans_rec = preprocessor.transform(input_record_df)
            X_trans_bg = preprocessor.transform(background_sample_df.head(50))

            if hasattr(X_trans_rec, "toarray"):
                X_trans_rec = X_trans_rec.toarray()
            if hasattr(X_trans_bg, "toarray"):
                X_trans_bg = X_trans_bg.toarray()

            # If tree ensemble
            if hasattr(model, "feature_importances_") and hasattr(model, "estimators_"):
                import shap
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_trans_rec)
                
                # If binary classification with 2 arrays
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                elif shap_vals.ndim == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = shap_vals[0]
                base_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
            
            # Linear model coefficients attribution approximation
            elif hasattr(model, "coef_"):
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                diff = X_trans_rec[0] - np.mean(X_trans_bg, axis=0)
                vals = coefs * diff
                if hasattr(model, "intercept_"):
                    base_value = float(model.intercept_[0]) if hasattr(model.intercept_, "__getitem__") else float(model.intercept_)
                else:
                    base_value = 0.0
            
            # Fallback permutation heuristic attribution
            else:
                base_pred = float(model.predict_proba(X_trans_rec)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(X_trans_rec)[0])
                vals = np.zeros(X_trans_rec.shape[1])
                bg_mean = np.mean(X_trans_bg, axis=0)
                for j in range(min(15, X_trans_rec.shape[1])):
                    x_pert = X_trans_rec.copy()
                    x_pert[0, j] = bg_mean[j]
                    pred_pert = float(model.predict_proba(x_pert)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(x_pert)[0])
                    vals[j] = base_pred - pred_pert
                base_value = float(np.mean(model.predict_proba(X_trans_bg)[:, 1])) if hasattr(model, "predict_proba") else 0.5

            # Align with feature names
            f_names = feature_names if len(feature_names) == len(vals) else [f"Feature_{i}" for i in range(len(vals))]
            attributions = {name: round(float(val), 4) for name, val in zip(f_names, vals)}
            sorted_attrs = dict(sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

            return {
                "base_value": round(base_value, 4),
                "attributions": sorted_attrs,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

    @staticmethod
    def render_waterfall_chart(
        attributions: Dict[str, float],
        base_value: float,
        title: str = "SHAP Decision Attribution Waterfall"
    ) -> go.Figure:
        """
        Renders a sleek Plotly Waterfall chart visualizing positive & negative pushes towards prediction.
        """
        names = list(attributions.keys())
        values = list(attributions.values())

        # Construct waterfall elements
        x_data = ["Baseline Expected"] + names + ["Final Outcome"]
        y_data = [base_value] + values + [0]  # Plotly waterfall total
        measure = ["absolute"] + ["relative"] * len(names) + ["total"]

        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=x_data,
            textposition="outside",
            text=[f"{v:+.3f}" if i > 0 and i <= len(names) else f"{v:.3f}" for i, v in enumerate([base_value] + values + [sum(values) + base_value])],
            y=y_data,
            connector={"line": {"color": "rgba(255,255,255,0.2)"}},
            decreasing={"marker": {"color": "#f43f5e"}},
            increasing={"marker": {"color": "#00f2fe"}},
            totals={"marker": {"color": "#a855f7"}}
        ))

        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            showlegend=False,
            height=380,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        return fig
