"""
Synthetic Data Augmentation & Minority Class Balancer.
Detects acute class imbalance and generates synthetic minority samples
with real-vs-synthetic distribution projection.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go


class DataBalancer:
    @staticmethod
    def balance_classification_dataset(
        df: pd.DataFrame,
        target_column: str,
        k_neighbors: int = 4
    ) -> Dict[str, Any]:
        """
        Synthetically augments minority class records to balance binary/multiclass classification.
        Handles missing values gracefully via automated median imputation.
        """
        counts = df[target_column].value_counts()
        max_count = counts.max()
        synthetic_records = []

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]

        # Impute missing numeric values if present for KNN
        imputer = SimpleImputer(strategy="median")
        df_num_imputed = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        ) if len(num_cols) >= 2 else pd.DataFrame()

        for cls_label, count in counts.items():
            diff = max_count - count
            if diff <= 0 or len(num_cols) < 2 or df_num_imputed.empty:
                continue

            sub_mask = (df[target_column] == cls_label)
            sub_indices = df[sub_mask].index
            if len(sub_indices) < 2:
                continue

            sub_num_vals = df_num_imputed.loc[sub_indices].values

            # Fit nearest neighbors on minority class
            k = min(k_neighbors, len(sub_indices) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(sub_num_vals)
            distances, indices = nn.kneighbors(sub_num_vals)

            # Generate synthetic samples
            for _ in range(diff):
                idx = np.random.randint(0, len(sub_indices))
                neighbor_idx = np.random.choice(indices[idx][1:])
                alpha = np.random.uniform(0.1, 0.9)

                sample_vals = sub_num_vals[idx] + alpha * (
                    sub_num_vals[neighbor_idx] - sub_num_vals[idx]
                )
                
                synth_row = dict(zip(num_cols, sample_vals))
                synth_row[target_column] = cls_label
                # Copy categorical features from base row
                orig_row_idx = sub_indices[idx]
                for cat_col in df.columns:
                    if cat_col not in num_cols and cat_col != target_column:
                        synth_row[cat_col] = df.loc[orig_row_idx, cat_col]

                synthetic_records.append(synth_row)

        synth_df = pd.DataFrame(synthetic_records) if synthetic_records else pd.DataFrame()
        final_balanced_df = pd.concat([df, synth_df], ignore_index=True) if not synth_df.empty else df.copy()

        # Generate 2D PCA visual comparison of Real vs Synthetic
        fig_comp = None
        if not synth_df.empty and len(num_cols) >= 2 and not df_num_imputed.empty:
            pca = PCA(n_components=2, random_state=42)
            real_pca = pca.fit_transform(df_num_imputed)
            synth_num_imputed = imputer.transform(synth_df[num_cols])
            synth_pca = pca.transform(synth_num_imputed)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=real_pca[:, 0],
                y=real_pca[:, 1],
                mode="markers",
                name="Original Data",
                marker=dict(color="#00f2fe", size=6, opacity=0.7)
            ))
            fig_comp.add_trace(go.Scatter(
                x=synth_pca[:, 0],
                y=synth_pca[:, 1],
                mode="markers",
                name="Synthetic Augmentations",
                marker=dict(color="#f59e0b", size=7, symbol="diamond", opacity=0.85)
            ))
            fig_comp.update_layout(
                title=f"Original vs Synthetic Augmentation 2D Manifold (+{len(synth_df)} synthetic samples)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=320,
                margin=dict(l=30, r=30, t=50, b=30)
            )

        return {
            "balanced_df": final_balanced_df,
            "synthetic_count": len(synth_df),
            "original_shape": df.shape,
            "balanced_shape": final_balanced_df.shape,
            "comparison_figure": fig_comp
        }
