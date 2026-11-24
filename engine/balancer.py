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
        """
        counts = df[target_column].value_counts()
        max_count = counts.max()
        augmented_dfs = [df.copy()]
        synthetic_records = []

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]

        for cls_label, count in counts.items():
            diff = max_count - count
            if diff <= 0 or len(num_cols) < 2:
                continue

            sub_df = df[df[target_column] == cls_label]
            if len(sub_df) < 2:
                continue

            # Fit nearest neighbors on minority class
            k = min(k_neighbors, len(sub_df) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(sub_df[num_cols].values)
            distances, indices = nn.kneighbors(sub_df[num_cols].values)

            # Generate synthetic samples
            for _ in range(diff):
                idx = np.random.randint(0, len(sub_df))
                neighbor_idx = np.random.choice(indices[idx][1:])
                alpha = np.random.uniform(0.1, 0.9)

                sample_vals = sub_df[num_cols].iloc[idx].values + alpha * (
                    sub_df[num_cols].iloc[neighbor_idx].values - sub_df[num_cols].iloc[idx].values
                )
                
                synth_row = dict(zip(num_cols, sample_vals))
                synth_row[target_column] = cls_label
                # Copy categorical features from base row
                for cat_col in df.columns:
                    if cat_col not in num_cols and cat_col != target_column:
                        synth_row[cat_col] = sub_df[cat_col].iloc[idx]

                synthetic_records.append(synth_row)

        synth_df = pd.DataFrame(synthetic_records) if synthetic_records else pd.DataFrame()
        final_balanced_df = pd.concat([df, synth_df], ignore_index=True) if not synth_df.empty else df.copy()

        # Generate 2D PCA visual comparison of Real vs Synthetic
        fig_comp = None
        if not synth_df.empty and len(num_cols) >= 2:
            pca = PCA(n_components=2, random_state=42)
            real_pca = pca.fit_transform(df[num_cols].dropna())
            synth_pca = pca.transform(synth_df[num_cols].dropna())

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
