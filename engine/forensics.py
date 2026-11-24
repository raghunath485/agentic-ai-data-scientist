"""
Forensic Anomaly Scanner & Latent Manifold Projection Engine.
Uses IsolationForest for unsupervised anomaly detection and PCA for 2D latent space visualization.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataForensics:
    @staticmethod
    def scan_anomalies_and_project(
        df: pd.DataFrame,
        target_column: str = None,
        contamination: float = 0.05
    ) -> Dict[str, Any]:
        """
        Runs unsupervised anomaly detection (IsolationForest) and PCA 2D manifold projection.
        """
        # Select numeric columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]
        if len(num_cols) < 2:
            return {"error": "Insufficient numeric features for manifold projection and anomaly scanning."}

        X_num = df[num_cols].copy()
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # 1. Unsupervised Anomaly Detection via Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
        anomaly_scores = iso.decision_function(X_scaled)

        # 2. Latent 2D Projection via PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        exp_variance = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

        proj_df = pd.DataFrame({
            "PCA_Dim_1": coords[:, 0],
            "PCA_Dim_2": coords[:, 1],
            "Anomaly_Score": np.round(anomaly_scores, 4),
            "Status": ["Anomaly (Outlier)" if p == -1 else "Normal Cluster" for p in preds]
        })

        if target_column and target_column in df.columns:
            proj_df[target_column] = df[target_column].values

        anomaly_count = int((preds == -1).sum())
        total_records = len(df)
        anomaly_ratio = round((anomaly_count / total_records) * 100, 2)

        return {
            "projection_df": proj_df,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "total_records": total_records,
            "explained_variance": exp_variance,
            "numeric_features_used": num_cols
        }
