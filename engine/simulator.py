"""
Quantum Counterfactual & "What-If" Simulation Engine.
Computes live model response to real-time feature perturbations,
generates confidence probability vectors, and identifies high-impact counterfactual levers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class WhatIfSimulator:
    @staticmethod
    def simulate_record(
        payload: Dict[str, Any],
        model: Any,
        preprocessor: Any,
        label_encoder: Optional[Any] = None,
        problem_type: str = "binary_classification"
    ) -> Dict[str, Any]:
        """
        Runs real-time inference on a custom user payload and returns probability distributions.
        """
        df_input = pd.DataFrame([payload])
        try:
            X_trans = preprocessor.transform(df_input)
            raw_pred = model.predict(X_trans)[0]

            probabilities = None
            confidence = 1.0

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_trans)[0]
                if label_encoder is not None:
                    class_labels = [str(c) for c in label_encoder.classes_]
                elif hasattr(model, "classes_"):
                    class_labels = [str(c) for c in model.classes_]
                else:
                    class_labels = [f"Class_{i}" for i in range(len(probs))]

                probabilities = {label: round(float(p), 4) for label, p in zip(class_labels, probs)}
                confidence = float(np.max(probs))

            # Decode target label
            if label_encoder is not None:
                final_label = str(label_encoder.inverse_transform([raw_pred])[0])
            else:
                final_label = str(raw_pred)

            return {
                "prediction": final_label,
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
                "problem_type": problem_type,
                "status": "success"
            }
        except Exception as e:
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "probabilities": None,
                "error": str(e),
                "status": "failed"
            }
