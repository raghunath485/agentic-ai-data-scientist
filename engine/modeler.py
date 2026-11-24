"""
Engine module for Model Training, Cross-Validation, Evaluation, and Serialization.
Benchmarks multiple models, extracts feature importances, and creates deployment assets.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from core.state import ModelResult


class ModelTrainer:
    @staticmethod
    def benchmark_models(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Trains and evaluates multiple competitive models with Cross-Validation.
        """
        results: Dict[str, ModelResult] = {}
        candidate_models = {}

        if problem_type in ["binary_classification", "multiclass_classification"]:
            candidate_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
                "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42)
            }
            scoring = "f1_weighted"
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if y_train.value_counts().min() >= cv_folds else None
        else:
            candidate_models = {
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
                "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=100, random_state=42)
            }
            scoring = "r2"
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in candidate_models.items():
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
            except Exception:
                cv_mean = 0.0
                cv_std = 0.0

            # Fit on full training set
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {}
            if problem_type in ["binary_classification", "multiclass_classification"]:
                acc = float(accuracy_score(y_test, preds))
                bal_acc = float(balanced_accuracy_score(y_test, preds))
                prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
                f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                
                # ROC-AUC if probability estimates available
                auc_score = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_test)
                        if problem_type == "binary_classification":
                            auc_score = float(roc_auc_score(y_test, probs[:, 1]))
                        else:
                            auc_score = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
                    except Exception:
                        auc_score = None

                cm = confusion_matrix(y_test, preds).tolist()

                metrics = {
                    "Accuracy": round(acc, 4),
                    "Balanced Accuracy": round(bal_acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1 Score": round(f1, 4),
                    "ROC-AUC": round(auc_score, 4) if auc_score is not None else None,
                    "Confusion Matrix": cm
                }
            else:
                mse = float(mean_squared_error(y_test, preds))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_test, preds))
                r2 = float(r2_score(y_test, preds))

                metrics = {
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "R2 Score": round(r2, 4),
                    "MSE": round(mse, 4)
                }

            # Feature importances
            feat_imp = {}
            feature_names = list(X_train.columns)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_imp = {feat: round(float(imp), 4) for feat, imp in zip(feature_names, importances)}
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
                feat_imp = {feat: round(float(c), 4) for feat, c in zip(feature_names, coefs)}

            # Sort top features
            if feat_imp:
                feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15])

            results[name] = ModelResult(
                model_name=name,
                model_obj=model,
                metrics=metrics,
                feature_importances=feat_imp,
                cv_mean=round(cv_mean, 4),
                cv_std=round(cv_std, 4)
            )

        # Select best model
        if problem_type in ["binary_classification", "multiclass_classification"]:
            best_model_name = max(results.keys(), key=lambda k: results[k].metrics["F1 Score"])
        else:
            best_model_name = max(results.keys(), key=lambda k: results[k].metrics["R2 Score"])

        return {
            "model_results": results,
            "best_model_name": best_model_name,
            "best_model_obj": results[best_model_name].model_obj
        }

    @staticmethod
    def export_best_pipeline(
        best_model: Any,
        preprocessor: Any,
        label_encoder: Optional[Any],
        output_dir: str = "outputs/models"
    ) -> str:
        """Saves the complete inference pipeline to disk."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best_pipeline.joblib")
        
        pipeline_bundle = {
            "model": best_model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder
        }
        joblib.dump(pipeline_bundle, model_path)
        return model_path

    @staticmethod
    def generate_inference_snippet(
        model_name: str,
        problem_type: str,
        feature_names: List[str]
    ) -> str:
        """Generates ready-to-run Python code for inference."""
        sample_feature_lines = "\n".join([f'    "{f}": [None],' for f in feature_names[:5]])
        return f'''# Standalone Inference Script for {model_name}
import joblib
import pandas as pd

# 1. Load the trained pipeline bundle
bundle = joblib.load("outputs/models/best_pipeline.joblib")
model = bundle["model"]
preprocessor = bundle["preprocessor"]
label_encoder = bundle.get("label_encoder")

# 2. Example payload (replace with your raw input)
sample_data = {{
    # Provide values for features:
{sample_feature_lines}
}}
df_input = pd.DataFrame(sample_data)

# 3. Transform and predict
X_transformed = preprocessor.transform(df_input)
raw_predictions = model.predict(X_transformed)

# 4. Decode if classification
if label_encoder is not None:
    final_predictions = label_encoder.inverse_transform(raw_predictions)
else:
    final_predictions = raw_predictions

print("Predictions:", final_predictions)
'''
