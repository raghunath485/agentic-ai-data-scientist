"""
Automated Bayesian Hyperparameter Optimization Engine via Optuna.
Executes parallel trials, logs convergence history, and visualizes optimization curves.
"""

from typing import Dict, Any, List
import optuna
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor

# Disable verbose optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianTuner:
    @staticmethod
    def tune_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "HistGradientBoosting",
        problem_type: str = "binary_classification",
        n_trials: int = 15
    ) -> Dict[str, Any]:
        """
        Runs Bayesian hyperparameter optimization over candidate search space.
        """
        is_classification = "classification" in problem_type
        scoring = "f1_weighted" if is_classification else "r2"
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if is_classification and y_train.value_counts().min() >= 3 else KFold(n_splits=3, shuffle=True, random_state=42)

        trial_records = []

        def objective(trial: optuna.Trial) -> float:
            if "Gradient" in model_type:
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                max_iter = trial.suggest_int("max_iter", 50, 200, step=25)
                max_depth = trial.suggest_int("max_depth", 3, 12)
                l2_reg = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)

                if is_classification:
                    clf = HistGradientBoostingClassifier(
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        max_depth=max_depth,
                        l2_regularization=l2_reg,
                        random_state=42
                    )
                else:
                    clf = HistGradientBoostingRegressor(
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        max_depth=max_depth,
                        l2_regularization=l2_reg,
                        random_state=42
                    )
            else:
                n_est = trial.suggest_int("n_estimators", 50, 150, step=25)
                max_depth = trial.suggest_int("max_depth", 4, 16)
                min_samples = trial.suggest_int("min_samples_split", 2, 8)

                if is_classification:
                    clf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_depth,
                        min_samples_split=min_samples,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    clf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_depth,
                        min_samples_split=min_samples,
                        random_state=42,
                        n_jobs=-1
                    )

            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            mean_score = float(np.mean(scores))
            trial_records.append({"trial": trial.number + 1, "score": round(mean_score, 4)})
            return mean_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_val = round(float(study.best_value), 4)

        # Plotly convergence figure
        trials_df = pd.DataFrame(trial_records)
        trials_df["best_so_far"] = trials_df["score"].cummax()

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=trials_df["trial"],
            y=trials_df["score"],
            mode="markers",
            name="Trial Score",
            marker=dict(color="#a855f7", size=8)
        ))
        fig_conv.add_trace(go.Scatter(
            x=trials_df["trial"],
            y=trials_df["best_so_far"],
            mode="lines",
            name="Optimal Frontier",
            line=dict(color="#00f2fe", width=3)
        ))
        fig_conv.update_layout(
            title=f"Optuna Bayesian Convergence ({model_type})",
            xaxis_title="Trial Number",
            yaxis_title=f"Cross-Validation Metric ({scoring})",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=320,
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return {
            "best_params": best_params,
            "best_score": best_val,
            "convergence_fig": fig_conv,
            "n_trials": n_trials,
            "model_type": model_type
        }
