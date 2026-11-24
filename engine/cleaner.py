"""
Engine module for Data Cleaning and Feature Preprocessing.
Performs leak-free imputation, encoding, and scaling via Scikit-Learn ColumnTransformer.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class DataCleaner:
    @staticmethod
    def prepare_and_clean_data(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Cleans data and produces leak-free train/test splits with fitted preprocessors.
        """
        cleaning_actions: List[str] = []
        df_work = df.copy()

        # Step 1: Remove completely empty or duplicate rows
        init_rows = len(df_work)
        df_work = df_work.drop_duplicates()
        dropped_dups = init_rows - len(df_work)
        if dropped_dups > 0:
            cleaning_actions.append(f"Removed {dropped_dups} duplicate row(s).")

        # Step 2: Drop target missing rows
        if target_column in df_work.columns:
            target_nulls = df_work[target_column].isnull().sum()
            if target_nulls > 0:
                df_work = df_work.dropna(subset=[target_column])
                cleaning_actions.append(f"Dropped {target_nulls} row(s) missing the target value '{target_column}'.")

        # Step 3: Identify non-informative columns (constants, ID columns, >70% nulls)
        cols_to_drop = []
        for col in df_work.columns:
            if col == target_column:
                continue
            
            null_ratio = df_work[col].isnull().mean()
            if null_ratio > 0.70:
                cols_to_drop.append(col)
                cleaning_actions.append(f"Dropped column '{col}' (>70% missing values: {round(null_ratio*100, 1)}%).")
                continue

            n_unique = df_work[col].nunique(dropna=True)
            if n_unique <= 1:
                cols_to_drop.append(col)
                cleaning_actions.append(f"Dropped zero-variance constant column '{col}'.")
                continue

            # Check potential high-cardinality ID column
            is_num = pd.api.types.is_numeric_dtype(df_work[col])
            if not is_num and n_unique > (len(df_work) * 0.7) and len(df_work) > 50:
                cols_to_drop.append(col)
                cleaning_actions.append(f"Dropped high-cardinality identifier column '{col}' ({n_unique} unique strings).")

        if cols_to_drop:
            df_work = df_work.drop(columns=cols_to_drop)

        # Step 4: Separate X and y
        X = df_work.drop(columns=[target_column])
        y = df_work[target_column]

        # Classify features
        numerical_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

        # Target encoding if target is categorical / text
        label_encoder = None
        is_target_numeric = pd.api.types.is_numeric_dtype(y)
        if not is_target_numeric or y.nunique() <= 10:
            problem_type = "binary_classification" if y.nunique() == 2 else "multiclass_classification"
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y.astype(str))
            y = pd.Series(y_encoded, index=y.index, name=target_column)
            cleaning_actions.append(f"Encoded categorical target '{target_column}' into {len(label_encoder.classes_)} classes.")
        else:
            problem_type = "regression"

        # Step 5: Leak-Free Train-Test Split
        stratify = y if problem_type.endswith("classification") and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        cleaning_actions.append(
            f"Partitioned data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets (stratify={'Yes' if stratify is not None else 'No'})."
        )

        # Step 6: Define Preprocessing Pipelines
        transformers = []

        if numerical_features:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipeline, numerical_features))
            cleaning_actions.append(f"Constructed Numerical Pipeline: Median Imputation + Standard Scaling for {len(numerical_features)} feature(s).")

        if categorical_features:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipeline, categorical_features))
            cleaning_actions.append(f"Constructed Categorical Pipeline: Most-Frequent Imputation + One-Hot Encoding for {len(categorical_features)} feature(s).")

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # Fit strictly on X_train to prevent leakage
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Extract generated feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
        except Exception:
            feature_names = [f"feat_{i}" for i in range(X_train_transformed.shape[1])]

        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

        return {
            "cleaned_df": df_work,
            "X_train": X_train_df,
            "X_test": X_test_df,
            "y_train": y_train,
            "y_test": y_test,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "dropped_features": cols_to_drop,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "problem_type": problem_type,
            "cleaning_actions": cleaning_actions
        }
