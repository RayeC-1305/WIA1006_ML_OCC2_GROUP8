"""
train_models.py
----------------
Train 7 ML models with Logistic Regression as baseline.

Models:
    1. Logistic Regression   (Linear - Baseline)
    2. Decision Tree          (Rule-Based)
    3. Random Forest          (Bagging Ensemble)
    4. XGBoost                (Boosting Ensemble)
    5. SVM                    (Margin-Based)
    6. MLPClassifier          (Neural Network)
    7. Voting Classifier      (Ensemble: LR + RF + XGB)

Supports both class_weight='balanced' and SMOTE for imbalance handling.
Hyperparameter tuning via RandomizedSearchCV(cv=3, scoring='f1_weighted').
"""

import os
import sys
import warnings
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")


def _get_models_and_params(use_smote: bool = False):
    """
    Return a list of (name, estimator, param_grid) tuples.
    When use_smote is True, the param keys are prefixed with 'model__'
    because the estimator is wrapped in an ImbPipeline with SMOTE.
    """
    prefix = "model__" if use_smote else ""

    models = [
        # ---- 1. Logistic Regression (Baseline) ----
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            {
                f"{prefix}C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
        # ---- 2. Decision Tree ----
        (
            "Decision Tree",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            ),
            {
                f"{prefix}max_depth": [10, 20, 30, None],
                f"{prefix}min_samples_split": [2, 5, 10, 20],
                f"{prefix}min_samples_leaf": [1, 5, 10],
            },
        ),
        # ---- 3. Random Forest ----
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            {
                f"{prefix}n_estimators": [100, 200, 300],
                f"{prefix}max_depth": [10, 20, 30, None],
                f"{prefix}min_samples_split": [2, 5, 10],
            },
        ),
        # ---- 4. XGBoost ----
        (
            "XGBoost",
            XGBClassifier(
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1,
            ),
            {
                f"{prefix}learning_rate": [0.01, 0.05, 0.1, 0.2],
                f"{prefix}max_depth": [3, 5, 7, 10],
                f"{prefix}n_estimators": [100, 200, 300],
                f"{prefix}subsample": [0.8, 1.0],
            },
        ),
        # ---- 5. SVM ----
        (
            "SVM",
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
            {
                f"{prefix}C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        # ---- 6. Neural Network (MLP) ----
        (
            "Neural Network (MLP)",
            MLPClassifier(
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            {
                f"{prefix}hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
                f"{prefix}alpha": [0.0001, 0.001, 0.01],
                f"{prefix}learning_rate_init": [0.001, 0.01],
            },
        ),
    ]
    return models


def train_all_models(
    preprocessor,
    X_train,
    y_train,
    models_dir: str,
    use_smote: bool = False,
):
    """
    Train all models with GridSearchCV and save the best estimators.

    Parameters
    ----------
    preprocessor : ColumnTransformer (unfitted)
    X_train      : raw training features
    y_train      : training labels
    models_dir   : directory to save .pkl models
    use_smote    : if True, inject SMOTE before the classifier in the pipeline

    Returns
    -------
    dict mapping model_name → fitted pipeline
    """
    models_and_params = _get_models_and_params(use_smote=use_smote)
    fitted = {}

    for name, estimator, param_grid in models_and_params:
        print(f"\n{'-'*60}")
        print(f"  Training: {name}" + (" [SMOTE]" if use_smote else ""))
        print(f"{'-'*60}")
        sys.stdout.flush()

        # Build pipeline
        if use_smote:
            pipe = ImbPipeline([
                ("preprocessor", preprocessor),
                ("selector", SelectKBest(score_func=f_classif, k=15)), # Feature Selection
                ("smote", SMOTEENN(random_state=42)),
                ("model", estimator),
            ])
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("selector", SelectKBest(score_func=f_classif, k=15)), # Feature Selection
                ("model", estimator),
            ])

        # Hyperparameter tuning
        if param_grid:
            grid = RandomizedSearchCV(
                pipe,
                param_distributions={f"model__{k}" if not k.startswith("model__") else k: v
                                     for k, v in param_grid.items()},
                n_iter=10, # To keep training time reasonable
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
                random_state=42,
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV F1:  {grid.best_score_:.4f}")
        else:
            pipe.fit(X_train, y_train)
            best_pipe = pipe

        # Save model
        suffix = "_smote" if use_smote else ""
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + suffix + ".pkl"
        joblib.dump(best_pipe, os.path.join(models_dir, fname))

        fitted[name] = best_pipe

    # ==== Add Voting Classifier ====
    print(f"\n{'-'*60}")
    print("  Training: Voting Classifier" + (" [SMOTE]" if use_smote else ""))
    print(f"{'-'*60}")
    sys.stdout.flush()

    # Extract best estimators from the fitted pipelines to construct the voting classifier
    estimators = [
        ("lr", fitted["Logistic Regression"].named_steps["model"]),
        ("rf", fitted["Random Forest"].named_steps["model"]),
        ("xgb", fitted["XGBoost"].named_steps["model"])
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting="hard") # using hard voting since LinearSVC doesn't support predict_proba by default, but we aren't including SVC. Let's stick to hard voting to be safe, or soft if all support it. XGB/RF/LR support soft.
    voting_clf.voting = "soft"

    if use_smote:
        voting_pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("smote", SMOTEENN(random_state=42)),
            ("model", voting_clf)
        ])
    else:
        voting_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", voting_clf)
        ])
    
    voting_pipe.fit(X_train, y_train)
    fitted["Voting Classifier"] = voting_pipe
    
    suffix = "_smote" if use_smote else ""
    joblib.dump(voting_pipe, os.path.join(models_dir, f"voting_classifier{suffix}.pkl"))

    print(f"\n[Training] All models saved to: {models_dir}")
    return fitted
