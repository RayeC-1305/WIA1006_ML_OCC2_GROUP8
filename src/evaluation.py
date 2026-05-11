"""
evaluation.py
--------------
Model evaluation utilities.

Computes Accuracy, Precision, Recall, F1 (all weighted), ROC-AUC (OvR),
and generates per-model classification reports + a final comparison table.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test, model_name: str, classes: list) -> dict:
    """
    Evaluate a single model and return a metrics dict.

    Parameters
    ----------
    model      : fitted sklearn pipeline / estimator
    X_test     : test features (raw, pipeline handles transform)
    y_test     : true labels
    model_name : display name
    classes    : list of class labels for ROC-AUC binarisation

    Returns
    -------
    dict with Accuracy, Precision, Recall, F1, ROC_AUC
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (One-vs-Rest) — requires probability estimates
    try:
        y_prob = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        roc = roc_auc_score(y_test_bin, y_prob, average="weighted", multi_class="ovr")
    except Exception:
        roc = np.nan  # SVM with certain kernels may not support predict_proba

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Weighted": round(f1, 4),
        "ROC_AUC": round(roc, 4) if not np.isnan(roc) else "N/A",
    }


def build_comparison_table(results: list[dict], reports_dir: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame sorted by F1_Weighted and save as CSV.
    """
    df = pd.DataFrame(results)
    # Sort: models with valid ROC_AUC by F1 descending
    df = df.sort_values("F1_Weighted", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed rank
    df.index.name = "Rank"

    path = os.path.join(reports_dir, "model_comparison.csv")
    df.to_csv(path)
    print(f"\n[Evaluation] Comparison table saved -> {path}")
    print(df.to_string())
    return df
