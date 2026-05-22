"""
visualization.py
------------------
Comprehensive visualization suite for EDA, model evaluation, and explainability.

Outputs all figures to the visualizations/ directory.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

# Set beautiful seaborn theme for all plots
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300  # High resolution for all plots

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

# ── Style defaults ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "viridis"
FIG_DPI = 150


def _save(fig, viz_dir, filename):
    path = os.path.join(viz_dir, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] Saved -> {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df, target_col, viz_dir):
    """Bar chart of target class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    counts.plot.bar(ax=ax, color=sns.color_palette(PALETTE, len(counts)), edgecolor="black")
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Match Outcome Group")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 100, f"{v:,}", ha="center", fontsize=10)
    _save(fig, viz_dir, "01_class_distribution.png")


def plot_correlation_heatmap(df, numerical_cols, viz_dir):
    """Pearson correlation heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, viz_dir, "02_correlation_heatmap.png")


def plot_boxplots(df, numerical_cols, viz_dir):
    """Boxplots for outlier detection across numerical features."""
    n = len(numerical_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette(PALETTE, n)[i])
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Outlier Detection - Boxplots", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, viz_dir, "03_boxplots.png")


def plot_feature_distributions_by_target(df, numerical_cols, target_col, viz_dir):
    """Distributions of selected numerical features split by target class."""
    selected = numerical_cols[:6]  # top 6 for readability
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(selected):
        for cls in sorted(df[target_col].unique()):
            subset = df[df[target_col] == cls][col]
            axes[i].hist(subset, bins=30, alpha=0.5, label=cls)
        axes[i].set_title(col)
        axes[i].legend(fontsize=7)
    fig.suptitle("Feature Distributions by Match Outcome", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, viz_dir, "04_distributions_by_target.png")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(fitted_models, X_test, y_test, classes, viz_dir, display_labels=None):
    """Confusion matrix heatmaps for each model."""
    labels = display_labels if display_labels is not None else classes
    n = len(fitted_models)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10))
    axes = axes.flatten()
    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False, values_format="d")
        axes[i].set_title(name, fontsize=10, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, viz_dir, "05_confusion_matrices.png")


def plot_roc_curves(fitted_models, X_test, y_test, classes, viz_dir, display_labels=None):
    """ROC curves (One-vs-Rest) for each model."""
    labels = display_labels if display_labels is not None else classes
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for cls_idx, cls_name in enumerate(labels):
        ax = axes[cls_idx]
        for name, model in fitted_models.items():
            try:
                y_prob = model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], y_prob[:, cls_idx])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"ROC - {cls_name}", fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, viz_dir, "06_roc_curves.png")


def plot_model_comparison(comparison_df, viz_dir):
    """Bar chart comparing models by key metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1_Weighted"]
    df_plot = comparison_df.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_plot))
    width = 0.18
    for i, m in enumerate(metrics):
        vals = pd.to_numeric(df_plot[m], errors="coerce")
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_plot["Model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, viz_dir, "07_model_comparison.png")


def plot_feature_importance(fitted_models, feature_names, viz_dir):
    """Feature importance from Random Forest and XGBoost."""
    for name in ["Random Forest", "XGBoost"]:
        model = fitted_models.get(name)
        if model is None:
            continue
        # Extract the classifier from the pipeline
        clf = model.named_steps.get("model") or model[-1]
        importances = clf.feature_importances_

        # Align with one-hot encoded feature names
        preprocessor = model.named_steps["preprocessor"]
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(ohe.get_feature_names_out())
        except Exception:
            cat_names = []
        num_names = list(preprocessor.transformers_[0][2])  # numerical col names
        all_names = num_names + cat_names

        selector = model.named_steps.get("selector")
        if selector is not None:
            mask = selector.get_support()
            all_names = [name for name, m in zip(all_names, mask) if m]

        # If lengths mismatch, fallback to indices
        if len(all_names) != len(importances):
            all_names = [f"f{i}" for i in range(len(importances))]

        idx = np.argsort(importances)[-15:]  # top-15

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(idx)), importances[idx], color=sns.color_palette(PALETTE, len(idx)))
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([all_names[i] for i in idx])
        ax.set_title(f"Feature Importance - {name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fname = f"08_feature_importance_{name.lower().replace(' ', '_')}.png"
        _save(fig, viz_dir, fname)


def plot_learning_curves(fitted_models, X_train, y_train, viz_dir):
    """Learning curves for overfitting analysis (selected models)."""
    selected = ["Logistic Regression", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, len(selected), figsize=(7 * len(selected), 5))

    for i, name in enumerate(selected):
        model = fitted_models.get(name)
        if model is None:
            continue
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1,
        )
        axes[i].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
        axes[i].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
        axes[i].fill_between(train_sizes,
                             train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        axes[i].fill_between(train_sizes,
                             val_scores.mean(axis=1) - val_scores.std(axis=1),
                             val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        axes[i].set_title(f"Learning Curve - {name}", fontweight="bold")
        axes[i].set_xlabel("Training Samples")
        axes[i].set_ylabel("F1 (weighted)")
        axes[i].legend()
    fig.suptitle("Learning Curves - Overfitting Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, viz_dir, "09_learning_curves.png")


def plot_shap_summary(fitted_models, X_test, preprocessor, viz_dir):
    """SHAP summary plot for XGBoost (Explainable AI)."""
    try:
        import shap
    except ImportError:
        print("  [Viz] SHAP not installed - skipping.")
        return

    xgb_model = fitted_models.get("XGBoost")
    if xgb_model is None:
        return

    clf = xgb_model.named_steps.get("model") or xgb_model[-1]
    prep = xgb_model.named_steps["preprocessor"]
    X_transformed = prep.transform(X_test)

    # Feature names
    try:
        ohe = prep.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out())
    except Exception:
        cat_names = []
    num_names = list(prep.transformers_[0][2])
    all_names = num_names + cat_names

    selector = xgb_model.named_steps.get("selector")
    if selector is not None:
        mask = selector.get_support()
        all_names = [name for name, m in zip(all_names, mask) if m]
        X_transformed = selector.transform(X_transformed)

    # SHAP (Sampled to avoid hanging)
    np.random.seed(42)
    sample_indices = np.random.choice(X_transformed.shape[0], size=min(200, X_transformed.shape[0]), replace=False)
    X_sample = X_transformed[sample_indices]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(12, 8))
    # For multi-class, shap_values is a list; use first class or average
    if isinstance(shap_values, list):
        shap_vals = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_vals = shap_values

    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=all_names if len(all_names) == X_sample.shape[1] else None,
        show=False,
        max_display=15,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Summary - XGBoost", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, viz_dir, "10_shap_summary_xgboost.png")

    # Also do SHAP for Random Forest if available
    rf_model = fitted_models.get("Random Forest")
    if rf_model:
        try:
            rf_clf = rf_model.named_steps.get("model") or rf_model[-1]
            rf_prep = rf_model.named_steps["preprocessor"]
            X_rf = rf_prep.transform(X_test)
            
            rf_selector = rf_model.named_steps.get("selector")
            if rf_selector is not None:
                X_rf = rf_selector.transform(X_rf)

            X_rf_sample = X_rf[sample_indices]

            explainer_rf = shap.TreeExplainer(rf_clf)
            shap_vals_rf = explainer_rf.shap_values(X_rf_sample)
            if isinstance(shap_vals_rf, list):
                shap_vals_rf = np.abs(np.array(shap_vals_rf)).mean(axis=0)
            fig_rf = plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_vals_rf, X_rf_sample,
                feature_names=all_names if len(all_names) == X_rf_sample.shape[1] else None,
                show=False, max_display=15,
            )
            fig_rf = plt.gcf()
            fig_rf.suptitle("SHAP Summary - Random Forest", fontsize=14, fontweight="bold", y=1.02)
            _save(fig_rf, viz_dir, "11_shap_summary_random_forest.png")
        except Exception as e:
            print(f"  [Viz] SHAP for RF failed: {e}")
