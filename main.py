"""
main.py
--------
End-to-end orchestrator for the Dating App Match Prediction ML project.

Workflow:
    Raw Dataset → Data Cleaning → EDA → Feature Engineering →
    Encoding + Scaling → Train-Test Split → Cross Validation →
    Train 6 Models (class_weight + SMOTE) → Hyperparameter Tuning →
    Model Evaluation → Visualization + Explainability →
    Final Comparison → Best Model Selection
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dating_app_behavior_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

for d in [MODELS_DIR, REPORTS_DIR, VIZ_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.preprocessing import (
    load_and_clean,
    build_preprocessor,
    split_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    label_encoder,
)
from src.feature_engineering import add_engineered_features
from src.train_models import train_all_models
from src.evaluation import evaluate_model, build_comparison_table
from src.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_feature_distributions_by_target,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance,
    plot_learning_curves,
    plot_shap_summary,
)


def main():
    print("=" * 70)
    print("  DATING APP MATCH PREDICTION - ML PIPELINE")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # 1. LOAD & CLEAN
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 1: Data Loading & Cleaning")
    df = load_and_clean(DATA_PATH)

    # ──────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 2: Feature Engineering")
    df = add_engineered_features(df)

    # ──────────────────────────────────────────────────────────────────────
    # 3. EDA VISUALISATIONS
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 3: Exploratory Data Analysis")
    plot_class_distribution(df, TARGET, VIZ_DIR)
    plot_correlation_heatmap(df, NUMERICAL_FEATURES, VIZ_DIR)
    plot_boxplots(df, NUMERICAL_FEATURES, VIZ_DIR)
    plot_feature_distributions_by_target(df, NUMERICAL_FEATURES, TARGET, VIZ_DIR)

    # ──────────────────────────────────────────────────────────────────────
    # 4. TRAIN-TEST SPLIT
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 4: Train-Test Split")
    X_train, X_test, y_train, y_test = split_data(df)
    classes = sorted(y_test.unique().tolist())
    class_names = list(label_encoder.classes_)  # human-readable names
    print(f"  Classes (encoded): {classes}")
    print(f"  Class names:       {class_names}")

    # ──────────────────────────────────────────────────────────────────────
    # 5. TRAIN MODELS — class_weight='balanced' approach
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 5a: Training Models (class_weight='balanced')")
    preprocessor = build_preprocessor()
    fitted_balanced = train_all_models(
        preprocessor, X_train, y_train, MODELS_DIR, use_smote=False
    )

    # ──────────────────────────────────────────────────────────────────────
    # 6. TRAIN MODELS — SMOTE approach
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 5b: Training Models (SMOTE)")
    preprocessor_smote = build_preprocessor()
    fitted_smote = train_all_models(
        preprocessor_smote, X_train, y_train, MODELS_DIR, use_smote=True
    )

    # ──────────────────────────────────────────────────────────────────────
    # 7. EVALUATE BOTH APPROACHES
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 6: Model Evaluation")

    # Evaluate balanced models
    print("\n-- Results: class_weight='balanced' --")
    results_balanced = []
    for name, model in fitted_balanced.items():
        metrics = evaluate_model(model, X_test, y_test, name, classes)
        results_balanced.append(metrics)

    comp_balanced = build_comparison_table(results_balanced, REPORTS_DIR)
    comp_balanced.to_csv(os.path.join(REPORTS_DIR, "model_comparison_balanced.csv"))

    # Evaluate SMOTE models
    print("\n-- Results: SMOTE --")
    results_smote = []
    for name, model in fitted_smote.items():
        metrics = evaluate_model(model, X_test, y_test, f"{name} (SMOTE)", classes)
        results_smote.append(metrics)

    comp_smote = build_comparison_table(results_smote, REPORTS_DIR)
    comp_smote.to_csv(os.path.join(REPORTS_DIR, "model_comparison_smote.csv"))

    # Combined comparison
    all_results = results_balanced + results_smote
    comp_all = build_comparison_table(all_results, REPORTS_DIR)

    # ──────────────────────────────────────────────────────────────────────
    # 8. VISUALISATIONS — use balanced models as primary
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 7: Generating Visualisations")
    plot_confusion_matrices(fitted_balanced, X_test, y_test, classes, VIZ_DIR, display_labels=class_names)
    plot_roc_curves(fitted_balanced, X_test, y_test, classes, VIZ_DIR, display_labels=class_names)
    plot_model_comparison(comp_balanced, VIZ_DIR)
    plot_feature_importance(fitted_balanced, NUMERICAL_FEATURES + CATEGORICAL_FEATURES, VIZ_DIR)

    print("\n>> STEP 8: Learning Curves (may take a moment)")
    plot_learning_curves(fitted_balanced, X_train, y_train, VIZ_DIR)

    # ──────────────────────────────────────────────────────────────────────
    # 9. SHAP EXPLAINABILITY
    # ──────────────────────────────────────────────────────────────────────
    print("\n>> STEP 9: SHAP Explainability")
    plot_shap_summary(fitted_balanced, X_test, preprocessor, VIZ_DIR)

    # ──────────────────────────────────────────────────────────────────────
    # 10. FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Models saved to:         {MODELS_DIR}")
    print(f"  Reports saved to:        {REPORTS_DIR}")
    print(f"  Visualisations saved to: {VIZ_DIR}")

    # Print best model
    best = comp_balanced.iloc[0]
    print(f"\n  [BEST] Best Model (balanced): {best['Model']}")
    print(f"     F1 (weighted):  {best['F1_Weighted']}")
    print(f"     Accuracy:       {best['Accuracy']}")
    print(f"     ROC-AUC:        {best['ROC_AUC']}")

    best_smote = comp_smote.iloc[0]
    print(f"\n  [BEST] Best Model (SMOTE):    {best_smote['Model']}")
    print(f"     F1 (weighted):  {best_smote['F1_Weighted']}")
    print(f"     Accuracy:       {best_smote['Accuracy']}")
    print(f"     ROC-AUC:        {best_smote['ROC_AUC']}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
