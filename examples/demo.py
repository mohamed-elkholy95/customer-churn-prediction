"""Advanced demo showing the full feature set of Customer Churn Prediction.

Covers:
  - Data generation and exploration
  - Preprocessing deep-dive
  - Model training, evaluation, and comparison
  - Cross-validation for robust estimates
  - Confusion matrix analysis with business impact
  - Feature importance across all model types
  - Optimal threshold selection
  - Model persistence (save/load)

Run: python examples/demo.py
"""
import sys
sys.path.insert(0, ".")

import logging
import tempfile

import numpy as np

from src.churn_model import (
    generate_synthetic_churn_data,
    preprocess,
    train_and_evaluate,
    cross_validate_models,
    get_feature_importance,
    get_confusion_matrices,
    predict_single_customer,
    save_model,
    load_model,
    SUPPORTED_MODELS,
)

logging.basicConfig(level=logging.WARNING)


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Data Generation ──────────────────────────────────────────────────────────

section("1. Data Generation")

df = generate_synthetic_churn_data(n_samples=2000, seed=42)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate:    {df['churn'].mean():.1%}")
print(f"\nFeature types:")
for col in df.columns:
    dtype = "numeric" if np.issubdtype(df[col].dtype, np.number) else "categorical"
    unique = df[col].nunique()
    print(f"  {col:20s}  {dtype:12s}  {unique:>5} unique values")

# ── Preprocessing ────────────────────────────────────────────────────────────

section("2. Preprocessing")

X, y, scaler = preprocess(df)
print(f"Feature matrix:  {X.shape}  (samples × features)")
print(f"Label vector:    {y.shape}  (binary: 0=stayed, 1=churned)")
print(f"Scaler mean:     [{', '.join(f'{m:.2f}' for m in scaler.mean_[:4])}  ...]")
print(f"Scaler std:      [{', '.join(f'{s:.2f}' for s in scaler.scale_[:4])}  ...]")
print(f"\nAfter scaling, features have mean≈0 and std≈1:")
print(f"  Column means:  {np.abs(X.mean(axis=0)).max():.6f} (max absolute mean)")
print(f"  Column stds:   {X.std(axis=0).mean():.4f} (average std)")

# ── Train/Test Evaluation ────────────────────────────────────────────────────

section("3. Model Training & Evaluation")

results = train_and_evaluate(df, test_size=0.2)
print(f"\n{'Model':25s} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-" * 75)
for name, m in results.items():
    print(
        f"{name:25s} {m['accuracy']:9.4f} {m['precision']:10.4f} "
        f"{m['recall']:8.4f} {m['f1']:8.4f} {m['roc_auc']:9.4f}"
    )

# Find best model
best_model = max(results, key=lambda k: results[k]["f1"])
print(f"\n🏆 Best F1 score: {best_model} ({results[best_model]['f1']:.4f})")

# ── Cross-Validation ─────────────────────────────────────────────────────────

section("4. Cross-Validation (5-fold)")

cv_results = cross_validate_models(df, n_splits=5)
print(f"\n{'Model':25s} {'Accuracy':>12} {'F1':>12}")
print("-" * 55)
for name, m in cv_results.items():
    print(
        f"{name:25s} {m['mean_accuracy']:.4f}±{m['std_accuracy']:.4f} "
        f"{m['mean_f1']:.4f}±{m['std_f1']:.4f}"
    )

# ── Confusion Matrices ───────────────────────────────────────────────────────

section("5. Confusion Matrix Analysis")

cm_results = get_confusion_matrices(df, test_size=0.2)
for name, cm in cm_results.items():
    total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
    accuracy = (cm["tn"] + cm["tp"]) / total
    print(f"\n  {name}:")
    print(f"    ┌──────────────────┬──────────────────┐")
    print(f"    │  TN = {cm['tn']:4d}       │  FP = {cm['fp']:4d}       │")
    print(f"    │  (correct stay)  │  (false alarm)   │")
    print(f"    ├──────────────────┼──────────────────┤")
    print(f"    │  FN = {cm['fn']:4d}       │  TP = {cm['tp']:4d}       │")
    print(f"    │  (missed churn)  │  (caught churn)  │")
    print(f"    └──────────────────┴──────────────────┘")

# ── Feature Importance ────────────────────────────────────────────────────────

section("6. Feature Importance (all models)")

for model_name in SUPPORTED_MODELS:
    importance = get_feature_importance(df, model_name=model_name)
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n  {model_name}:")
    for feat, score in ranked:
        bar = "█" * int(score * 30)
        print(f"    {feat:20s} {score:.3f} {bar}")

# ── Single Customer Scoring ──────────────────────────────────────────────────

section("7. Single Customer Predictions")

customers = [
    {
        "label": "High-risk (new, month-to-month, Fiber)",
        "data": {
            "age": 22, "tenure": 2, "monthly_charges": 110,
            "total_charges": 220, "contract_type": "month",
            "internet_service": "Fiber", "payment_method": "electronic",
        },
    },
    {
        "label": "Low-risk (loyal, two-year, DSL)",
        "data": {
            "age": 55, "tenure": 60, "monthly_charges": 35,
            "total_charges": 2100, "contract_type": "two_year",
            "internet_service": "DSL", "payment_method": "auto",
        },
    },
    {
        "label": "Medium-risk (mid-tenure, yearly, Fiber)",
        "data": {
            "age": 38, "tenure": 18, "monthly_charges": 75,
            "total_charges": 1350, "contract_type": "year",
            "internet_service": "Fiber", "payment_method": "check",
        },
    },
]

for c in customers:
    result = predict_single_customer(c["data"], df, "gradient_boosting")
    print(f"\n  {c['label']}:")
    print(f"    Churn probability: {result['churn_probability']:.1%}")
    print(f"    Risk level:        {result['risk_emoji']} {result['risk_level']}")

# ── Model Persistence ────────────────────────────────────────────────────────

section("8. Model Persistence")

with tempfile.TemporaryDirectory() as tmpdir:
    print(f"  Saving gradient_boosting model to {tmpdir}/...")
    save_model(df, model_name="gradient_boosting", output_dir=tmpdir)

    model, scaler, metadata = load_model(
        model_name="gradient_boosting", model_dir=tmpdir
    )
    print(f"  Loaded model: {metadata['model_name']}")
    print(f"  Training samples: {metadata['n_training_samples']}")
    print(f"  Features: {metadata['feature_names']}")
    print(f"  ✅ Save/load round-trip successful!")

# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")
print(f"  Dataset:         {len(df)} customers, {df['churn'].mean():.1%} churn rate")
print(f"  Best model (F1): {best_model} ({results[best_model]['f1']:.4f})")
print(f"  Models trained:  {len(SUPPORTED_MODELS)}")
print(f"  CV folds:        5")
print(f"\n  For the interactive dashboard: streamlit run streamlit_app/app.py")
print(f"  For the REST API: uvicorn src.api.main:app --port 8011")
