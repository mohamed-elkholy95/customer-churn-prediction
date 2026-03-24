"""Quick start example for Customer Churn Prediction.

Demonstrates the core pipeline in under 30 lines:
  1. Generate synthetic customer data
  2. Train and evaluate three classifiers
  3. Get feature importance rankings
  4. Score a single customer's churn risk

Run: python examples/quickstart.py
"""
import sys
sys.path.insert(0, ".")

from src.churn_model import (
    generate_synthetic_churn_data,
    train_and_evaluate,
    get_feature_importance,
    predict_single_customer,
)

print("=" * 60)
print("  Customer Churn Prediction — Quick Start")
print("=" * 60)

# Step 1: Generate 1000 synthetic customers
df = generate_synthetic_churn_data(n_samples=1000)
churn_rate = df["churn"].mean() * 100
print(f"\n📊 Generated {len(df)} customers (churn rate: {churn_rate:.1f}%)")

# Step 2: Train all three models and compare
print("\n🧠 Training models...")
results = train_and_evaluate(df)
for name, metrics in results.items():
    print(f"  {name:25s}  F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

# Step 3: Which features matter most?
print("\n🔍 Top features (Random Forest):")
importance = get_feature_importance(df, model_name="random_forest")
for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
    bar = "█" * int(score * 40)
    print(f"  {feat:20s} {score:.3f} {bar}")

# Step 4: Score a single customer
customer = {
    "age": 25,
    "tenure": 3,
    "monthly_charges": 95,
    "total_charges": 285,
    "contract_type": "month",
    "internet_service": "Fiber",
    "payment_method": "electronic",
}
result = predict_single_customer(customer, df, "gradient_boosting")
print(f"\n👤 Single customer prediction:")
print(f"   Churn probability: {result['churn_probability']:.1%}")
print(f"   Risk level: {result['risk_emoji']} {result['risk_level']}")

print("\n✅ Done! For the full interactive experience:")
print("   streamlit run streamlit_app/app.py")
