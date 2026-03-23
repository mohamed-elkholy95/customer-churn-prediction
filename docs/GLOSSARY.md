# 📖 ML Glossary for Churn Prediction

A reference guide to machine learning terminology used in this project, written for practitioners and interviewers alike.

---

## Data & Preprocessing

### Feature
An input variable (column) used by the model to make predictions. In this project: `age`, `tenure`, `monthly_charges`, `total_charges`, `contract_type`, `internet_service`, `payment_method`.

### Target Variable (Label)
The variable we're trying to predict. Here: `churn` (0 = stayed, 1 = left).

### Label Encoding
Converting categorical text values to integers. Example: `month → 0`, `year → 1`, `two_year → 2`. Simple but assumes ordinal relationship — fine for tree-based models.

### StandardScaler (Z-Score Normalization)
Transforms features to have mean=0 and standard deviation=1: `z = (x - μ) / σ`. Prevents high-magnitude features from dominating distance-based or gradient-based algorithms.

### Train/Test Split
Dividing data into training set (model learns from this) and test set (evaluates on unseen data). Typical ratio: 80/20. **Stratified** splitting preserves class balance in both sets.

### Synthetic Data
Artificially generated data that mimics real-world patterns. Used here for demonstration — in production, you'd use actual CRM/billing data.

---

## Models

### Logistic Regression
A linear model that predicts probability using the sigmoid function: `P(churn) = 1 / (1 + e^(-z))` where `z = w₁x₁ + w₂x₂ + ... + b`. Despite the name, it's used for **classification**, not regression.

**Strengths:** Fast, interpretable, good baseline.
**Weaknesses:** Assumes linear decision boundary.

### Random Forest
An ensemble of 100 decision trees, each trained on a random subset of data and features (bagging). Final prediction = majority vote. Randomization reduces overfitting.

**Strengths:** Handles non-linearity, robust to outliers, provides feature importance.
**Weaknesses:** Slower, less interpretable than logistic regression.

### Gradient Boosting
Sequential ensemble where each tree corrects the previous trees' errors. Optimizes a loss function by following the gradient direction (gradient descent in function space).

**Strengths:** Often most accurate, handles mixed feature types.
**Weaknesses:** Prone to overfitting, slowest to train, more hyperparameters.

### Ensemble Methods
Combining multiple models to get better predictions than any single model. **Bagging** (Random Forest) trains in parallel on random subsets. **Boosting** (Gradient Boosting) trains sequentially, focusing on hard examples.

---

## Evaluation Metrics

### Accuracy
`(TP + TN) / Total` — fraction of correct predictions. **Misleading for imbalanced data:** a dataset with 5% churn gets 95% accuracy by always predicting "no churn."

### Precision
`TP / (TP + FP)` — of all customers flagged as churners, how many actually churned? High precision = fewer wasted retention offers.

### Recall (Sensitivity / True Positive Rate)
`TP / (TP + FN)` — of all actual churners, how many did we catch? High recall = fewer missed churners.

### F1 Score
`2 × (Precision × Recall) / (Precision + Recall)` — harmonic mean of precision and recall. Best single metric for imbalanced datasets because it penalizes models that sacrifice one for the other.

### ROC-AUC (Area Under the ROC Curve)
Measures how well the model ranks positive examples above negative ones across all classification thresholds. AUC=1.0 (perfect), AUC=0.5 (random). Threshold-independent — ideal for model comparison.

### Confusion Matrix
A 2×2 table showing: True Negatives (correct "stay"), False Positives (incorrectly flagged), False Negatives (missed churners), True Positives (correctly caught churners).

---

## Cross-Validation

### K-Fold Cross-Validation
Split data into K equally-sized folds. Train on K-1 folds, test on the remaining fold. Repeat K times, rotating the test fold. Average the results for a robust performance estimate.

### Stratified K-Fold
Same as K-fold but preserves the class ratio (e.g., 20% churn) in every fold. Essential for imbalanced datasets to avoid folds with very few positive examples.

---

## Feature Analysis

### Feature Importance
A score indicating how much each feature contributes to predictions. Tree-based models use **mean decrease in impurity** (Gini). Linear models use **absolute coefficient magnitude**. Higher score = more predictive.

### Correlation ≠ Causation
A feature with high importance is *associated* with churn but may not *cause* it. Example: customers with electronic payment may churn more, but switching their payment method won't necessarily make them stay. Validate with A/B tests.

---

## Deployment Concepts

### Model Persistence
Saving a trained model to disk (using joblib or pickle) so it can be loaded later without retraining. Important for production where training and inference happen in separate processes.

### Risk Scoring
Assigning a churn probability (0-1) to each customer. Typically bucketed into risk levels:
- **LOW** (< 0.3): No intervention needed
- **MEDIUM** (0.3-0.6): Proactive outreach
- **HIGH** (> 0.6): Immediate retention action

### Class Imbalance
When one class is much more common than the other (e.g., 80% stay vs 20% churn). Addressed through: stratified sampling, class weights, SMOTE oversampling, or choosing metrics like F1/AUC over accuracy.
