"""
Churn Prediction Model
======================

This module implements an end-to-end customer churn prediction pipeline:

1. **Data Generation** — Creates realistic synthetic customer datasets with
   configurable size and reproducible randomness.
2. **Preprocessing** — Transforms raw features (label encoding + standardization)
   into model-ready numerical arrays.
3. **Training & Evaluation** — Trains three classifiers (Logistic Regression,
   Random Forest, Gradient Boosting) and evaluates on held-out test data.
4. **Feature Importance** — Extracts and normalizes feature importance scores
   to understand which attributes drive churn predictions.
5. **Cross-Validation** — Runs stratified k-fold CV for robust performance
   estimates that account for class imbalance.
6. **Model Persistence** — Save and load trained model artifacts (model, scaler,
   feature names) for deployment or later analysis.

Key Design Decisions:
    - Synthetic data uses a logistic function for label generation, ensuring
      the relationship between features and churn is learnable but stochastic.
    - StandardScaler is applied to all features (numeric + encoded categorical)
      to ensure equal contribution regardless of original scale.
    - Stratified splits preserve class distribution in both train/test and CV folds,
      which is critical for imbalanced churn datasets (typically 20-30% positive).

Example:
    >>> from src.churn_model import generate_synthetic_churn_data, train_and_evaluate
    >>> df = generate_synthetic_churn_data(n_samples=1000)
    >>> results = train_and_evaluate(df)
    >>> print(results["random_forest"]["roc_auc"])
    0.8523
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The three model architectures supported by this pipeline.
# Each represents a different ML paradigm:
#   - logistic_regression: linear model (fast, interpretable)
#   - random_forest: bagging ensemble (robust, parallelizable)
#   - gradient_boosting: boosting ensemble (accurate, sequential)
SUPPORTED_MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]

# Expected columns in generated/input data
EXPECTED_COLUMNS = {
    "customer_id", "age", "tenure", "monthly_charges", "total_charges",
    "contract_type", "internet_service", "payment_method", "churn",
}


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame has the expected schema for churn prediction.

    Checks for:
        - Non-empty DataFrame
        - All required columns present
        - Churn column contains only binary values (0 and 1)
        - No completely null columns

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any validation check fails, with a descriptive message
            explaining what's wrong and how to fix it.
    """
    if df.empty:
        raise ValueError(
            "DataFrame is empty. Provide at least 1 row of customer data. "
            "Use generate_synthetic_churn_data() to create sample data."
        )

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(EXPECTED_COLUMNS)}"
        )

    # Churn must be binary (0 or 1) — the target variable for classification
    unique_churn = set(df["churn"].dropna().unique())
    if not unique_churn.issubset({0, 1}):
        raise ValueError(
            f"Column 'churn' must contain only 0 and 1, got: {unique_churn}. "
            "Encode churn as binary before training."
        )

    # Check for columns that are entirely null (likely data loading issues)
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        raise ValueError(
            f"Columns with all null values: {null_cols}. "
            "Check your data source for loading errors."
        )


def generate_synthetic_churn_data(
    n_samples: int = 1000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate a synthetic customer dataset with realistic churn patterns.

    Creates customer records with demographics, service attributes, and billing
    information. The churn label is generated using a logistic function that
    combines multiple features, producing realistic but stochastic labels.

    The logistic function for churn probability:
        P(churn) = sigmoid(monthly_charges/100 - 0.5
                          + 1.5 * (tenure < 12)
                          - 2.0 * (contract == 'two_year'))

    This means:
        - Short-tenure customers (< 12 months) have ~4.5x higher churn odds
        - Two-year contract customers have dramatically lower churn odds
        - Higher monthly charges increase churn probability

    Args:
        n_samples: Number of customer records to generate. Must be >= 1.
            Recommended: 500+ for meaningful model training.
        seed: Random seed for reproducibility. Same seed = same dataset.

    Returns:
        DataFrame with columns: customer_id, age, tenure, monthly_charges,
        total_charges, contract_type, internet_service, payment_method, churn.

    Raises:
        ValueError: If n_samples < 1.

    Example:
        >>> df = generate_synthetic_churn_data(500, seed=42)
        >>> df.shape
        (500, 9)
        >>> df["churn"].value_counts()
        0    312
        1    188
        Name: churn, dtype: int64
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": range(n_samples),
        "age": rng.integers(18, 70, n_samples),
        "tenure": rng.integers(1, 72, n_samples),
        "monthly_charges": rng.normal(65, 30, n_samples),
        "total_charges": rng.normal(2000, 1500, n_samples),
        "contract_type": rng.choice(
            ["month", "year", "two_year"], n_samples
        ),
        "internet_service": rng.choice(["DSL", "Fiber", "No"], n_samples),
        "payment_method": rng.choice(
            ["auto", "check", "electronic"], n_samples
        ),
    })

    # Clip to realistic ranges — no negative charges or unreasonably low bills
    df["monthly_charges"] = df["monthly_charges"].clip(lower=20)
    df["total_charges"] = df["total_charges"].clip(lower=0)

    # Generate churn labels using a logistic (sigmoid) function.
    # The sigmoid maps any real number to a probability in (0, 1):
    #   sigmoid(x) = 1 / (1 + exp(-x))
    #
    # We combine multiple risk factors into a single "risk score":
    #   - monthly_charges/100: higher charges = higher risk (normalized)
    #   - tenure < 12: new customers are high-risk (+1.5 boost)
    #   - two_year contract: long commitment = low risk (-2.0 penalty)
    churn_prob = 1 / (
        1 + np.exp(
            -(
                df["monthly_charges"] / 100
                - 0.5
                + (df["tenure"] < 12).astype(float) * 1.5
                - (df["contract_type"] == "two_year").astype(float) * 2
            )
        )
    )

    # Stochastic assignment: each customer churns with their computed probability
    # This introduces realistic noise — not all high-risk customers actually churn
    df["churn"] = (rng.random(n_samples) < churn_prob).astype(int)

    logger.info(
        "Generated %d samples (churn rate: %.1f%%)",
        n_samples,
        df["churn"].mean() * 100,
    )

    return df


def preprocess(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Transform raw customer data into model-ready feature arrays.

    Applies two transformations:
        1. **Label Encoding** — Converts categorical columns (contract_type,
           internet_service, payment_method) to integer codes.
        2. **Standard Scaling** — Normalizes all features to mean=0, std=1.
           This prevents high-magnitude features (e.g., total_charges ~ 2000)
           from dominating low-magnitude ones (e.g., age ~ 40).

    Feature ordering: numeric columns first, then categorical columns.
    This ordering is consistent with _get_feature_names().

    Args:
        df: DataFrame with expected churn dataset columns.

    Returns:
        Tuple of (X, y, scaler) where:
            - X: np.ndarray of shape (n_samples, n_features), scaled features
            - y: np.ndarray of shape (n_samples,), binary churn labels
            - scaler: fitted StandardScaler instance (needed for new predictions)

    Raises:
        ValueError: If DataFrame validation fails.
    """
    validate_dataframe(df)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("churn", "customer_id")
    ]

    df_proc = df.copy()

    # Label encode each categorical column independently.
    # Note: LabelEncoder is fit per-column, which is fine for training.
    # In production, you'd store the encoders for consistent encoding of new data.
    for c in cat_cols:
        df_proc[c] = LabelEncoder().fit_transform(df_proc[c].astype(str))

    # StandardScaler: z = (x - mean) / std for each feature
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc[num_cols + cat_cols])
    y = df_proc["churn"].values

    return X, y, scaler


def _get_feature_names(df: pd.DataFrame) -> List[str]:
    """Return ordered feature names matching the preprocess() output columns.

    The order is: numeric columns first, then categorical columns.
    This must stay in sync with preprocess() to ensure correct feature-importance
    mapping.

    Args:
        df: DataFrame with the expected column schema.

    Returns:
        List of feature names in the same order as preprocess() output columns.
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("churn", "customer_id")
    ]
    return num_cols + cat_cols


def _build_model(model_name: str) -> Any:
    """Instantiate a scikit-learn classifier by name.

    All models use RANDOM_SEED for reproducibility. Hyperparameters are set
    to reasonable defaults for a medium-sized tabular dataset.

    Args:
        model_name: One of 'logistic_regression', 'random_forest',
            or 'gradient_boosting'.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        )
    else:
        raise ValueError(
            f"Unknown model_name: {model_name!r}. "
            f"Choose from: {SUPPORTED_MODELS}"
        )


def get_feature_importance(
    df: pd.DataFrame, model_name: str = "random_forest"
) -> Dict[str, float]:
    """Train a model on the full dataset and return normalized feature importance.

    Different model types compute importance differently:
        - **Tree-based models** (random_forest, gradient_boosting): Uses
          `feature_importances_` — measures mean decrease in impurity (Gini)
          across all tree splits for each feature.
        - **Logistic Regression**: Uses absolute coefficient values |w|,
          normalized to sum to 1. Larger |w| = stronger influence on prediction.

    Args:
        df: DataFrame with churn dataset columns.
        model_name: Model to use for importance calculation.

    Returns:
        Dict mapping feature_name -> importance_score.
        All values >= 0 and sum approximately to 1.0.

    Example:
        >>> importance = get_feature_importance(df, "random_forest")
        >>> sorted(importance.items(), key=lambda x: -x[1])[:3]
        [('monthly_charges', 0.31), ('tenure', 0.28), ('total_charges', 0.18)]
    """
    feature_names = _get_feature_names(df)
    X, y, _ = preprocess(df)
    model = _build_model(model_name)
    model.fit(X, y)

    if model_name == "logistic_regression":
        # For linear models, coefficient magnitude indicates feature influence.
        # We take absolute value because sign only indicates direction (positive
        # = increases churn probability, negative = decreases it).
        raw = np.abs(model.coef_[0])
    else:
        # Tree-based models provide feature_importances_ directly —
        # the mean decrease in Gini impurity weighted by sample count.
        raw = model.feature_importances_

    # Normalize to [0, 1] so scores are comparable across models
    total = raw.sum()
    scores = raw / total if total > 0 else raw

    return dict(zip(feature_names, scores.tolist()))


def cross_validate_models(
    df: pd.DataFrame, n_splits: int = 5
) -> Dict[str, Dict[str, float]]:
    """Run stratified k-fold cross-validation for all three model types.

    Cross-validation provides a more robust performance estimate than a single
    train/test split by training and evaluating on k different data partitions.

    Stratified splitting ensures each fold preserves the original churn class
    ratio, which is important for imbalanced datasets.

    Args:
        df: DataFrame with churn dataset columns.
        n_splits: Number of CV folds. Higher = more robust but slower.
            Common values: 3 (fast), 5 (balanced), 10 (thorough).

    Returns:
        Dict of model_name -> metrics dict with keys:
            mean_accuracy, std_accuracy, mean_f1, std_f1

    Example:
        >>> results = cross_validate_models(df, n_splits=5)
        >>> results["gradient_boosting"]["mean_f1"]
        0.7234
    """
    X, y, _ = preprocess(df)
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    results: Dict[str, Dict[str, float]] = {}

    for name in SUPPORTED_MODELS:
        model = _build_model(name)

        # cross_val_score handles the train/test splitting internally
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        results[name] = {
            "mean_accuracy": round(float(acc_scores.mean()), 4),
            "std_accuracy": round(float(acc_scores.std()), 4),
            "mean_f1": round(float(f1_scores.mean()), 4),
            "std_f1": round(float(f1_scores.std()), 4),
        }

        logger.info(
            "%s CV: accuracy=%.4f±%.4f, f1=%.4f±%.4f",
            name,
            results[name]["mean_accuracy"],
            results[name]["std_accuracy"],
            results[name]["mean_f1"],
            results[name]["std_f1"],
        )

    return results


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = 0.2
) -> Dict[str, Any]:
    """Train all three models and evaluate on a held-out test set.

    Uses stratified train/test split to preserve class distribution.
    Computes five standard classification metrics per model.

    Args:
        df: DataFrame with churn dataset columns.
        test_size: Fraction of data reserved for testing (0.0 to 1.0).
            Default 0.2 means 80% train, 20% test.

    Returns:
        Dict of model_name -> metrics dict with keys:
            accuracy, precision, recall, f1, roc_auc

    Raises:
        ValueError: If test_size is not in (0.0, 1.0) or data is invalid.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be between 0 and 1 (exclusive), got {test_size}"
        )

    X, y, _ = preprocess(df)

    # Stratified split ensures the churn ratio is preserved in both sets.
    # Without stratification, random splits could create a test set with
    # very few churners, making evaluation unreliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # predict_proba returns [P(class=0), P(class=1)] for each sample.
        # We take [:, 1] for the churn probability, which ROC-AUC needs.
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(
                precision_score(y_test, y_pred, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, y_pred, zero_division=0), 4
            ),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        results[name] = metrics

        logger.info(
            "%s: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, roc_auc=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

    return results


def get_confusion_matrices(
    df: pd.DataFrame, test_size: float = 0.2
) -> Dict[str, Dict[str, Any]]:
    """Train models and return confusion matrices for each.

    A confusion matrix shows the breakdown of predictions:
        [[True Negatives,  False Positives],
         [False Negatives, True Positives]]

    This is essential for understanding model behavior beyond aggregate metrics:
        - High FP (false positives) = wasting retention budget on non-churners
        - High FN (false negatives) = missing actual churners (costly!)

    Args:
        df: DataFrame with churn dataset columns.
        test_size: Fraction of data for testing.

    Returns:
        Dict of model_name -> {
            "matrix": [[TN, FP], [FN, TP]],
            "labels": ["Not Churned", "Churned"],
            "tn": int, "fp": int, "fn": int, "tp": int,
        }
    """
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    results = {}
    for name in SUPPORTED_MODELS:
        model = _build_model(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "matrix": cm.tolist(),
            "labels": ["Not Churned", "Churned"],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        logger.info(
            "%s confusion matrix: TN=%d, FP=%d, FN=%d, TP=%d",
            name, tn, fp, fn, tp,
        )

    return results


def predict_single_customer(
    customer: Dict[str, Any],
    df: pd.DataFrame,
    model_name: str = "gradient_boosting",
) -> Dict[str, Any]:
    """Predict churn probability for a single customer.

    Trains the specified model on the provided dataset, then predicts
    the churn probability and risk level for one customer.

    Risk levels:
        - P(churn) < 0.3  → LOW risk (green)
        - P(churn) 0.3-0.6 → MEDIUM risk (yellow)
        - P(churn) > 0.6  → HIGH risk (red)

    Args:
        customer: Dict with keys matching feature columns (age, tenure,
            monthly_charges, total_charges, contract_type, internet_service,
            payment_method).
        df: Training dataset to fit the model on.
        model_name: Which model to use for prediction.

    Returns:
        Dict with keys:
            - churn_probability: float in [0, 1]
            - risk_level: "LOW", "MEDIUM", or "HIGH"
            - risk_emoji: colored emoji indicator
            - model_used: name of the model

    Example:
        >>> result = predict_single_customer(
        ...     {"age": 25, "tenure": 3, "monthly_charges": 90,
        ...      "total_charges": 270, "contract_type": "month",
        ...      "internet_service": "Fiber", "payment_method": "electronic"},
        ...     df, "gradient_boosting"
        ... )
        >>> result["risk_level"]
        'HIGH'
    """
    # Build a single-row DataFrame matching the training schema
    customer_df = pd.DataFrame([{
        "customer_id": 99999,
        "churn": 0,  # placeholder — won't be used for prediction
        **customer,
    }])

    # Combine with training data so LabelEncoder sees all categories
    combined = pd.concat([df, customer_df], ignore_index=True)

    X_all, _, scaler = preprocess(combined)
    X_train = X_all[:-1]
    X_customer = X_all[-1:]
    y_train = combined["churn"].values[:-1]

    model = _build_model(model_name)
    model.fit(X_train, y_train)

    proba = float(model.predict_proba(X_customer)[0, 1])

    if proba < 0.3:
        risk_level, risk_emoji = "LOW", "🟢"
    elif proba < 0.6:
        risk_level, risk_emoji = "MEDIUM", "🟡"
    else:
        risk_level, risk_emoji = "HIGH", "🔴"

    return {
        "churn_probability": round(proba, 4),
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "model_used": model_name,
    }


def save_model(
    df: pd.DataFrame,
    model_name: str = "gradient_boosting",
    output_dir: str = "models",
) -> Path:
    """Train a model and save it to disk with its preprocessing artifacts.

    Saves three files:
        - {model_name}_model.joblib — the trained sklearn estimator
        - {model_name}_scaler.joblib — the fitted StandardScaler
        - {model_name}_metadata.json — feature names, training info

    Args:
        df: Training DataFrame.
        model_name: Which model to train and save.
        output_dir: Directory to save model files (created if needed).

    Returns:
        Path to the output directory containing saved files.

    Example:
        >>> save_model(df, "random_forest", "models/")
        PosixPath('models')
    """
    import joblib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = _get_feature_names(df)
    X, y, scaler = preprocess(df)

    model = _build_model(model_name)
    model.fit(X, y)

    # Save model and scaler as separate files for flexibility
    joblib.dump(model, output_path / f"{model_name}_model.joblib")
    joblib.dump(scaler, output_path / f"{model_name}_scaler.joblib")

    # Metadata for reproducibility and deployment documentation
    metadata = {
        "model_name": model_name,
        "feature_names": feature_names,
        "n_training_samples": len(df),
        "churn_rate": round(float(df["churn"].mean()), 4),
        "random_seed": RANDOM_SEED,
    }
    with open(output_path / f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved %s model to %s (%d training samples)",
        model_name, output_path, len(df),
    )

    return output_path


def load_model(
    model_name: str = "gradient_boosting",
    model_dir: str = "models",
) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """Load a previously saved model, scaler, and metadata from disk.

    Args:
        model_name: Name of the model to load.
        model_dir: Directory containing saved model files.

    Returns:
        Tuple of (model, scaler, metadata) where:
            - model: fitted sklearn estimator
            - scaler: fitted StandardScaler
            - metadata: dict with training info and feature names

    Raises:
        FileNotFoundError: If model files don't exist in model_dir.
    """
    import joblib

    model_path = Path(model_dir)

    model_file = model_path / f"{model_name}_model.joblib"
    scaler_file = model_path / f"{model_name}_scaler.joblib"
    metadata_file = model_path / f"{model_name}_metadata.json"

    for f in [model_file, scaler_file, metadata_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"Model file not found: {f}. "
                f"Run save_model() first to train and save the model."
            )

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    with open(metadata_file) as f:
        metadata = json.load(f)

    logger.info("Loaded %s model from %s", model_name, model_path)

    return model, scaler, metadata
