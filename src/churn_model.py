"""Churn prediction model."""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def generate_synthetic_churn_data(n_samples: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": range(n_samples), "age": rng.integers(18, 70, n_samples),
        "tenure": rng.integers(1, 72, n_samples), "monthly_charges": rng.normal(65, 30, n_samples),
        "total_charges": rng.normal(2000, 1500, n_samples), "contract_type": rng.choice(["month", "year", "two_year"], n_samples),
        "internet_service": rng.choice(["DSL", "Fiber", "No"], n_samples), "payment_method": rng.choice(["auto", "check", "electronic"], n_samples),
    })
    df["monthly_charges"] = df["monthly_charges"].clip(20)
    df["total_charges"] = df["total_charges"].clip(0)
    # Create label based on features
    churn_prob = (1 / (1 + np.exp(-(df["monthly_charges"] / 100 - 0.5 + (df["tenure"] < 12) * 1.5 - (df["contract_type"] == "two_year") * 2))))
    df["churn"] = (rng.random(n_samples) < churn_prob).astype(int)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "churn" and c != "customer_id"]
    df_proc = df.copy()
    for c in cat_cols:
        df_proc[c] = LabelEncoder().fit_transform(df_proc[c].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc[num_cols + cat_cols])
    y = df_proc["churn"].values
    return X, y, scaler


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {"accuracy": round(accuracy_score(y_test, y_pred), 4),
                   "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                   "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                   "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                   "roc_auc": round(roc_auc_score(y_test, y_proba), 4)}
        results[name] = metrics
        logger.info("%s: accuracy=%.4f, roc_auc=%.4f", name, metrics["accuracy"], metrics["roc_auc"])
    return results
