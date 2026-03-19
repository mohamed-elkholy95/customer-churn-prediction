import pytest
import pandas as pd
from src.churn_model import generate_synthetic_churn_data, preprocess, train_and_evaluate

class TestGenerate:
    def test_shape(self):
        df = generate_synthetic_churn_data(100)
        assert df.shape == (100, 9)
    def test_has_churn(self):
        assert "churn" in generate_synthetic_churn_data().columns
    def test_reproducible(self):
        assert generate_synthetic_churn_data(seed=42).equals(generate_synthetic_churn_data(seed=42))

class TestPreprocess:
    def test_shapes(self):
        df = generate_synthetic_churn_data(200)
        X, y, _ = preprocess(df)
        assert X.shape[0] == 200

class TestTrainEvaluate:
    def test_returns_results(self):
        df = generate_synthetic_churn_data(200)
        results = train_and_evaluate(df)
        assert "logistic_regression" in results
        assert "roc_auc" in results["logistic_regression"]
