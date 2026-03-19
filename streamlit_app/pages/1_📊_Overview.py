import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
st.title("📉 Customer Churn Prediction")
st.markdown("Predict customer churn using logistic regression and random forest models.")
col1, col2 = st.columns(2)
with col1: st.subheader("Models"); st.markdown("- Logistic Regression\n- Random Forest Classifier")
with col2: st.subheader("Metrics"); st.markdown("- Accuracy, Precision, Recall, F1\n- ROC-AUC Score")
