import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from src.churn_model import (
    generate_synthetic_churn_data,
    find_optimal_threshold,
    compute_learning_curve,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1c26 0%, #4a1942 50%, #1a1c26 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        border: 1px solid #5a2d52;
    ">
        <h1 style="color:#ffffff;margin:0;font-size:2.2rem;">⚙️ Threshold Tuning & Learning Curves</h1>
        <p style="color:#a0aec0;font-size:1.05rem;margin-top:10px;">
            Find the optimal decision boundary and understand how performance scales with data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Settings")
n_samples = st.sidebar.slider("Training samples", 200, 5000, 1000, 100)
model_name = st.sidebar.selectbox(
    "Model",
    ["gradient_boosting", "random_forest", "logistic_regression"],
    format_func=lambda x: {
        "gradient_boosting": "🚀 Gradient Boosting",
        "random_forest": "🌲 Random Forest",
        "logistic_regression": "📐 Logistic Regression",
    }[x],
)

# ── Generate data ─────────────────────────────────────────────────────────────
df = generate_synthetic_churn_data(n_samples)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Threshold Optimization", "📈 Learning Curve"])

# ── Tab 1: Threshold Optimization ────────────────────────────────────────────
with tab1:
    st.markdown("## 🎯 Optimal Classification Threshold")

    with st.expander("📖 Why does the threshold matter?"):
        st.markdown(
            """
            ### The Decision Boundary

            When a model predicts churn, it outputs a **probability** (0.0 to 1.0).
            The threshold determines when we act on that prediction:

            - **P(churn) >= threshold** → predict **churn** (flag for retention)
            - **P(churn) < threshold** → predict **not churn** (do nothing)

            ### Default = 0.5 is rarely optimal

            The default threshold of 0.5 assumes equal costs for both error types.
            In reality:

            | Error Type | Business Impact |
            |-----------|----------------|
            | **False Negative** (missed churner) | Lost customer = lost revenue ($500+/year) |
            | **False Positive** (false alarm) | Unnecessary retention offer ($20-50) |

            Since missing a churner is ~10× more expensive than a false alarm,
            **lowering the threshold** (e.g., 0.3) often makes business sense:
            more false alarms, but far fewer missed churners.
            """
        )

    metric = st.selectbox(
        "Metric to optimize",
        ["f1", "recall", "precision", "accuracy"],
        format_func=lambda x: {
            "f1": "F1 Score (balanced precision & recall)",
            "recall": "Recall (catch more churners)",
            "precision": "Precision (fewer false alarms)",
            "accuracy": "Accuracy (overall correctness)",
        }[x],
    )

    if st.button("🔍 Find Optimal Threshold", type="primary"):
        with st.spinner("Sweeping thresholds from 0.05 to 0.95..."):
            result = find_optimal_threshold(
                df, model_name=model_name, metric=metric
            )

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal Threshold", f"{result['optimal_threshold']:.2f}")
        with col2:
            st.metric(f"Best {metric.upper()}", f"{result['best_score']:.4f}")
        with col3:
            delta = result["optimal_threshold"] - 0.5
            direction = "lower" if delta < 0 else "higher"
            st.metric(
                "vs Default (0.50)",
                f"{abs(delta):.2f} {direction}",
                delta=f"{delta:+.2f}",
            )

        # Plot threshold curve
        st.markdown("### Threshold vs Score Curve")
        curve_df = pd.DataFrame(
            result["threshold_scores"],
            columns=["Threshold", metric.capitalize()],
        )
        st.line_chart(curve_df.set_index("Threshold"))

        # Highlight optimal point
        st.success(
            f"**Optimal threshold: {result['optimal_threshold']:.2f}** — "
            f"achieves {metric.upper()} of **{result['best_score']:.4f}** "
            f"(vs. ~{result['threshold_scores'][45][1]:.4f} at default 0.50)"
        )

# ── Tab 2: Learning Curve ────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📈 Learning Curve")

    with st.expander("📖 How to read a learning curve"):
        st.markdown(
            """
            ### What Learning Curves Tell You

            A learning curve shows model performance as training data increases:

            | Pattern | Diagnosis | Action |
            |---------|-----------|--------|
            | Both scores low | **Underfitting** | Use a more complex model |
            | Train high, test low | **Overfitting** | Get more data or regularize |
            | Both converge high | **Good fit** | You have enough data |
            | Test still rising | **More data helps** | Collect more samples |

            The **gap** between train and test scores indicates generalization:
            - Small gap = model generalizes well
            - Large gap = model memorizes training data
            """
        )

    n_points = st.slider("Number of evaluation points", 4, 15, 8)

    if st.button("📊 Compute Learning Curve", type="primary"):
        with st.spinner("Training models at different data sizes..."):
            curve = compute_learning_curve(
                df, model_name=model_name, n_points=n_points
            )

        # Plot learning curve
        curve_df = pd.DataFrame({
            "Training Samples": curve["train_sizes"],
            "Train F1": curve["train_scores"],
            "Test F1": curve["test_scores"],
        }).set_index("Training Samples")

        st.line_chart(curve_df)

        # Summary
        gap = curve["train_scores"][-1] - curve["test_scores"][-1]
        final_test = curve["test_scores"][-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Train F1", f"{curve['train_scores'][-1]:.4f}")
        with col2:
            st.metric("Final Test F1", f"{final_test:.4f}")
        with col3:
            st.metric("Generalization Gap", f"{gap:.4f}")

        # Interpretation
        if gap > 0.15:
            st.warning(
                "⚠️ **Large generalization gap** — the model may be overfitting. "
                "Consider collecting more data or using regularization."
            )
        elif final_test < 0.5:
            st.warning(
                "⚠️ **Low test performance** — the model may be underfitting. "
                "Consider using a more complex model or engineering better features."
            )
        else:
            test_improvement = curve["test_scores"][-1] - curve["test_scores"][0]
            if test_improvement > 0.05:
                st.info(
                    "📈 **Performance is still improving** with more data. "
                    "Collecting additional samples would likely boost accuracy."
                )
            else:
                st.success(
                    "✅ **Performance has converged** — the model has enough data "
                    "to learn the underlying patterns."
                )
