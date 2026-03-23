"""
Page 4: Confusion Matrix Visualization
=======================================

Visualizes confusion matrices for all three models, helping users understand
the tradeoff between false positives (wasted budget) and false negatives
(missed churners).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from src.churn_model import generate_synthetic_churn_data, get_confusion_matrices

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1c26 0%, #4a1a1a 50%, #1a1c26 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        border: 1px solid #5a2d2d;
    ">
        <h1 style="color:#ffffff;margin:0;font-size:2.2rem;">🎯 Confusion Matrix Analysis</h1>
        <p style="color:#a0aec0;font-size:1.05rem;margin-top:10px;">
            Understand <em>how</em> each model gets things wrong — and what that means
            for your retention budget and missed-churn risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Educational Section ───────────────────────────────────────────────────────
with st.expander("📖 What is a Confusion Matrix?"):
    st.markdown(
        """
        A **confusion matrix** is a 2×2 table that breaks down every prediction into four categories:

        |  | **Predicted: No Churn** | **Predicted: Churn** |
        |--|:--:|:--:|
        | **Actual: No Churn** | ✅ True Negative (TN) | ⚠️ False Positive (FP) |
        | **Actual: Churn** | ❌ False Negative (FN) | ✅ True Positive (TP) |

        ### What Each Quadrant Means for Business

        - **True Negatives (TN):** Customers correctly identified as staying → no action needed, no cost
        - **True Positives (TP):** Churners correctly caught → retention team can intervene
        - **False Positives (FP):** Loyal customers incorrectly flagged → wasted retention offers ($$)
        - **False Negatives (FN):** Churners we missed → lost revenue, the most expensive error

        ### The Business Tradeoff

        > Every model makes a tradeoff between **FP cost** (wasted budget) and **FN cost** (lost customers).

        - **Conservative models** (high threshold): Few FP, but more FN — saves budget, loses customers
        - **Aggressive models** (low threshold): Few FN, but more FP — catches more churners, costs more

        The right balance depends on your unit economics:
        - If a retention offer costs $50 but a churned customer = $500 lost revenue → aggressive is better
        - If retention offers are expensive → conservative may be smarter
        """
    )

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Configuration")

col1, col2 = st.columns(2)
with col1:
    n_samples = st.slider(
        "Training samples",
        min_value=200, max_value=5000, value=1000, step=100,
        help="More samples = more stable confusion matrix estimates.",
    )
with col2:
    test_size = st.slider(
        "Test set fraction",
        min_value=0.1, max_value=0.4, value=0.2, step=0.05,
        help="Fraction of data held out for evaluation.",
    )

if st.button("🎯 Compute Confusion Matrices", type="primary"):
    with st.spinner("Training models and computing confusion matrices..."):
        df = generate_synthetic_churn_data(n_samples)
        results = get_confusion_matrices(df, test_size=test_size)
    st.session_state["cm_results"] = results
    st.session_state["cm_n_test"] = int(n_samples * test_size)
    st.success("✅ Confusion matrices computed for all three models!")

# ── Results ───────────────────────────────────────────────────────────────────
if "cm_results" in st.session_state:
    results = st.session_state["cm_results"]
    n_test = st.session_state["cm_n_test"]

    model_display = {
        "logistic_regression": ("📐 Logistic Regression", "#4299e1"),
        "random_forest": ("🌲 Random Forest", "#48bb78"),
        "gradient_boosting": ("🚀 Gradient Boosting", "#ed8936"),
    }

    st.markdown("## 📊 Confusion Matrices")
    st.markdown(f"*Based on {n_test} test samples*")

    # Create three columns for side-by-side matrices
    cols = st.columns(3)

    for i, (model_key, (display_name, color)) in enumerate(model_display.items()):
        with cols[i]:
            st.markdown(
                f"<h4 style='color:{color};text-align:center;'>{display_name}</h4>",
                unsafe_allow_html=True,
            )

            r = results[model_key]
            tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]

            # Plotly annotated heatmap for the confusion matrix
            z = [[tn, fp], [fn, tp]]
            labels = [
                [f"TN\n{tn}", f"FP\n{fp}"],
                [f"FN\n{fn}", f"TP\n{tp}"],
            ]

            fig = ff.create_annotated_heatmap(
                z=z,
                annotation_text=labels,
                x=["Pred: Stay", "Pred: Churn"],
                y=["Actual: Stay", "Actual: Churn"],
                colorscale="Blues",
                showscale=False,
            )
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Derived rates
            total_positives = fn + tp
            total_negatives = tn + fp
            tpr = tp / total_positives if total_positives > 0 else 0
            fpr = fp / total_negatives if total_negatives > 0 else 0

            st.metric("Catch Rate (TPR)", f"{tpr:.1%}")
            st.metric("False Alarm Rate (FPR)", f"{fpr:.1%}")

    st.divider()

    # ── Business Impact Analysis ──────────────────────────────────────────────
    st.markdown("## 💰 Business Impact Simulator")
    st.markdown(
        "Estimate the financial impact of each model's confusion matrix "
        "using your own cost assumptions."
    )

    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        cost_per_fn = st.number_input(
            "💸 Cost per missed churner (FN)",
            min_value=0, max_value=10000, value=500, step=50,
            help="Revenue lost when a customer churns undetected.",
        )
    with cost_col2:
        cost_per_fp = st.number_input(
            "📤 Cost per false alarm (FP)",
            min_value=0, max_value=10000, value=50, step=10,
            help="Cost of sending a retention offer to a non-churner.",
        )

    st.markdown("### 📊 Estimated Costs by Model")

    cost_data = []
    for model_key, (display_name, color) in model_display.items():
        r = results[model_key]
        fn_cost = r["fn"] * cost_per_fn
        fp_cost = r["fp"] * cost_per_fp
        total_cost = fn_cost + fp_cost
        saved = r["tp"] * cost_per_fn  # revenue saved by catching churners

        cost_data.append({
            "Model": display_name,
            "Missed Churner Cost": f"${fn_cost:,.0f}",
            "False Alarm Cost": f"${fp_cost:,.0f}",
            "Total Error Cost": f"${total_cost:,.0f}",
            "Revenue Saved (caught churners)": f"${saved:,.0f}",
            "Net Impact": f"${saved - total_cost:,.0f}",
        })

    st.dataframe(
        pd.DataFrame(cost_data),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("💡 How to interpret the business impact"):
        st.markdown(
            """
            ### Reading the Cost Table

            - **Missed Churner Cost** = FN × cost per missed churner
              - This is revenue you *lose* because the model didn't flag these customers
            - **False Alarm Cost** = FP × cost per retention offer
              - This is budget *wasted* on customers who weren't going to churn
            - **Total Error Cost** = sum of both error costs
            - **Revenue Saved** = TP × cost per missed churner
              - Revenue *protected* by successfully identifying at-risk customers
            - **Net Impact** = Revenue Saved − Total Error Cost
              - Positive = model is generating net value; negative = errors outweigh catches

            ### Key Insight
            The "best" model depends on your **cost ratio**. If missing a churner costs
            10× more than a false alarm (common in telecom), prioritize **high recall**
            models even if they have more false positives.
            """
        )

else:
    st.info("👆 Click **Compute Confusion Matrices** to analyze model predictions.")
