import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from src.churn_model import generate_synthetic_churn_data, get_feature_importance

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1c26 0%, #1a2a1a 50%, #1a1c26 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        border: 1px solid #2d4a2d;
    ">
        <h1 style="color:#ffffff;margin:0;font-size:2.2rem;">🔍 Feature Importance</h1>
        <p style="color:#a0aec0;font-size:1.05rem;margin-top:10px;">
            Discover which customer attributes drive churn — and understand <em>why</em>
            each feature matters for retention strategy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Educational expanders ─────────────────────────────────────────────────────
with st.expander("📖 What is Feature Importance?"):
    st.markdown(
        """
        **Feature importance** answers the question: *"Which inputs does the model actually rely on
        when making predictions?"*

        Think of it like asking a doctor: "Which symptoms most influenced your diagnosis?"
        The doctor might say fever and cough were critical, while eye colour was irrelevant.

        For a churn model, feature importance tells us:
        - Which customer attributes are **most predictive** of churn
        - Which features are **noise** (low importance, could be removed)
        - Where to **focus business interventions** — if tenure is most important,
          focus on early-life customer engagement programmes

        ### Why This Matters
        - **Model transparency** — understand *why* the model makes predictions
        - **Feature engineering** — focus on collecting the most impactful data
        - **Business decisions** — prioritise retention strategies around high-importance drivers
        - **Debugging** — unexpected high-importance features can signal data leakage
        """
    )

with st.expander("📖 How Do Different Models Compute Importance Differently?"):
    st.markdown(
        """
        ### Random Forest & Gradient Boosting: `feature_importances_`
        Tree-based models compute importance by measuring how much each feature **reduces
        impurity** (Gini impurity or entropy) across all split points in all trees.

        - A feature that consistently splits data cleanly → high importance
        - Averaged across all 100 trees → more stable than a single tree

        **Formula:** Σ (weighted impurity decrease for all splits on feature f) / total impurity decrease

        ### Logistic Regression: Absolute Coefficient Values
        Logistic Regression assigns a **weight (coefficient)** to each feature.
        Larger absolute weights = stronger influence on the predicted probability.

        We use `|coefficient|` (absolute value) because sign only indicates direction,
        not magnitude of influence.

        **Normalisation:** All importance scores are divided by their sum to get values in [0, 1].

        ### Important Caveat
        Feature importances can be misleading when features are **correlated**.
        If `tenure` and `total_charges` are correlated (they usually are),
        importance may be split between them, underestimating each one's true impact.
        """
    )

with st.expander("📖 How to Use Feature Importance in Business Decisions?"):
    st.markdown(
        """
        ### From Model to Action

        | Feature | High Importance Means | Business Action |
        |---------|----------------------|-----------------|
        | **Tenure** | New customers are highest risk | Invest in onboarding programs; 30/60/90-day check-ins |
        | **Monthly Charges** | Price sensitivity drives churn | Offer loyalty discounts; review pricing tiers |
        | **Contract Type** | Commitment level matters | Promote annual contracts with incentives |
        | **Internet Service** | Service quality affects retention | Prioritise Fiber upgrade programs |
        | **Payment Method** | Auto-pay = lower churn | Incentivise auto-payment sign-ups |
        | **Total Charges** | Cumulative spend signals loyalty | Recognise high-value customers with rewards |

        ### Actionable Framework
        1. **Top 3 features** → design targeted retention interventions
        2. **Bottom features** → consider removing from data collection (saves cost)
        3. **Unexpected top features** → investigate for data quality issues
        4. **Compare across models** → consistent importance = more trustworthy signal

        > 💡 Feature importance is descriptive, not causal. High-importance features are
        > *correlated* with churn but may not *cause* it. Validate with A/B tests before
        > making major strategy changes.
        """
    )

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Configuration")

col_ctrl1, col_ctrl2 = st.columns([2, 2])

with col_ctrl1:
    model_options = {
        "🌲 Random Forest": "random_forest",
        "🚀 Gradient Boosting": "gradient_boosting",
        "📐 Logistic Regression": "logistic_regression",
    }
    model_label = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        help="Different models may rank features differently.",
    )
    model_name = model_options[model_label]

with col_ctrl2:
    n_samples = st.slider(
        "Training samples",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
        help="Larger datasets produce more stable importance estimates.",
    )

if st.button("🔍 Compute Feature Importance", type="primary"):
    with st.spinner(f"Training {model_label} on {n_samples:,} samples..."):
        df = generate_synthetic_churn_data(n_samples)
        importance = get_feature_importance(df, model_name=model_name)
    st.session_state["importance"] = importance
    st.session_state["importance_model"] = model_label
    st.session_state["importance_df"] = df
    st.success(f"✅ Feature importance computed using {model_label}!")

# ── Results ───────────────────────────────────────────────────────────────────
if "importance" in st.session_state:
    importance = st.session_state["importance"]
    active_model = st.session_state["importance_model"]

    # Sort descending
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f[0] for f in sorted_features]
    feature_scores = [f[1] for f in sorted_features]

    st.markdown(f"## 📊 Feature Importance — {active_model}")

    # Bar chart
    chart_data = pd.DataFrame(
        {"Importance Score": feature_scores},
        index=feature_names,
    )
    st.bar_chart(chart_data)

    # Ranked table
    st.markdown("### 📋 Ranked Feature Table")
    ranked_df = pd.DataFrame(
        {
            "Rank": range(1, len(sorted_features) + 1),
            "Feature": feature_names,
            "Importance Score": [round(s, 4) for s in feature_scores],
            "% of Total": [f"{s / sum(feature_scores) * 100:.1f}%" for s in feature_scores],
        }
    )
    st.dataframe(ranked_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-feature interpretations ───────────────────────────────────────────
    st.markdown("## 🧠 Feature Interpretations")
    st.markdown("Understanding *why* each feature predicts churn — and what you can do about it.")

    feature_interpretations = {
        "tenure": {
            "icon": "⏳",
            "headline": "How long the customer has been with you",
            "why": "Customers with **shorter tenure** are significantly more likely to churn. They haven't yet experienced enough value to build loyalty. The first 3–6 months are the critical retention window — this is when most churn happens.",
            "action": "🎯 **Action:** Invest in structured onboarding, early success milestones, and proactive check-ins at 30/60/90 days.",
        },
        "monthly_charges": {
            "icon": "💰",
            "headline": "How much the customer pays each month",
            "why": "**Higher monthly charges** correlate with increased churn risk. Customers feeling they're not getting value-for-money are prime churn candidates. Price sensitivity is a major driver, especially in competitive markets.",
            "action": "🎯 **Action:** Offer loyalty discounts to high-charge, short-tenure customers. Review pricing tiers. Bundle services to increase perceived value.",
        },
        "total_charges": {
            "icon": "📈",
            "headline": "Cumulative lifetime spend",
            "why": "**Higher total charges** generally indicate longer tenure and **lower churn risk** — customers who've spent more have typically stayed longer and see more value. It's a proxy for customer lifetime value.",
            "action": "🎯 **Action:** Recognise and reward high-spend customers with VIP programmes. Protect your most valuable segments first.",
        },
        "contract_type": {
            "icon": "📝",
            "headline": "Month-to-month vs annual vs two-year contract",
            "why": "Contract type is one of the **strongest churn predictors**. Month-to-month customers can leave any time — they have no switching cost. Two-year contract customers are committed and rarely churn.",
            "action": "🎯 **Action:** Aggressively promote annual and two-year plans with upfront discounts. Make it easy to upgrade. Create annual plan landing pages.",
        },
        "internet_service": {
            "icon": "🌐",
            "headline": "Type of internet service (DSL / Fiber / None)",
            "why": "Service type correlates with customer profile. **Fiber customers** often pay more and have higher expectations — any service disruption or price increase hits harder. DSL customers may churn when fibre becomes available.",
            "action": "🎯 **Action:** Monitor satisfaction scores by service tier. Proactively reach out before Fiber price increases.",
        },
        "payment_method": {
            "icon": "💳",
            "headline": "How the customer pays (auto / check / electronic)",
            "why": "**Auto-pay customers** churn significantly less — automatic billing removes friction and reduces the number of times a customer actively thinks about their bill. Manual payment methods create regular decision points where customers may cancel.",
            "action": "🎯 **Action:** Offer incentives (discount, free month) for switching to auto-pay. Default new signups to auto-pay with opt-out.",
        },
        "age": {
            "icon": "🎂",
            "headline": "Customer age",
            "why": "Older customers tend to be **more loyal** and less likely to switch providers. Younger customers are more digitally native, more price-sensitive, and more willing to switch for better deals.",
            "action": "🎯 **Action:** Tailor retention messaging by age segment. Younger customers may respond to tech-forward features; older customers to reliability and service quality.",
        },
    }

    for rank, (feature, score) in enumerate(sorted_features[:6], 1):
        interp = feature_interpretations.get(feature)
        if interp is None:
            continue

        score_pct = score / sum(feature_scores) * 100
        with st.expander(f"#{rank} {interp['icon']} `{feature}` — {score_pct:.1f}% importance"):
            st.markdown(f"**{interp['headline']}**")
            st.markdown(f"📌 **Importance score:** `{score:.4f}` ({score_pct:.1f}% of total)")
            st.markdown(f"**Why it predicts churn:**  {interp['why']}")
            st.info(interp["action"])

    st.divider()

    # ── Model comparison note ─────────────────────────────────────────────────
    st.markdown("## 🔄 Compare Across Models")
    st.info(
        "💡 **Tip:** Change the model selector above and re-run to compare how different algorithms "
        "rank features. Features that rank highly across **all three models** are the most "
        "reliable and trustworthy predictors of churn."
    )

    st.warning(
        "⚠️ **Correlation ≠ Causation:** High-importance features are correlated with churn, "
        "but may not *cause* it. Before major strategy changes, validate with controlled A/B tests "
        "or natural experiments."
    )

else:
    st.info("👆 Configure the model above and click **Compute Feature Importance** to begin.")
