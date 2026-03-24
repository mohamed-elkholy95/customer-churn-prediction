import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📉")
st.markdown(
    '<style>'
    '[data-testid="stSidebar"]{background-color:#1a1c26}'
    '.stApp{background-color:#0e1117;color:#fff}'
    'h1,h2,h3{color:#e0e0e0}'
    '.stMetric{background-color:#1a1c26;border-radius:8px;padding:12px}'
    '</style>',
    unsafe_allow_html=True,
)
pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_📈_Predict.py", title="Train & Evaluate", icon="📈"),
    st.Page("pages/3_🔍_Feature_Importance.py", title="Feature Importance", icon="🔍"),
    st.Page("pages/4_🎯_Confusion_Matrix.py", title="Confusion Matrix", icon="🎯"),
    st.Page("pages/5_⚙️_Threshold_Tuning.py", title="Threshold & Learning", icon="⚙️"),
])
pg.run()
