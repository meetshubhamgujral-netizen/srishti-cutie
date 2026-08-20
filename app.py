"""
ChurnLab — Credit Card Attrition Intelligence
=============================================

A six-tab Streamlit dashboard covering the full analytics lifecycle for credit
card churn: exploratory analysis, a five-model comparison with rigorous
cross-validation, threshold and A/B-test economics, unsupervised segmentation,
NLP/LLM text intelligence, and a live scoring form.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then point Streamlit Community Cloud at app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ChurnLab — Credit Card Attrition Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data import TARGET, load_data  # noqa: E402
from src.models import train_all  # noqa: E402
from src.tabs import evaluation, model_lab, nlp_tab, overview, predictor, segmentation  # noqa: E402
from src.theme import AZURE, CHURN, GOLD, MUTED, RETAIN, VIOLET, apply_theme, callout  # noqa: E402

apply_theme()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 💳 ChurnLab")
    st.caption("Credit card attrition intelligence")
    st.divider()

    st.markdown("**Data source**")
    uploaded = st.file_uploader(
        "Use your own CSV", type=["csv"],
        help="Must contain the same columns as the bundled dataset. Leave empty to use it.",
    )
    st.caption("Bundled: `data/customer_churn_data.csv`" if uploaded is None else f"Using: `{uploaded.name}`")

    st.divider()
    st.markdown("**Validation settings**")
    test_size = st.slider("Held-out test share", 0.15, 0.40, 0.25, 0.05,
                          help="Rows quarantined from training entirely, used once for final scoring.")
    n_splits = st.slider("Cross-validation folds", 3, 10, 5, 1)
    n_repeats = st.slider("CV repeats", 1, 5, 3, 1,
                          help="Repeating the whole k-fold with different shuffles. On 500 rows this is the difference between a stable estimate and a lucky split.")
    st.caption(f"→ {n_splits * n_repeats} model fits per algorithm, 5 algorithms = "
               f"**{n_splits * n_repeats * 5} fits** per training run.")

    st.divider()
    if st.button("♻️ Retrain from scratch", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:0.78rem;color:{MUTED};line-height:1.6'>"
        "Built with scikit-learn, Plotly and Streamlit.<br>"
        "The NLP tab uses a <b>synthetic</b> verbatim corpus — the source CSV has no text column. "
        "That is flagged in the tab itself.<br><br>"
        "LLM features are optional and need an Anthropic API key."
        "</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ data
try:
    raw, df, quality = load_data(uploaded.getvalue() if uploaded else None)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

missing = [c for c in [TARGET, "CustomerID"] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

# ------------------------------------------------------------------ masthead
churn_rate = df[TARGET].mean()
st.markdown(
    f"""
    <div class="masthead">
        <div class="eyebrow">Credit card portfolio · attrition intelligence</div>
        <h1>ChurnLab</h1>
        <p>
            {len(df):,} card holders, {churn_rate:.1%} of them gone. This dashboard finds the ones who are
            leaving before they close the account — five competing models, cross-validated rather than
            cherry-picked, evaluated on the metrics that survive class imbalance, and priced against the
            actual cost of a retention offer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ training
cache_key = f"{len(df)}-{test_size}-{n_splits}-{n_repeats}-{int(df[TARGET].sum())}"
with st.spinner(f"Training 5 models across {n_splits * n_repeats} folds each — cached after the first run…"):
    try:
        run = train_all(df, test_size=test_size, n_splits=n_splits, n_repeats=n_repeats, _cache_key=cache_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")
        st.stop()

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Overview",
    "🤖  Model Lab",
    "📈  Evaluation",
    "🧩  Segmentation",
    "🧠  NLP & LLM",
    "🔮  Predict",
])

with tab1:
    overview.render(raw, df, quality)
with tab2:
    model_lab.render(run)
with tab3:
    evaluation.render(run)
with tab4:
    segmentation.render(df)
with tab5:
    nlp_tab.render(df)
with tab6:
    predictor.render(run, df)

st.divider()
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:0.8rem;padding:10px 0'>"
    f"ChurnLab · {len(df):,} customers · {len(run.models)} models · "
    f"{run.config['n_splits'] * run.config['n_repeats']} folds per model · "
    f"leading model: <b style='color:{GOLD}'>{run.best_name}</b> "
    f"(CV ROC-AUC {run.leaderboard.iloc[0]['CV ROC-AUC']:.3f})"
    f"</div>",
    unsafe_allow_html=True,
)
