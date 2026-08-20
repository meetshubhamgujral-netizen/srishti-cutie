"""Tab 6 — Live churn predictor. Fill the form, get a verdict and a retention play."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..data import FEATURE_LABELS, build_single_customer, field_ranges
from ..models import TrainingRun
from ..nlp import (ANTHROPIC_MODELS, build_retention_prompt, call_claude, churn_intent,
                   get_api_key, RETENTION_SYSTEM, score_sentiment)
from ..theme import AZURE, CHURN, GOLD, LIME, MUTED, RETAIN, VIOLET, callout, kpi

INCOME_BANDS = {
    "₹3–6 L (entry)": 450_000,
    "₹6–10 L (mid)": 800_000,
    "₹10–18 L (senior)": 1_400_000,
    "₹18–30 L (affluent)": 2_400_000,
    "₹30 L+ (HNI)": 3_500_000,
}


def _fmt_prob(prob: float) -> str:
    """Display probability without ever claiming certainty.

    A model fitted on 500 rows has no business printing 0.0% or 100.0%. Extremes
    are shown as bounded statements so the card reads as a strong signal rather
    than a guarantee.
    """
    if prob >= 0.995:
        return "&gt;99%"
    if prob <= 0.005:
        return "&lt;1%"
    return f"{prob:.1%}"


def _verdict_card(prob: float, threshold: float, card_type: str, customer_ref: str) -> str:
    churning = prob >= threshold
    cls = "leave" if churning else "stay"
    verdict = "LIKELY TO CHURN" if churning else "LIKELY TO STAY"
    foot = (f"Above the {threshold:.0%} action threshold — route to retention today."
            if churning else
            f"Below the {threshold:.0%} action threshold — no intervention needed this cycle.")
    return f"""
    <div class="verdict-card {cls}">
        <div class="tier">{card_type} card</div>
        <div class="chip"></div>
        <div class="verdict">{verdict}</div>
        <div style="font-size:2.6rem;font-weight:700;font-family:'Space Grotesk',sans-serif;line-height:1.1">
            {_fmt_prob(prob)}
        </div>
        <div style="font-size:0.85rem;opacity:0.9">probability of attrition next quarter</div>
        <div class="pan">{customer_ref}</div>
        <div class="foot">{foot}</div>
    </div>
    """


def render(run: TrainingRun, df: pd.DataFrame) -> None:
    ranges = field_ranges(df)

    st.markdown("### Score a customer")
    st.caption(
        "Enter a customer's details and every trained model scores them. The engineered ratios "
        "(spend-to-limit, engagement, friction, inactivity) are computed automatically from these inputs — "
        "exactly as they were during training, so nothing drifts between fitting and serving."
    )

    model_name = st.selectbox(
        "Scoring model", list(run.models.keys()),
        index=list(run.models.keys()).index(run.best_name),
        help="Defaults to the leaderboard winner by cross-validated ROC-AUC.",
    )
    model = run.models[model_name]

    # ------------------------------------------------------------- form
    with st.form("churn_form"):
        st.markdown("#### Who is the customer?")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.slider("Age", int(ranges["Age"]["min"]), int(ranges["Age"]["max"]),
                        int(ranges["Age"]["median"]))
        gender = c2.selectbox("Gender", ranges["Gender"]["options"])
        city = c3.selectbox("City", ranges["City"]["options"])
        income_band = c4.selectbox("Annual income band", list(INCOME_BANDS.keys()), index=1)

        st.markdown("#### What do they hold?")
        c1, c2, c3, c4 = st.columns(4)
        card_type = c1.selectbox("Card type", ranges["CardType"]["options"],
                                 index=ranges["CardType"]["options"].index("Silver")
                                 if "Silver" in ranges["CardType"]["options"] else 0)
        credit_limit = c2.number_input("Credit limit (₹)", int(ranges["CreditLimit_INR"]["min"]),
                                       int(ranges["CreditLimit_INR"]["max"]),
                                       int(ranges["CreditLimit_INR"]["median"]), 10_000)
        tenure = c3.slider("Tenure (months)", int(ranges["TenureMonths"]["min"]),
                           int(ranges["TenureMonths"]["max"]), int(ranges["TenureMonths"]["median"]))
        products = c4.slider("Products held with the bank", 1, 5, 1)

        st.markdown("#### How do they behave?")
        c1, c2, c3, c4 = st.columns(4)
        days_since = c1.slider("Days since last transaction", int(ranges["DaysSinceLastTxn"]["min"]),
                               int(ranges["DaysSinceLastTxn"]["max"]),
                               int(ranges["DaysSinceLastTxn"]["median"]),
                               help="The strongest single predictor in this dataset.")
        txn_month = c2.slider("Transactions per month", int(ranges["TxnPerMonth"]["min"]),
                              int(ranges["TxnPerMonth"]["max"]), int(ranges["TxnPerMonth"]["median"]))
        spend = c3.number_input("Avg monthly spend (₹)", int(ranges["AvgMonthlySpend_INR"]["min"]),
                                int(ranges["AvgMonthlySpend_INR"]["max"]),
                                int(ranges["AvgMonthlySpend_INR"]["median"]), 500)
        utilisation = c4.slider("Credit utilisation", 0.0, 1.0,
                                float(round(ranges["CreditUtilization"]["median"], 2)), 0.01)

        st.markdown("#### Relationship health")
        c1, c2, c3, c4, c5 = st.columns(5)
        late = c1.slider("Late payments (12m)", 0, int(ranges["LatePayments12M"]["max"]), 0)
        complaints = c2.slider("Complaints (6m)", 0, int(ranges["Complaints6M"]["max"]), 0)
        calls = c3.slider("Service calls (6m)", 0, int(ranges["ServiceCalls6M"]["max"]), 1)
        logins = c4.slider("App logins (3m)", 0, int(ranges["MobileAppLogins3M"]["max"]),
                           int(ranges["MobileAppLogins3M"]["median"]))
        rewards = c5.slider("Rewards redeemed (12m)", 0, int(ranges["RewardsRedeemed12M"]["max"]), 1)
        autopay = st.checkbox("Autopay enabled", value=False)

        st.markdown("#### Latest complaint text (optional)")
        verbatim = st.text_area(
            "Pasting a recent complaint adds a sentiment and exit-intent read alongside the model score.",
            "", height=90, placeholder="e.g. Third time I am raising this. Another bank is offering me a better card…",
        )

        submitted = st.form_submit_button("🎯 Predict churn risk", width="stretch")

    if not submitted:
        st.markdown(
            callout(
                "Fill in the form and hit predict. Defaults are the median customer in this book, which scores "
                "close to the base rate — move <i>days since last transaction</i> and <i>app logins</i> first "
                "to see the model react most sharply.",
                "info",
            ),
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------- predict
    values = {
        "Age": age, "Gender": gender, "City": city, "AnnualIncome_INR": INCOME_BANDS[income_band],
        "CardType": card_type, "CreditLimit_INR": credit_limit, "TenureMonths": tenure,
        "NumProductsHeld": products, "DaysSinceLastTxn": days_since, "TxnPerMonth": txn_month,
        "AvgMonthlySpend_INR": spend, "CreditUtilization": utilisation, "LatePayments12M": late,
        "Complaints6M": complaints, "ServiceCalls6M": calls, "MobileAppLogins3M": logins,
        "RewardsRedeemed12M": rewards, "AutoPayEnabled": int(autopay),
    }
    row = build_single_customer(values, df)
    prob = float(model.pipeline.predict_proba(row)[0, 1])
    threshold = model.best_threshold

    st.divider()
    left, right = st.columns([1, 1.25])

    with left:
        st.markdown(
            _verdict_card(prob, threshold, card_type, f"{city.upper()} · {tenure} MO · {age} YRS"),
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(kpi("Decision threshold", f"{threshold:.1%}",
                        f"tuned on out-of-fold predictions for {model_name}", VIOLET),
                    unsafe_allow_html=True)

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=prob * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#EEF0FF"}},
            delta={"reference": threshold * 100, "suffix": " pts vs threshold",
                   "increasing": {"color": CHURN}, "decreasing": {"color": RETAIN}},
            title={"text": f"Churn probability — {model_name}", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": CHURN if prob >= threshold else RETAIN, "thickness": 0.75},
                "bgcolor": "rgba(20,20,43,0.6)", "borderwidth": 1, "bordercolor": "#2A2A52",
                "steps": [
                    {"range": [0, 25], "color": "rgba(0,229,192,0.15)"},
                    {"range": [25, 50], "color": "rgba(46,155,255,0.15)"},
                    {"range": [50, 75], "color": "rgba(255,182,39,0.17)"},
                    {"range": [75, 100], "color": "rgba(255,46,99,0.20)"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 4}, "thickness": 0.85,
                              "value": threshold * 100},
            },
        ))
        fig.update_layout(height=330)
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------- consensus
    st.markdown("### What every model says")
    votes = []
    for name, m in run.models.items():
        p = float(m.pipeline.predict_proba(row)[0, 1])
        votes.append({"Model": name, "Probability": p, "Threshold": m.best_threshold,
                      "Verdict": "Churn" if p >= m.best_threshold else "Stay"})
    votes_df = pd.DataFrame(votes)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=votes_df["Probability"] * 100, y=votes_df["Model"], orientation="h",
            marker=dict(color=[CHURN if v == "Churn" else RETAIN for v in votes_df["Verdict"]]),
            text=[f"{p:.1%}" for p in votes_df["Probability"]], textposition="outside",
            name="Churn probability", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=votes_df["Threshold"] * 100, y=votes_df["Model"], mode="markers",
            marker=dict(symbol="line-ns", size=22, line=dict(color=GOLD, width=3)),
            name="Each model's threshold",
        ))
        fig.update_layout(title="Probability against each model's own cut-off", height=360,
                          xaxis_title="Churn probability (%)", xaxis_range=[0, 108],
                          legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, width="stretch")

    with c2:
        churn_votes = int((votes_df["Verdict"] == "Churn").sum())
        spread = votes_df["Probability"].max() - votes_df["Probability"].min()
        agreement = ("unanimous" if churn_votes in (0, len(votes_df))
                     else f"split {churn_votes}–{len(votes_df) - churn_votes}")
        st.markdown(kpi("Model consensus", f"{churn_votes}/{len(votes_df)} say churn", agreement,
                        CHURN if churn_votes > len(votes_df) / 2 else RETAIN), unsafe_allow_html=True)
        st.markdown("")
        st.markdown(kpi("Probability spread", f"{spread:.1%}",
                        "disagreement between highest and lowest", GOLD if spread > 0.25 else AZURE),
                    unsafe_allow_html=True)
        st.markdown(
            callout(
                ("The models disagree materially on this customer. That usually means the profile sits in a "
                 "sparse region of the training data — treat the score as a weak signal and lean on the "
                 "qualitative read." if spread > 0.25 else
                 "The models agree closely, which is the reassuring case: the verdict does not depend on which "
                 "algorithm happened to be selected."),
                "bad" if spread > 0.25 else "good",
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------- drivers
    st.markdown("### Why — the drivers behind this score")
    drivers = _explain(row, df, values)
    d1, d2 = st.columns([1.3, 1])
    with d1:
        fig = go.Figure(go.Bar(
            x=drivers["Contribution"], y=drivers["Factor"], orientation="h",
            marker=dict(color=[CHURN if v > 0 else RETAIN for v in drivers["Contribution"]]),
            text=drivers["Label"], textposition="outside", textfont=dict(size=11),
            cliponaxis=False, customdata=drivers["Detail"],
            hovertemplate="%{y}<br>%{customdata}<br>%{x:+.2f} SD<extra></extra>",
        ))
        span = max(float(drivers["Contribution"].abs().max()), 0.5)
        fig.update_layout(
            title="Risk-increasing (magenta) vs risk-reducing (aqua)", height=430,
            xaxis_title="Standard deviations from the book median",
            xaxis_range=[-span * 2.4, span * 2.4], bargap=0.35,
        )
        fig.add_vline(x=0, line_color="#2A2A52", line_width=1)
        st.plotly_chart(fig, width="stretch")
        flat = int((drivers["Contribution"].abs() < 0.05).sum())
        st.caption(
            "This is a z-score comparison against the portfolio median on the drivers permutation importance "
            "ranked highest — a plain-language read on where this customer is unusual. It is not a SHAP "
            "decomposition of the model's internals, and it is labelled that way deliberately rather than "
            "presented as something it is not."
            + (f" {flat} of these {len(drivers)} factors sit essentially on the book median for this customer, "
               f"so their bars are at zero — an unremarkable profile, which is itself the finding."
               if flat else "")
        )

    with d2:
        if verbatim.strip():
            sent = score_sentiment(verbatim)
            intent = churn_intent(verbatim)
            st.markdown(kpi("Text sentiment", sent["label"], f"score {sent['score']:+.2f}",
                            CHURN if sent["label"] == "Negative" else RETAIN), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(kpi("Exit intent", intent["band"], f"{intent['risk']:.0f}/100",
                            CHURN if intent["risk"] >= 60 else GOLD if intent["risk"] >= 30 else RETAIN),
                        unsafe_allow_html=True)
            if intent["phrases"]:
                st.markdown("**Flagged phrases**")
                st.markdown(" · ".join(f"`{p}`" for p in intent["phrases"]))
        else:
            st.markdown(
                callout("No complaint text supplied. Behavioural score only — add a verbatim in the form to "
                        "layer the language read on top.", "info"),
                unsafe_allow_html=True,
            )

        risk_band = ("Critical" if prob >= 0.7 else "High" if prob >= threshold
                     else "Moderate" if prob >= threshold * 0.6 else "Low")
        playbook = {
            "Critical": "Call within 48 hours. Lead with a concrete fee waiver or limit increase, not a survey.",
            "High": "Add to this cycle's outbound list. A targeted rewards bonus on their top spend category.",
            "Moderate": "Automated nudge only — a category-specific cashback offer to rebuild transaction habit.",
            "Low": "No intervention. Spending retention budget here is pure margin leakage.",
        }[risk_band]
        st.markdown(callout(f"<b>Suggested action — {risk_band} risk.</b> {playbook}",
                            "bad" if risk_band in ("Critical", "High") else "good"), unsafe_allow_html=True)

    # ------------------------------------------------------------- LLM play
    st.divider()
    st.markdown("### AI-generated retention play")
    c1, c2 = st.columns([2, 1])
    key_input = c1.text_input("Anthropic API key", type="password", placeholder="sk-ant-…",
                              key="pred_key", help="Optional. Also read from st.secrets or the environment.")
    model_choice = c2.selectbox("Model", ANTHROPIC_MODELS, index=0, key="pred_model")
    api_key = get_api_key(key_input)

    if not api_key:
        st.caption("Add an Anthropic API key to generate a tailored retention play for this customer. "
                   "Everything above works without one.")
    if st.button("Generate retention play", disabled=not api_key):
        profile = {
            "Age": age, "City": city, "Card type": card_type, "Income band": income_band,
            "Tenure": f"{tenure} months", "Credit limit": f"₹{credit_limit:,}",
            "Avg monthly spend": f"₹{spend:,}", "Utilisation": f"{utilisation:.0%}",
            "Transactions/month": txn_month, "Days since last transaction": days_since,
            "App logins (3m)": logins, "Late payments (12m)": late,
            "Complaints (6m)": complaints, "Service calls (6m)": calls,
            "Rewards redeemed (12m)": rewards, "Autopay": "yes" if autopay else "no",
            "Products held": products,
        }
        top_drivers = drivers.nlargest(4, "Contribution")["Factor"].tolist()
        with st.spinner("Claude is drafting the play…"):
            out = call_claude(build_retention_prompt(profile, prob, top_drivers, verbatim),
                              api_key, model_choice, system=RETENTION_SYSTEM, max_tokens=1200)
        st.markdown(f'<div class="panel">{out}</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------- export
    export = row.copy()
    export.insert(0, "PredictedProbability", prob)
    export.insert(0, "Verdict", "Churn" if prob >= threshold else "Stay")
    export.insert(0, "ScoringModel", model_name)
    st.download_button("⬇️ Download this scored customer (CSV)", export.to_csv(index=False),
                       "scored_customer.csv", "text/csv")


def _explain(row: pd.DataFrame, df: pd.DataFrame, values: dict) -> pd.DataFrame:
    """
    Plain-language driver read: z-score this customer against the book on the
    features that matter most, and orient each one so positive always means
    'pushes risk up'.
    """
    # (feature, direction) — +1 where a high value raises risk, −1 where it lowers it.
    spec = [
        ("DaysSinceLastTxn", +1), ("TxnPerMonth", -1), ("MobileAppLogins3M", -1),
        ("RewardsRedeemed12M", -1), ("TenureMonths", -1), ("CreditUtilization", +1),
        ("LatePayments12M", +1), ("Complaints6M", +1), ("ServiceCalls6M", +1),
        ("NumProductsHeld", -1),
    ]
    rows = []
    for feat, direction in spec:
        series = df[feat].dropna()
        med, sd = series.median(), series.std()
        if not sd or np.isnan(sd):
            continue
        val = float(row[feat].iloc[0])
        contribution = ((val - med) / sd) * direction
        # Fractional features (utilisation) need decimals; counts and rupees do not.
        fmt = "{:.2f}" if feat == "CreditUtilization" else "{:,.0f}"
        near = abs(val - med) < (sd * 0.05)
        comparison = "at median" if near else ("above" if val > med else "below")
        detail = (f"{fmt.format(val)} vs median {fmt.format(med)}"
                  + ("" if near else f" ({comparison})"))
        rows.append({
            "Factor": FEATURE_LABELS.get(feat, feat),
            "Contribution": contribution,
            "Detail": detail,
            # Bars sitting on zero get no floating label — the text would hover
            # in empty space and read as a rendering fault rather than a finding.
            "Label": "" if abs(contribution) < 0.05 else detail,
        })
    out = pd.DataFrame(rows)
    # Always show the same eight bars so the chart shape is stable between
    # customers — a single surviving bar stretched across the canvas reads as
    # a bug, and near-median customers are exactly the interesting boring case.
    out = out.reindex(out["Contribution"].abs().sort_values().index).tail(8)
    return out.sort_values("Contribution")
