"""
NLP layer: unstructured-text intelligence over customer service interactions.

IMPORTANT — read this before quoting any number from this tab
-------------------------------------------------------------
`customer_churn_data.csv` contains no free-text column. It records Complaints6M
and ServiceCalls6M as *counts*, not transcripts. To demonstrate the NLP pipeline
end to end, this module generates a synthetic verbatim corpus, conditioned on
each customer's real complaint/service-call counts, card tier and churn outcome.

The pipeline (sentiment, topic extraction, intent detection, LLM summarisation)
is production-grade and would run unchanged against real verbatims. The *text*
is manufactured. Findings from it describe the generator, not your customers.
Swap in a real transcript column and every number below becomes meaningful.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------------------- lexicons
NEGATIVE = {
    "terrible": -3, "awful": -3, "worst": -3, "horrible": -3, "unacceptable": -3, "disgusted": -3,
    "furious": -3, "outrageous": -3, "fraud": -3, "scam": -3, "cheated": -3, "harassment": -3,
    "angry": -2, "frustrated": -2, "disappointed": -2, "poor": -2, "useless": -2, "failed": -2,
    "rude": -2, "ignored": -2, "denied": -2, "wrong": -2, "hidden": -2, "unfair": -2, "misleading": -2,
    "delay": -2, "delayed": -2, "stuck": -2, "blocked": -2, "declined": -2, "penalty": -2,
    "slow": -1, "confusing": -1, "difficult": -1, "unhappy": -1, "waiting": -1, "expensive": -1,
    "issue": -1, "problem": -1, "complaint": -1, "error": -1, "charge": -1, "reduced": -1, "again": -1,
}
POSITIVE = {
    "excellent": 3, "outstanding": 3, "delighted": 3, "love": 3, "fantastic": 3,
    "great": 2, "happy": 2, "helpful": 2, "resolved": 2, "quick": 2, "smooth": 2, "appreciate": 2,
    "good": 1, "fine": 1, "okay": 1, "thanks": 1, "improved": 1, "clear": 1,
}
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "really": 1.4, "totally": 1.6, "absolutely": 1.8, "completely": 1.7}
NEGATIONS = {"not", "no", "never", "nothing", "cannot", "cant", "wont", "didnt", "doesnt", "isnt"}

# Phrases that signal a customer is actively planning to leave, weighted by severity.
CHURN_INTENT = {
    "cancel my card": 3.0, "close my account": 3.0, "closing the account": 3.0, "want to cancel": 3.0,
    "switch to": 2.5, "switching to": 2.5, "moving to": 2.5, "better offer from": 2.5, "shifted to": 2.5,
    "stop using": 2.0, "last chance": 2.0, "not worth": 2.0, "no longer worth": 2.0, "why should i keep": 2.0,
    "considering other": 1.8, "looking at other": 1.8, "another bank": 1.8, "competitor": 1.5,
    "disappointed": 1.0, "escalate": 1.2, "third time": 1.2, "still not resolved": 1.5,
}

THEMES: Dict[str, List[str]] = {
    "Fees & charges": [
        "The annual fee of {fee} was debited without any prior intimation and I only saw it on the statement.",
        "I was told the {tier} card fee would be waived on reaching the spend milestone, but the charge appeared anyway.",
        "Hidden convenience charges keep showing up on my statement. Nobody explains what they are for.",
        "The renewal fee is unfair when I barely get any benefit from this card.",
    ],
    "Rewards & benefits": [
        "My reward points expired before I could redeem them and no reminder was ever sent.",
        "The redemption catalogue has become useless. The same points fetched far more value last year.",
        "Lounge access was declined at the airport even though the {tier} card is supposed to include it.",
        "Cashback on online spends was quietly reduced without informing existing customers.",
    ],
    "Service & call centre": [
        "I have called customer care three times about this and every agent gives a different answer.",
        "The executive was rude and disconnected the call while I was still explaining the issue.",
        "Nobody has responded to my complaint reference for over two weeks now.",
        "Being transferred between four departments for a simple query is unacceptable service.",
    ],
    "Digital & app experience": [
        "The mobile app crashes every time I try to view my statement. Very frustrating.",
        "OTP never arrives on time so my transactions keep failing at checkout.",
        "Net banking login has been blocked twice this month for no reason.",
        "The app is slow and the new update removed the spend analysis I actually used.",
    ],
    "Fraud & disputes": [
        "There is an unauthorised transaction of {amount} on my card and the dispute is still not resolved.",
        "I reported a fraudulent charge weeks ago and the amount has still not been reversed.",
        "My card was blocked without notice and I was stranded while travelling.",
        "The chargeback was denied without any proper explanation of the investigation.",
    ],
    "Credit limit & billing": [
        "My credit limit was reduced without any warning even though I have never missed a payment.",
        "Late payment charges were levied although the payment was made on the due date.",
        "Interest was charged on the full amount despite a partial payment being made.",
        "I requested a limit increase four months ago and there has been no update at all.",
    ],
    "Competitor comparison": [
        "Another bank is offering me a lifetime free card with better rewards. Give me a reason to stay.",
        "My colleague's {other_tier} card from a competitor gives double the cashback on the same spends.",
        "I am seriously considering switching to a card that actually values long-term customers.",
        "If the benefits are not improved I will move my spends to another card next month.",
    ],
}

CLOSERS_CHURN = [
    " I want to cancel my card if this is not fixed.",
    " Please tell me the process to close my account.",
    " I am switching to another bank after this.",
    " This is my last chance before I stop using this card.",
    " Frankly this card is no longer worth keeping.",
]
CLOSERS_STAY = [
    " Please look into it and revert.",
    " Kindly resolve this at the earliest.",
    " I would appreciate a clear explanation.",
    " Hoping this gets sorted quickly.",
]

STOPWORDS = set("""
a an the and or but if while of to in on for with without from by at as is are was were be been being
i me my we our you your he she it they them this that these those have has had do does did will would
can could should may might must not no nor so than then there here when where which who whom whose why how
have been card bank please kindly sir madam am pm rs inr get got very really just also am
""".split())


# --------------------------------------------------------------- corpus
@st.cache_data(show_spinner=False)
def generate_corpus(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Build one synthetic verbatim per customer who had a complaint or service call.

    Theme selection is conditioned on real columns (late payments push toward
    billing themes, low app logins toward digital, high tenure + churn toward
    competitor comparison) so the corpus is *correlated* with the structured data
    rather than random noise — which is what makes the downstream analysis
    behave like it would on real transcripts.
    """
    rng = np.random.default_rng(seed)
    theme_names = list(THEMES.keys())
    records = []

    for _, r in df.iterrows():
        interactions = int(r.get("Complaints6M", 0)) + int(r.get("ServiceCalls6M", 0))
        if interactions <= 0:
            continue

        weights = np.ones(len(theme_names))
        idx = {t: i for i, t in enumerate(theme_names)}
        if r.get("LatePayments12M", 0) >= 2:
            weights[idx["Credit limit & billing"]] += 2.5
        if pd.notna(r.get("MobileAppLogins3M")) and r.get("MobileAppLogins3M", 99) < 6:
            weights[idx["Digital & app experience"]] += 2.0
        if r.get("RewardsRedeemed12M", 0) <= 1:
            weights[idx["Rewards & benefits"]] += 2.0
        if r.get("CardType") in ("Gold", "Platinum"):
            weights[idx["Fees & charges"]] += 1.5
        if r.get("Churned", 0) == 1 and r.get("TenureMonths", 0) > 24:
            weights[idx["Competitor comparison"]] += 2.5
        if r.get("Complaints6M", 0) >= 2:
            weights[idx["Service & call centre"]] += 2.0
            weights[idx["Fraud & disputes"]] += 1.0

        theme = theme_names[int(rng.choice(len(theme_names), p=weights / weights.sum()))]
        body = str(rng.choice(THEMES[theme]))
        body = (body
                .replace("{tier}", str(r.get("CardType", "Silver")))
                .replace("{other_tier}", str(rng.choice(["Gold", "Platinum", "Signature"])))
                .replace("{fee}", f"₹{int(rng.choice([1500, 2500, 5000, 10000])):,}")
                .replace("{amount}", f"₹{int(rng.integers(3, 90)) * 1000:,}"))

        churned = int(r.get("Churned", 0)) == 1
        # Churners are more likely to voice explicit exit intent — but not always,
        # and non-churners sometimes threaten to leave. Silent attrition is real.
        closer = str(rng.choice(CLOSERS_CHURN)) if rng.random() < (0.62 if churned else 0.14) else str(rng.choice(CLOSERS_STAY))

        records.append({
            "CustomerID": r["CustomerID"],
            "CardType": r.get("CardType"),
            "City": r.get("City"),
            "Churned": int(r.get("Churned", 0)),
            "Interactions": interactions,
            "Theme (generated)": theme,
            "Verbatim": body + closer,
        })

    return pd.DataFrame(records)


# --------------------------------------------------------------- sentiment
def score_sentiment(text: str) -> Dict[str, float]:
    """
    Lexicon sentiment with negation and intensifier handling.

    Deliberately rule-based: it needs no model download, runs instantly on
    Streamlit Community Cloud's free tier, and every score is traceable to the
    exact tokens that produced it — which matters when an analyst asks why.
    """
    tokens = re.findall(r"[a-z']+", str(text).lower())
    score, hits = 0.0, []
    for i, tok in enumerate(tokens):
        val = NEGATIVE.get(tok, 0) or POSITIVE.get(tok, 0)
        if not val:
            continue
        mult = 1.0
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            mult *= INTENSIFIERS[tokens[i - 1]]
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATIONS for w in window):
            mult *= -0.75
        score += val * mult
        hits.append(tok)

    norm = float(np.tanh(score / 4.0))  # squash to [-1, 1]
    label = "Negative" if norm <= -0.25 else ("Positive" if norm >= 0.25 else "Neutral")
    return {"score": norm, "raw": score, "label": label, "terms": hits}


def churn_intent(text: str) -> Dict[str, float]:
    """Phrase-match explicit exit language and convert to a 0-100 risk signal."""
    low = str(text).lower()
    matched, total = [], 0.0
    for phrase, weight in CHURN_INTENT.items():
        if phrase in low:
            matched.append(phrase)
            total += weight
    risk = float(min(100.0, total / 6.0 * 100.0))
    band = "Critical" if risk >= 60 else ("Elevated" if risk >= 30 else ("Watch" if risk > 0 else "None detected"))
    return {"risk": risk, "band": band, "phrases": matched}


@st.cache_data(show_spinner=False)
def analyse_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    out = corpus.copy()
    sent = out["Verbatim"].apply(score_sentiment)
    out["Sentiment"] = [s["score"] for s in sent]
    out["Sentiment label"] = [s["label"] for s in sent]
    intent = out["Verbatim"].apply(churn_intent)
    out["Intent risk"] = [i["risk"] for i in intent]
    out["Intent band"] = [i["band"] for i in intent]
    out["Exit phrases"] = [", ".join(i["phrases"]) if i["phrases"] else "—" for i in intent]
    out["Words"] = out["Verbatim"].str.split().str.len()
    return out


# --------------------------------------------------------------- topics & keywords
@st.cache_data(show_spinner=False)
def extract_topics(texts: List[str], n_topics: int = 6, n_terms: int = 8) -> pd.DataFrame:
    """TF-IDF + NMF topic modelling — unsupervised discovery of what drives contacts."""
    if len(texts) < n_topics * 2:
        return pd.DataFrame()
    vec = TfidfVectorizer(max_df=0.85, min_df=2, stop_words=list(STOPWORDS), ngram_range=(1, 2), max_features=1200)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=600).fit(X)
    vocab = np.array(vec.get_feature_names_out())
    weights = nmf.transform(X).sum(axis=0)

    rows = []
    for i, comp in enumerate(nmf.components_):
        top = vocab[np.argsort(comp)[::-1][:n_terms]]
        rows.append({
            "Topic": f"Topic {i + 1}",
            "Top terms": ", ".join(top),
            "Share of corpus": weights[i] / weights.sum(),
        })
    return pd.DataFrame(rows).sort_values("Share of corpus", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def churn_discriminative_terms(texts: List[str], labels: List[int], top_n: int = 15) -> pd.DataFrame:
    """
    Which words appear disproportionately in churners' verbatims?

    Log-odds with add-one smoothing — more stable than a raw frequency ratio when
    counts are small, which they always are on a 500-row sample.
    """
    vec = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=3, max_features=900)
    X = vec.fit_transform(texts).toarray()
    y = np.asarray(labels)
    vocab = np.array(vec.get_feature_names_out())

    churn_counts = X[y == 1].sum(axis=0) + 1
    stay_counts = X[y == 0].sum(axis=0) + 1
    log_odds = np.log((churn_counts / churn_counts.sum()) / (stay_counts / stay_counts.sum()))

    df = pd.DataFrame({
        "Term": vocab, "Log-odds (churn)": log_odds,
        "In churner texts": X[y == 1].sum(axis=0), "In retained texts": X[y == 0].sum(axis=0),
    })
    top = df.nlargest(top_n, "Log-odds (churn)")
    bottom = df.nsmallest(top_n, "Log-odds (churn)")
    return pd.concat([top, bottom]).reset_index(drop=True)


# --------------------------------------------------------------- LLM
ANTHROPIC_MODELS = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"]


def get_api_key(user_key: str = "") -> Optional[str]:
    """Precedence: what the user typed > st.secrets > environment variable."""
    if user_key and user_key.strip():
        return user_key.strip()
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def call_claude(prompt: str, api_key: str, model: str = "claude-sonnet-5",
                system: str = "", max_tokens: int = 1400) -> str:
    """
    Thin wrapper over the Anthropic Messages API.

    Errors are returned as readable strings rather than raised — a dashboard
    should degrade to a clear message, not a traceback in the middle of the page.
    """
    try:
        import anthropic
    except ImportError:
        return "⚠️ The `anthropic` package is not installed. Add `anthropic` to requirements.txt and redeploy."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Anthropic API call failed: {type(exc).__name__} — {exc}"


RETENTION_SYSTEM = (
    "You are a senior retention strategist at an Indian credit card issuer. "
    "You write for a retention operations team that has to act tomorrow morning. "
    "Be concrete and specific: name the offer, the channel, the timing and the cost. "
    "Never invent statistics that were not given to you. If the evidence is thin, say so plainly. "
    "Amounts are in Indian Rupees (₹). Keep the whole response under 400 words."
)


def build_retention_prompt(profile: Dict, probability: float, drivers: List[str], verbatim: str = "") -> str:
    lines = [f"- {k}: {v}" for k, v in profile.items()]
    prompt = (
        f"A churn model scored this credit card customer at {probability:.1%} probability of attrition "
        f"in the next quarter.\n\nCustomer profile:\n" + "\n".join(lines) +
        f"\n\nThe model's strongest risk drivers for this customer: {', '.join(drivers) if drivers else 'not supplied'}."
    )
    if verbatim.strip():
        prompt += f"\n\nMost recent complaint verbatim from this customer:\n\"{verbatim.strip()}\""
    prompt += (
        "\n\nProduce:\n"
        "1. RISK READ — two sentences on why this customer is leaving, in plain language.\n"
        "2. RETENTION PLAY — the single best intervention, with channel, timing and an estimated cost in ₹.\n"
        "3. BACKUP PLAY — one cheaper fallback if the primary offer is declined.\n"
        "4. DO NOT — one thing the team should avoid with this specific customer.\n"
        "5. ECONOMICS — is this customer worth saving? Reason it out from the profile."
    )
    return prompt


def build_corpus_prompt(analysed: pd.DataFrame, sample_size: int = 40) -> str:
    sample = analysed.nlargest(min(sample_size, len(analysed)), "Intent risk")
    blocks = [
        f"[{r.CustomerID} | {r.CardType} | churned={r.Churned} | sentiment={r.Sentiment:+.2f}] {r.Verbatim}"
        for r in sample.itertuples()
    ]
    return (
        "Below are customer service verbatims from a credit card portfolio, sorted by exit-intent score. "
        "Note: this text is synthetically generated for a demonstration, so treat it as a structural exercise "
        "rather than real market evidence, and say so once at the top.\n\n"
        + "\n\n".join(blocks) +
        "\n\nProduce:\n"
        "1. The three dominant complaint drivers, ranked, with a rough share of the sample.\n"
        "2. The specific language that separates customers who left from those who stayed.\n"
        "3. Two operational fixes that would remove the root cause, not just the symptom.\n"
        "4. One early-warning trigger the bank could monitor automatically."
    )
