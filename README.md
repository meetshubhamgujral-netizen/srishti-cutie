# 💳 ChurnLab — Credit Card Attrition Intelligence

A six-tab Streamlit dashboard that predicts which credit card holders are about to close their account, and prices the decision to intervene.

Built on a 500-customer Indian credit card portfolio (₹ INR, six metro markets). Five competing classifiers are trained and cross-validated on every page load, evaluated with the metrics that survive class imbalance, and exposed through a live scoring form.

---

## What's inside

| Tab | What it does |
|---|---|
| 📊 **Overview** | Portfolio KPIs, data-quality audit (missingness, dirty categories, treatment applied), churn-rate breakdowns by card tier / city / behaviour, correlation structure |
| 🤖 **Model Lab** | Five algorithms trained side by side — Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Gradient Boosting. Repeated stratified k-fold CV, leaderboard, per-fold variance, feature importance, learning curves |
| 📈 **Evaluation** | ROC curves (all models overlaid), precision–recall curves, confusion matrices, calibration plot, decile lift, threshold economics, and an A/B test simulator with a two-proportion z-test and power analysis |
| 🧩 **Segmentation** | K-Means clustering with an **elbow chart** (WCSS) and silhouette second opinion, PCA projection, and named behavioural personas with churn rate per segment |
| 🧠 **NLP & LLM** | TF-IDF, NMF topic modelling, lexicon sentiment, churn-discriminative term extraction, plus optional Claude API integration for retention playbooks |
| 🔮 **Predict** | The scoring form — enter age, income band, card type, tenure, spend, utilisation and engagement signals, get a churn verdict with driver attribution and model consensus |

---

## Results

Cross-validated on 500 customers, 5 folds × 3 repeats — 15 fits per algorithm, 75 across the zoo:

| Model | CV ROC-AUC | CV F1 | Test ROC-AUC | R² (Efron) | Brier |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.854 ± 0.031** | 0.657 | 0.881 | 0.348 | 0.134 |
| Random Forest | 0.822 ± 0.038 | 0.606 | 0.883 | 0.349 | 0.133 |
| Gradient Boosting | 0.817 ± 0.035 | 0.577 | 0.842 | 0.367 | 0.130 |
| K-Nearest Neighbours | 0.801 ± 0.045 | 0.327 | 0.869 | 0.316 | 0.140 |
| Decision Tree | 0.724 ± 0.049 | 0.544 | 0.776 | 0.045 | 0.196 |

**The honest headline:** logistic regression wins. On 500 rows with a largely linear signal, the ensembles have nothing extra to find and pay for their flexibility in variance. That is a finding, not a failure — reporting it is the point of running five models instead of one.

**What actually drives churn here:** inactivity and disengagement, not demographics. `DaysSinceLastTxn` (r = +0.44), `MobileAppLogins3M` (r = −0.29) and `TxnPerMonth` (r = −0.28) carry the signal. Card tier, city and income are close to noise — worth knowing before anyone builds a segment-based retention campaign.

---

## Run it locally

```bash
git clone https://github.com/<your-username>/churnlab.git
cd churnlab
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. First load trains all five models (~30s), then caches.

---

## Deploy to Streamlit Community Cloud

1. **Push this folder to a new GitHub repo.**

   ```bash
   git init
   git add .
   git commit -m "ChurnLab: credit card attrition dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/churnlab.git
   git push -u origin main
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **New app** → select your repo → set:
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Click **Deploy**. First build takes 2–4 minutes while dependencies install.

5. *(Optional — for the LLM features)* In **App settings → Secrets**, paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

   Everything except the LLM panels works without a key.

> The app trains on startup rather than loading a pickle. That is deliberate — pickled scikit-learn models break when the cloud's library versions drift from your local ones, and 500 rows train in seconds.

---

## Repo structure

```
churnlab/
├── app.py                      # entry point, sidebar, tab routing
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── customer_churn_data.csv # bundled dataset (500 customers)
├── .streamlit/
│   ├── config.toml             # dark violet theme
│   └── secrets.toml.example    # template — real secrets are gitignored
└── src/
    ├── data.py                 # loading, cleaning, feature engineering
    ├── models.py               # model zoo, CV, metrics, clustering, A/B test
    ├── nlp.py                  # corpus synthesis, TF-IDF, topics, Claude client
    ├── theme.py                # palette + injected CSS
    └── tabs/                   # one module per tab
```

---

## Methodology notes

**Cleaning.** `Gender` arrives dirty (`Male`, `male `, `M`, `F`, `Female`) and is normalised to two levels. `Age`, `AnnualIncome_INR` and `MobileAppLogins3M` have missing values, imputed inside the cross-validation pipeline — never before the split, which would leak test statistics into training.

**Feature engineering.** Seven ratios are derived from the raw columns: spend-to-limit, engagement (transactions × app logins), friction (complaints + service calls + late payments), inactivity ratio (recency ÷ tenure), spend per transaction, income-to-limit, and rewards redeemed per year of tenure. They are computed by a single shared function so the values are identical at training and scoring time — the most common source of train/serve skew is two copies of this logic drifting apart.

**Validation.** Repeated stratified k-fold, with a held-out test set quarantined from the entire training and tuning process and touched once. Decision thresholds are tuned on out-of-fold predictions, not on the test set.

**R² on a binary target.** Plain R² is not defined the usual way for classification, so the dashboard reports **Efron's pseudo-R²** — the ordinary 1 − SSE/SST computed on predicted probabilities against the 0/1 outcome. It is a genuine variance-explained figure, and it is labelled as pseudo-R² rather than passed off as the regression statistic.

**Class imbalance.** 28.6% churn. Accuracy is reported but deliberately de-emphasised in favour of ROC-AUC, PR-AUC, F1 and Brier score, with `class_weight="balanced"` where the algorithm supports it.

---

## Known limitations

- **500 rows is small.** Confidence intervals on every metric are wide; the ±0.03–0.05 CV standard deviations are reported precisely so this stays visible rather than hidden behind a single number.
- **The NLP corpus is synthetic.** The source CSV has no free-text column. The NLP tab generates service-log narratives from each customer's structured signals, then runs real TF-IDF, NMF topic modelling and sentiment scoring over them. This demonstrates the pipeline honestly — it is flagged inside the tab itself, and the topic findings should not be read as customer voice-of-customer evidence. Point the same pipeline at real complaint transcripts and it works unchanged.
- **The A/B tab is a simulator**, for sizing and planning a retention test — not the readout of one that ran.
- **Churn is modelled cross-sectionally**, without a time dimension. A production version would use a survival model or a rolling observation window.

---

Built with scikit-learn, Plotly and Streamlit.
