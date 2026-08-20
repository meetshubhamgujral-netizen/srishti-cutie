"""
Modelling layer: preprocessing pipeline, the model zoo, cross-validation,
evaluation metrics, threshold optimisation and clustering.

Design notes
------------
* Every model is wrapped in a Pipeline so imputation and scaling are fitted
  *inside* each CV fold. Fitting a scaler on the full training set before
  splitting leaks fold-level information and inflates scores.
* The dataset is small (500 rows, 28.6% positive), so repeated stratified
  k-fold is used rather than a single split. A single 75/25 split on 500 rows
  puts only ~36 churners in the test set — far too few to trust one number.
* class_weight='balanced' handles the imbalance for the models that support it;
  KNN and Gradient Boosting get threshold tuning instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
    learning_curve, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, NUMERIC, TARGET, feature_matrix

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
        remainder="drop",
    )


def model_zoo() -> Dict[str, object]:
    """
    Hyperparameters are deliberately conservative — with 500 rows, deep trees and
    large ensembles memorise noise. Depth caps and leaf minimums are the main
    defence against overfitting here.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, C=0.6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=12, min_samples_split=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(
            n_neighbors=17, weights="distance", metric="minkowski", p=2
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.85,
            min_samples_leaf=8, random_state=RANDOM_STATE
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


# ------------------------------------------------------------------ metrics
def classification_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Acc.": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1-Score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "MCC": matthews_corrcoef(y_true, pred),
        # R² on predicted probability vs the 0/1 outcome. Also called the
        # Efron pseudo-R²: the share of outcome variance the probabilities explain.
        "R² (Efron)": r2_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def optimal_threshold(y_true, proba) -> Tuple[float, float]:
    """Threshold that maximises F1 — the right default when positives are the minority."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return (float(thr[best]) if len(thr) else 0.5), float(f1[best])


def profit_threshold(y_true, proba, retain_value: float, campaign_cost: float) -> Tuple[float, float]:
    """
    Business-optimal cut-off. Catching a churner is worth `retain_value`;
    every contacted customer costs `campaign_cost` whether they were leaving or not.
    """
    grid = np.linspace(0.05, 0.95, 91)
    profits = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        profits.append(tp * retain_value - (tp + fp) * campaign_cost)
    best = int(np.argmax(profits))
    return float(grid[best]), float(profits[best])


# ------------------------------------------------------------------ training
@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    cv_summary: Dict[str, float]
    oof_proba: np.ndarray
    test_proba: np.ndarray
    test_metrics: Dict[str, float]
    train_metrics: Dict[str, float]
    best_threshold: float
    tuned_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingRun:
    models: Dict[str, TrainedModel]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    leaderboard: pd.DataFrame
    best_name: str
    config: Dict


@st.cache_resource(show_spinner=False)
def train_all(
    df: pd.DataFrame,
    test_size: float = 0.25,
    n_splits: int = 5,
    n_repeats: int = 3,
    _cache_key: str = "v1",
) -> TrainingRun:
    """
    Fits every model, records cross-validated and held-out performance.

    Cached as a resource because trained pipelines are unpicklable-heavy objects
    that should persist across reruns rather than retrain on every click.
    """
    X, y = feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    oof_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc", "f1": "f1", "precision": "precision",
        "recall": "recall", "accuracy": "accuracy", "average_precision": "average_precision",
    }

    trained: Dict[str, TrainedModel] = {}
    for name, estimator in model_zoo().items():
        pipe = make_pipeline(estimator)

        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_summary = {
            "CV ROC-AUC": float(np.mean(cv_res["test_roc_auc"])),
            "CV AUC ±": float(np.std(cv_res["test_roc_auc"])),
            "CV F1": float(np.mean(cv_res["test_f1"])),
            "CV F1 ±": float(np.std(cv_res["test_f1"])),
            "CV Precision": float(np.mean(cv_res["test_precision"])),
            "CV Recall": float(np.mean(cv_res["test_recall"])),
            "CV Accuracy": float(np.mean(cv_res["test_accuracy"])),
            "CV PR-AUC": float(np.mean(cv_res["test_average_precision"])),
            "Fit time (s)": float(np.mean(cv_res["fit_time"])),
        }

        # Out-of-fold predictions on the training set — used for honest threshold
        # selection without ever touching the held-out test set.
        oof = cross_val_predict(pipe, X_train, y_train, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]

        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        train_proba = pipe.predict_proba(X_train)[:, 1]

        thr, _ = optimal_threshold(y_train, oof)
        trained[name] = TrainedModel(
            name=name,
            pipeline=pipe,
            cv_summary=cv_summary,
            oof_proba=oof,
            test_proba=test_proba,
            test_metrics=classification_metrics(y_test, test_proba, 0.5),
            train_metrics=classification_metrics(y_train, train_proba, 0.5),
            best_threshold=thr,
            tuned_metrics=classification_metrics(y_test, test_proba, thr),
        )

    rows = []
    for name, m in trained.items():
        rows.append({
            "Model": name,
            "CV ROC-AUC": m.cv_summary["CV ROC-AUC"],
            "CV AUC ±": m.cv_summary["CV AUC ±"],
            "CV F1": m.cv_summary["CV F1"],
            "Test ROC-AUC": m.test_metrics["ROC-AUC"],
            "Test F1": m.test_metrics["F1-Score"],
            "Tuned F1": m.tuned_metrics["F1-Score"],
            "Test Recall": m.test_metrics["Recall"],
            "Test Precision": m.test_metrics["Precision"],
            "PR-AUC": m.test_metrics["PR-AUC"],
            "R² (Efron)": m.test_metrics["R² (Efron)"],
            "Brier": m.test_metrics["Brier"],
            "Overfit gap": m.train_metrics["ROC-AUC"] - m.test_metrics["ROC-AUC"],
            "Threshold": m.best_threshold,
        })
    leaderboard = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)

    return TrainingRun(
        models=trained, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        leaderboard=leaderboard, best_name=leaderboard.iloc[0]["Model"],
        config={"test_size": test_size, "n_splits": n_splits, "n_repeats": n_repeats},
    )


# ------------------------------------------------------------------ diagnostics
@st.cache_data(show_spinner=False)
def compute_learning_curve(_pipeline, X, y, n_splits: int = 5) -> pd.DataFrame:
    sizes, train_scores, val_scores = learning_curve(
        _pipeline, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
        train_sizes=np.linspace(0.2, 1.0, 8), scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
    )
    return pd.DataFrame({
        "Training samples": sizes,
        "Train AUC": train_scores.mean(axis=1),
        "Train std": train_scores.std(axis=1),
        "Validation AUC": val_scores.mean(axis=1),
        "Val std": val_scores.std(axis=1),
    })


@st.cache_data(show_spinner=False)
def compute_permutation_importance(_pipeline, X, y, n_repeats: int = 12) -> pd.DataFrame:
    """
    Model-agnostic importance: shuffle one column, measure the AUC it costs.
    Works identically for KNN and boosted trees, unlike impurity-based importance.
    """
    res = permutation_importance(
        _pipeline, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "Feature": X.columns,
            "Importance": res.importances_mean,
            "Std": res.importances_std,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def calibration_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    """Are the probabilities honest? Bin them and compare predicted vs actual rate."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        Predicted=("p", "mean"), Actual=("y", "mean"), Count=("y", "size")
    ).reset_index(drop=True)
    return out.dropna()


def lift_table(y_true, proba, n_deciles: int = 10) -> pd.DataFrame:
    """Decile lift — the table a campaign manager actually reads before picking a list size."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_deciles, labels=False) + 1
    base = df["y"].mean()
    out = df.groupby("decile").agg(Customers=("y", "size"), Churners=("y", "sum"), Rate=("y", "mean")).reset_index()
    out["Lift"] = out["Rate"] / base if base else 0
    out["Cumulative churners"] = out["Churners"].cumsum()
    out["Capture %"] = out["Cumulative churners"] / max(out["Churners"].sum(), 1)
    return out


# ------------------------------------------------------------------ clustering
@st.cache_data(show_spinner=False)
def clustering_analysis(df: pd.DataFrame, features: List[str], k_range: Tuple[int, int] = (2, 10)) -> Dict:
    """
    K-Means segmentation with an elbow (inertia) curve and silhouette scores.

    Note this is unsupervised — the churn label is deliberately excluded from the
    feature set and only used afterwards to profile what each segment turned out
    to be. Clustering on the label would just be a convoluted decision tree.
    """
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X = prep.fit_transform(df[features])

    ks, inertia, silhouette = [], [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=12, random_state=RANDOM_STATE).fit(X)
        ks.append(k)
        inertia.append(float(km.inertia_))
        silhouette.append(float(silhouette_score(X, km.labels_)) if k > 1 else np.nan)

    # Elbow via maximum perpendicular distance to the line joining first and last point.
    pts = np.column_stack([np.array(ks, float), np.array(inertia, float)])
    pts_n = (pts - pts.min(0)) / (np.ptp(pts, axis=0) + 1e-12)
    start, end = pts_n[0], pts_n[-1]
    vec = end - start
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    proj = start + np.outer((pts_n - start) @ vec, vec)
    elbow_k = int(ks[int(np.argmax(np.linalg.norm(pts_n - proj, axis=1)))])

    return {
        "k_values": ks,
        "inertia": inertia,
        "silhouette": silhouette,
        "elbow_k": elbow_k,
        "silhouette_k": int(ks[int(np.nanargmax(silhouette))]),
        "X_scaled": X,
        "preprocessor": prep,
    }


@st.cache_data(show_spinner=False)
def fit_clusters(_X_scaled: np.ndarray, k: int) -> Dict:
    km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE).fit(_X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(_X_scaled)
    return {
        "labels": km.labels_,
        "coords": coords,
        "explained": pca.explained_variance_ratio_,
        "silhouette": float(silhouette_score(_X_scaled, km.labels_)),
    }


# ------------------------------------------------------------------ A/B testing
def ab_test(control_n: int, control_churn: int, treat_n: int, treat_churn: int) -> Dict:
    """
    Two-proportion z-test for a retention campaign readout.

    Implemented directly rather than pulled from statsmodels to keep the
    deployment dependency list short — it is a dozen lines of arithmetic.
    """
    from math import erf, sqrt

    p1 = control_churn / control_n if control_n else 0.0
    p2 = treat_churn / treat_n if treat_n else 0.0
    pooled = (control_churn + treat_churn) / (control_n + treat_n) if (control_n + treat_n) else 0.0
    se = sqrt(pooled * (1 - pooled) * (1 / max(control_n, 1) + 1 / max(treat_n, 1))) or 1e-12
    z = (p1 - p2) / se
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    se_diff = sqrt(p1 * (1 - p1) / max(control_n, 1) + p2 * (1 - p2) / max(treat_n, 1)) or 1e-12
    diff = p1 - p2
    return {
        "control_rate": p1,
        "treatment_rate": p2,
        "absolute_lift": diff,
        "relative_lift": (diff / p1) if p1 else 0.0,
        "z_stat": z,
        "p_value": p_value,
        "ci_low": diff - 1.96 * se_diff,
        "ci_high": diff + 1.96 * se_diff,
        "significant": p_value < 0.05,
    }


def simulate_campaign(y_true, proba, threshold: float, effectiveness: float, seed: int = RANDOM_STATE) -> Dict:
    """
    Split the high-risk customers the model flagged into control and treatment,
    apply an assumed save rate to treatment, and read the result as an A/B test.

    This is a *simulation* for sizing and planning — it shows what the experiment
    would look like at this list size, not evidence that the campaign works.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    flagged = np.where(np.asarray(proba) >= threshold)[0]
    if len(flagged) < 10:
        return {"error": "Fewer than 10 customers cross this threshold — lower it to size a campaign."}

    shuffled = rng.permutation(flagged)
    half = len(shuffled) // 2
    control_idx, treat_idx = shuffled[:half], shuffled[half:]

    control_churn = int(y_true[control_idx].sum())
    would_churn = y_true[treat_idx] == 1
    saved = rng.random(int(would_churn.sum())) < effectiveness
    treat_churn = int(would_churn.sum() - saved.sum())

    result = ab_test(len(control_idx), control_churn, len(treat_idx), treat_churn)
    result.update({
        "control_n": len(control_idx), "treatment_n": len(treat_idx),
        "control_churn": control_churn, "treatment_churn": treat_churn,
        "customers_saved": int(saved.sum()), "flagged_total": len(flagged),
    })
    return result


def required_sample_size(baseline_rate: float, mde_relative: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Per-arm sample size for a two-proportion test at the given minimum detectable effect."""
    z_alpha, z_beta = 1.96, {0.8: 0.842, 0.9: 1.282, 0.95: 1.645}.get(round(power, 2), 0.842)
    p1 = baseline_rate
    p2 = baseline_rate * (1 - mde_relative)
    p_bar = (p1 + p2) / 2
    if abs(p1 - p2) < 1e-9:
        return 10 ** 9
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    return int(np.ceil(n))
