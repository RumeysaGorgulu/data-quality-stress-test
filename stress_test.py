"""
Data-Quality Stress Test

How much do common training-data problems change an ML model's accuracy and
fairness? We corrupt the *training* data in controlled ways, train models,
and evaluate on a *clean* test set.

Corruptions (each at increasing severity):
  missing_random   values go missing completely at random (MCAR)
  missing_group    values go missing only for the minority group (missingness tied to group)
  label_noise      a random share of training labels is flipped
  label_noise_group  positive labels of the minority group are flipped to negative
  label_noise_matched  the SAME NUMBER of labels as label_noise_group, flipped at random
                       (control: separates "how many labels are wrong" from "which ones")
  selection_bias   positive examples of the minority group are under-sampled

Inspired by work on data errors in ML pipelines (e.g., Zhu, ..., Salimi, 2025,
"Stress-Testing ML Pipelines with Adversarial Data Corruption"); this is a
small, independent study with simple, non-adversarial corruptions.

Usage
    python stress_test.py --dataset adult        # UCI Adult census data (downloads via OpenML)
    python stress_test.py --dataset synthetic    # offline, simulated data
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEVERITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CORRUPTIONS = ["missing_random", "missing_group", "label_noise", "label_noise_group",
               "label_noise_matched", "selection_bias"]


# Data
def load_adult():
    """UCI Adult: predict income > $50K. Group = sex (1 = female, the minority in this data)."""
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna().reset_index(drop=True)
    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).to_numpy()
    group = (df["sex"].astype(str).str.strip() == "Female").astype(int).to_numpy()
    X = df.drop(columns=["class", "sex", "fnlwgt"])
    return X, y, group, "Adult census (group = female)"


def load_synthetic(seed=0):
    """Simulated data where the minority group (30%) has a shifted feature distribution."""
    X, y = make_classification(n_samples=10000, n_features=8, n_informative=5,
                               n_redundant=1, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    group = (rng.random(len(y)) < 0.3).astype(int)
    X[group == 1, 0] += 0.75
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    X["region"] = rng.choice(["north", "south", "east"], size=len(y))
    return X, y, group, "Synthetic (group = simulated minority, 30%)"


# Corruptions (applied to TRAINING data only)
def corrupt(X, y, g, kind, severity, rng):
    X, y, g = X.copy(), y.copy(), g.copy()
    num_cols = X.select_dtypes(include="number").columns
    if severity == 0:
        return X, y, g

    if kind == "missing_random":
        mask = rng.random((len(X), len(num_cols))) < severity
        X[num_cols] = X[num_cols].astype(float).mask(mask)

    elif kind == "missing_group":
        # same overall missing budget as missing_random, concentrated in the minority group
        rate = min(1.0, severity * len(X) / max(1, g.sum()))
        mask = (rng.random((len(X), len(num_cols))) < rate) & (g[:, None] == 1)
        X[num_cols] = X[num_cols].astype(float).mask(mask)

    elif kind == "label_noise":
        flip = rng.random(len(y)) < severity
        y[flip] = 1 - y[flip]

    elif kind == "label_noise_group":
        target = (g == 1) & (y == 1)
        flip = target & (rng.random(len(y)) < severity)
        y[flip] = 0

    elif kind == "label_noise_matched":
        # same expected number of flips as label_noise_group, spread over all rows and classes
        k = int(round(severity * ((g == 1) & (y == 1)).sum()))
        flip = rng.choice(len(y), size=k, replace=False)
        y[flip] = 1 - y[flip]

    elif kind == "selection_bias":
        target = (g == 1) & (y == 1)
        keep = ~(target & (rng.random(len(y)) < severity))
        X, y, g = X[keep], y[keep], g[keep]

    return X, y, g


# Models and missing-data strategies
def build_pipeline(X, model, imputation):
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    if imputation == "native":  # HistGradientBoosting handles NaN itself
        num_steps = [("pass", "passthrough")]
    else:
        num_steps = [("impute", SimpleImputer(strategy="mean", add_indicator=(imputation == "mean+indicator")))]
    if model == "logistic":
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    clf = (LogisticRegression(max_iter=2000) if model == "logistic"
           else HistGradientBoostingClassifier(random_state=0))
    return Pipeline([("pre", pre), ("clf", clf)])


# (model, missing-data strategy) combinations to compare
SETUPS = [
    ("logistic", "mean"),
    ("logistic", "mean+indicator"),
    ("logistic", "drop_rows"),
    ("boosting", "native"),
]


def fairness_metrics(y_true, y_pred, g):
    def tpr(mask):
        pos = mask & (y_true == 1)
        return y_pred[pos].mean() if pos.any() else np.nan
    return {
        "accuracy": (y_pred == y_true).mean(),
        "tpr_gap": tpr(g == 0) - tpr(g == 1),                         # equal-opportunity gap
        "selection_gap": y_pred[g == 0].mean() - y_pred[g == 1].mean(),  # demographic-parity gap
    }


# Experiment loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "synthetic"], default="adult")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    X, y, g, label = load_adult() if args.dataset == "adult" else load_synthetic()
    print(f"Dataset: {label} | n = {len(y)} | positive rate = {y.mean():.2f} | minority share = {g.mean():.2f}")

    rows = []
    start = time.time()
    for seed in range(args.seeds):
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, g, test_size=0.3, random_state=seed, stratify=y)
        for kind in CORRUPTIONS:
            for sev in SEVERITIES:
                rng = np.random.default_rng(1000 * seed + int(sev * 100))
                Xc, yc, gc = corrupt(X_tr, y_tr, g_tr, kind, sev, rng)
                # How much of the training data was actually changed (comparable "dose")
                if kind.startswith("missing"):
                    num = Xc.select_dtypes(include="number")
                    share_changed = float(num.isna().to_numpy().mean())
                elif kind == "selection_bias":
                    share_changed = 1 - len(yc) / len(y_tr)
                else:
                    share_changed = float((yc != y_tr).mean())
                for model, imputation in SETUPS:
                    if not kind.startswith("missing") and imputation in ("mean+indicator", "drop_rows"):
                        # missing-data strategies only differ when values are missing
                        continue
                    Xf, yf = Xc, yc
                    if imputation == "drop_rows":
                        keep = Xc.notna().all(axis=1).to_numpy()
                        Xf, yf = Xc[keep], yc[keep]
                        if len(np.unique(yf)) < 2:
                            continue
                    pipe = build_pipeline(X, model, imputation).fit(Xf, yf)
                    m = fairness_metrics(y_te, pipe.predict(X_te), g_te)
                    rows.append({"seed": seed, "corruption": kind, "severity": sev,
                                 "setup": f"{model} / {imputation}", "n_train": len(yf),
                                 "share_changed": share_changed, **m})
        print(f"  seed {seed + 1}/{args.seeds} done ({time.time() - start:.0f}s)")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res["dataset"] = label
    res.to_csv(out / "results.csv", index=False)
    print(f"Saved {len(res)} runs to {out / 'results.csv'}. Next: python plot_results.py")


if __name__ == "__main__":
    main()
