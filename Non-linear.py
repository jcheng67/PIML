#!/usr/bin/env python3
"""
Regularized non-Arrhenius model for log10 conductivity using SAME inputs:
T_K, dopant_A, xA_mol, dopant_B, xB_mol, xV_mol, density_matrix (g/mL), log(Conductivity(S/cm))

Pipeline:
- Categorical: OneHot(dopant_A, dopant_B)
- Numeric: PolynomialFeatures(deg=2) on [T_K, xA_mol, xB_mol, xV_mol, density], then StandardScaler
- Estimator: ElasticNetCV (L1+L2), target standardized via TransformedTargetRegressor

Artifacts:
- nonarr_elastic_model.joblib
- pred_vs_actual_nonarr_elastic.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

REQUIRED_COLS = [
    "T_K", "dopant_A", "xA_mol", "dopant_B", "xB_mol",
    "xV_mol", "density_matrix (g/mL)", "log(Conductivity(S/cm))"
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["log(Conductivity(S/cm))"].std() == 0:
        raise ValueError("Target has zero variance.")
    return df.copy()

def build_pipeline(cat_cols, num_cols):
    # OHE version-safe
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # IMPORTANT: poly -> scaler (scale AFTER generating powers/cross terms)
    num_map = Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer([
        ("cat",  ohe, cat_cols),
        ("num",  num_map, num_cols)
    ])

    # ElasticNetCV with target standardization
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-4, 2, 40),
        cv=5,
        max_iter=10000,
        n_jobs=None,
        fit_intercept=True,
        normalize=False
    )
    # Standardize y during fitting for a well-conditioned penalty
    model = TransformedTargetRegressor(
        regressor=enet,
        transformer=StandardScaler(with_mean=True, with_std=True)
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def kfold_eval(df, feat_cols, target_col, mk_pipeline, n_splits=5):
    kf = KFold(n_splits=min(n_splits, max(2, len(df)//10)), shuffle=True, random_state=42)
    maes, r2s = [], []
    for tr, te in kf.split(df):
        X_tr, X_te = df.iloc[tr][feat_cols], df.iloc[te][feat_cols]
        y_tr, y_te = df.iloc[tr][target_col].astype(float).values, df.iloc[te][target_col].astype(float).values
        pipe = mk_pipeline()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
        r2s.append(r2_score(y_te, pred))
    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std":  float(np.std(maes)),
        "r2_mean":  float(np.mean(r2s)),
        "r2_std":   float(np.std(r2s)),
    }

def parity_plot(y_true, y_pred, path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual log10(σ)")
    plt.ylabel("Predicted log10(σ)")
    plt.title("Predicted vs Actual (non-Arrhenius, ElasticNetCV)")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Train ElasticNet-based non-Arrhenius model")
    ap.add_argument("--csv", default="example.csv")
    ap.add_argument("--model_out", default="nonarr_elastic_model.joblib")
    ap.add_argument("--plot_out", default="pred_vs_actual_nonarr_elastic.png")
    args = ap.parse_args()

    df = load_data(args.csv)
    target = "log(Conductivity(S/cm))"
    cat_cols = ["dopant_A", "dopant_B"]
    num_cols = ["T_K", "xA_mol", "xB_mol", "xV_mol", "density_matrix (g/mL)"]
    feat_cols = cat_cols + num_cols

    def mk_pipeline():
        return build_pipeline(cat_cols, num_cols)

    pipe = mk_pipeline()
    pipe.fit(df[feat_cols], df[target].astype(float).values)

    # Train metrics
    y = df[target].astype(float).values
    yhat = pipe.predict(df[feat_cols])
    train_mae = mean_absolute_error(y, yhat)
    train_r2  = r2_score(y, yhat)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train R² : {train_r2:.4f}")

    # Cross-validation metrics
    cv = kfold_eval(df, feat_cols, target, mk_pipeline, n_splits=5)
    print(f"CV MAE (mean±std): {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")
    print(f"CV R²  (mean±std): {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")

    # Plot and save
    parity_plot(y, yhat, args.plot_out)
    joblib.dump({"model": pipe, "schema": {"cat_cols": cat_cols, "num_cols": num_cols, "target": target},
                 "metrics": {"train": {"mae": float(train_mae), "r2": float(train_r2)}, "cv": cv}},
                args.model_out)
    print(f"Model saved to {args.model_out}")
    print(f"Plot  saved to {args.plot_out}")

if __name__ == "__main__":
    main()
