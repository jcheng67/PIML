#!/usr/bin/env python3
"""
Train a basic Physics-Informed ML (PIML) model for ionic conductivity.

Inputs (CSV columns, required):
- T_K                         : temperature in Kelvin (float)
- dopant_A                    : primary dopant symbol/name (string)
- xA_mol                      : primary dopant concentration (molar fraction, float)
- dopant_B                    : co-dopant symbol/name (string; use "None" if absent)
- xB_mol                      : co-dopant concentration (molar fraction, float; 0 if absent)
- xV_mol                      : vacancy concentration (molar fraction, float)
- density_matrix (g/mL)       : matrix density in g/mL (float)
- log(Conductivity(S/cm))     : target (log10 of conductivity in S/cm)

Model:
- Physics base: Arrhenius form in log10 space → linear regression of log10(sigma) ~ a + b*(1/T_K).
- ML residuals: Gradient Boosting Regressor over composition/defect features; predicts the residual to add to the physics base.

Outputs:
- Saves model bundle to piml_model.joblib
- Saves parity plot to pred_vs_actual.png
- Prints train and CV metrics
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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
    return df.copy()

def fit_arrhenius_base(df: pd.DataFrame):
    inv_T = 1.0 / df["T_K"].astype(float).values.reshape(-1, 1)
    y = df["log(Conductivity(S/cm))"].astype(float).values
    lin = LinearRegression().fit(inv_T, y)
    base_pred = lin.predict(inv_T)
    residual = y - base_pred
    return lin, base_pred, residual, y

def fit_residual_ml(df: pd.DataFrame, residual: np.ndarray):
    cat_cols = ["dopant_A", "dopant_B"]
    num_cols = ["xA_mol", "xB_mol", "xV_mol", "density_matrix (g/mL)"]
    X = df[cat_cols + num_cols].copy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    gbr = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("gbr", gbr)])
    pipe.fit(X, residual)
    ml_residual_pred = pipe.predict(X)
    return pipe, ml_residual_pred, X, cat_cols, num_cols

def evaluate(y_true, base_pred, ml_pred):
    yhat = base_pred + ml_pred
    return {
        "mae": float(mean_absolute_error(y_true, yhat)),
        "r2": float(r2_score(y_true, yhat)),
        "yhat": yhat
    }

def cross_validate(df: pd.DataFrame, n_splits=5):
    kf = KFold(n_splits=min(n_splits, max(2, len(df)//10)), shuffle=True, random_state=42)
    maes, r2s = [], []
    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        lin_k, base_tr, resid_tr, y_tr = fit_arrhenius_base(df_tr)
        pipe_k, ml_tr, X_tr, cat_cols, num_cols = fit_residual_ml(df_tr, resid_tr)
        inv_T_te = 1.0 / df_te["T_K"].astype(float).values.reshape(-1,1)
        base_te = lin_k.predict(inv_T_te)
        X_te = df_te[cat_cols + num_cols].copy()
        ml_te = pipe_k.predict(X_te)
        yhat_te = base_te + ml_te
        y_te = df_te["log(Conductivity(S/cm))"].astype(float).values
        maes.append(mean_absolute_error(y_te, yhat_te))
        r2s.append(r2_score(y_te, yhat_te))
    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
    }

def save_parity_plot(y_true, y_pred, path="pred_vs_actual.png"):
    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual log10(σ)")
    plt.ylabel("Predicted log10(σ)")
    plt.title("Predicted vs Actual (Arrhenius base + ML residuals)")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Train a PIML model for log10 conductivity")
    ap.add_argument("--csv", default="example.csv")
    ap.add_argument("--model_out", default="piml_model.joblib")
    ap.add_argument("--plot_out", default="pred_vs_actual.png")
    args = ap.parse_args()

    df = load_data(args.csv)
    lin, base_pred, residual, y = fit_arrhenius_base(df)
    pipe, ml_pred, X, cat_cols, num_cols = fit_residual_ml(df, residual)
    metrics = evaluate(y, base_pred, ml_pred)

    print(f"Train MAE: {metrics['mae']:.4f}")
    print(f"Train R² : {metrics['r2']:.4f}")

    cv = cross_validate(df)
    print(f"CV MAE (mean±std): {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")
    print(f"CV R²  (mean±std): {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")

    save_parity_plot(y, metrics["yhat"], path=args.plot_out)
    bundle = {
        "linear_arrhenius": lin,
        "residual_pipeline": pipe,
        "feature_cols_cat": cat_cols,
        "feature_cols_num": num_cols,
        "metrics": {"train": metrics, "cv": cv}
    }
    joblib.dump(bundle, args.model_out)
    print(f"Model saved to {args.model_out}")
    print(f"Plot saved to {args.plot_out}")

if __name__ == "__main__":
    main()
