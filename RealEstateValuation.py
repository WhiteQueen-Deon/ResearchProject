#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCA/CBR implementation for the UCI Real Estate Valuation dataset (id=477).

Implements the Quantitative Comparative Approach (QCA) with:
- Sorting-quantile importance.
- Stepwise decomposition to fit dimensionless adjustment coefficient models k_i(x_i).
- Candidate regression forms: linear, quadratic, logarithmic, exponential, power.
- Geographic coordinate model: second-order polynomial in (lat, lon).
- Case distance & Gaussian-like case weight with effect radius (delta).
- Adjusted prices and weighted average integration over comparative cases.

The default column names follow UCI dataset (after normalization):
X1: transaction date (decimal year)
X2: house age (year)
X3: distance to nearest MRT (meter)
X4: number of convenience stores in walking circle
X5: latitude
X6: longitude
Y : house price of unit area (10k NTD per ping)
"""

import sys
import math
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathlib import Path

# For plotting artifacts (optional)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- Data Loading ---------------------------

def load_uci_real_estate(local_excel: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features X and target y. Try ucimlrepo first; fallback to a local Excel."""
    # Try ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
        ds = fetch_ucirepo(id=477)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        return normalize_columns(X), y
    except Exception as e:
        pass

    # Fallback to Excel
    candidates = []
    if local_excel:
        candidates.append(Path(local_excel))
    candidates += [
        Path.cwd() / "Real estate valuation data set.xlsx",
        Path("./Real estate valuation data set.xlsx"),
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_excel(p)
            df = df.rename(columns={
                'X1 transaction date': 'X1',
                'X2 house age': 'X2',
                'X3 distance to the nearest MRT station': 'X3',
                'X4 number of convenience stores': 'X4',
                'X5 latitude': 'X5',
                'X6 longitude': 'X6',
                'Y house price of unit area': 'Y'
            })
            if 'Y' not in df.columns:
                y_col = df.columns[-1]
                df = df.rename(columns={y_col: 'Y'})
            y = df['Y'].astype(float)
            X = df.drop(columns=['Y']).astype(float)
            return normalize_columns(X), y

    raise FileNotFoundError("Dataset not found. Install `ucimlrepo` with internet, or supply Excel via --excel")


def normalize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected UCI short column names exist. Drop any id column."""
    X = X.copy()
    rename_map = {
        'X1 transaction date': 'X1',
        'X2 house age': 'X2',
        'X3 distance to the nearest MRT station': 'X3',
        'X4 number of convenience stores': 'X4',
        'X5 latitude': 'X5',
        'X6 longitude': 'X6',
    }
    X = X.rename(columns=rename_map)
    for id_col in ['No', 'no', 'ID', 'id']:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
    return X


# --------------------------- Utilities ---------------------------

def ten_quantile_bins(x: np.ndarray) -> np.ndarray:
    """Return bin index 0..9 for ten-quantile slicing. Stable with ties."""
    q = np.quantile(x, np.linspace(0, 1, 11))
    for i in range(1, len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-9
    return np.clip(np.digitize(x, q[1:-1], right=True), 0, 9)


def safe_log(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def safe_pow(x: np.ndarray, b: float, eps: float = 1e-9) -> np.ndarray:
    return np.power(np.clip(x, eps, None), b)


# --------------------- Adjustment Coefficient Models ---------------------

@dataclass
class RegForm:
    name: str
    fit: Callable[[np.ndarray, np.ndarray], Tuple]
    predict: Callable[[np.ndarray, Tuple], np.ndarray]


def fit_linear(xi, ki):
    A = np.vstack([xi, np.ones_like(xi)]).T
    a, b = np.linalg.lstsq(A, ki, rcond=None)[0]
    return (a, b)

def pred_linear(x, params):
    a, b = params
    return a * x + b

def fit_quadratic(xi, ki):
    A = np.vstack([xi**2, xi, np.ones_like(xi)]).T
    a, b, c = np.linalg.lstsq(A, ki, rcond=None)[0]
    return (a, b, c)

def pred_quadratic(x, params):
    a, b, c = params
    return a * x**2 + b * x + c

def fit_logarithmic(xi, ki):
    zl = safe_log(xi)
    A = np.vstack([zl, np.ones_like(zl)]).T
    a, b = np.linalg.lstsq(A, ki, rcond=None)[0]
    return (a, b)

def pred_logarithmic(x, params):
    a, b = params
    return a * safe_log(x) + b

def fit_exponential(xi, ki):
    eps = 1e-9
    y = np.log(np.clip(ki, eps, None))
    A = np.vstack([xi, np.ones_like(xi)]).T
    b, ln_a = np.linalg.lstsq(A, y, rcond=None)[0]
    a = np.exp(ln_a)
    return (a, b)

def pred_exponential(x, params):
    a, b = params
    return a * np.exp(b * x)

def fit_power(xi, ki):
    eps = 1e-9
    y = np.log(np.clip(ki, eps, None))
    z = safe_log(xi)
    A = np.vstack([z, np.ones_like(z)]).T
    b, ln_a = np.linalg.lstsq(A, y, rcond=None)[0]
    a = np.exp(ln_a)
    return (a, b)

def pred_power(x, params):
    a, b = params
    return a * safe_pow(x, b)


REG_FORMS = [
    RegForm("linear", fit_linear, pred_linear),
    RegForm("quadratic", fit_quadratic, pred_quadratic),
    RegForm("logarithmic", fit_logarithmic, pred_logarithmic),
    RegForm("exponential", fit_exponential, pred_exponential),
    RegForm("power", fit_power, pred_power),
]


@dataclass
class OneFactorModel:
    name: str
    form: str
    params: Tuple


class QCAAdjustmentFitter:
    def __init__(self, order: Optional[List[str]] = None, coord_cols: Tuple[str, str] = ("X5", "X6")):
        self.order = order
        self.coord_cols = coord_cols
        self.models: Dict[str, OneFactorModel] = {}
        self.Ybar: float = 1.0

    @staticmethod
    def _portion_means(x: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bins = ten_quantile_bins(x)
        means_x = np.array([x[bins == b].mean() for b in range(10)])
        means_k = np.array([k[bins == b].mean() for b in range(10)])
        for i in range(10):
            if np.isnan(means_x[i]) or np.isnan(means_k[i]):
                means_x[i] = np.nanmean(means_x)
                means_k[i] = np.nanmean(means_k)
        return means_x, means_k

    @staticmethod
    def _choose_best_form(xs: np.ndarray, ks: np.ndarray) -> Tuple[str, Tuple]:
        best = None
        best_sse = np.inf
        for rf in REG_FORMS:
            try:
                params = rf.fit(xs, ks)
                pred = rf.predict(xs, params)
                sse = float(np.nanmean((pred - ks) ** 2))
                if np.isfinite(sse) and sse < best_sse:
                    best_sse = sse
                    best = (rf.name, params)
            except Exception:
                continue
        if best is None:
            a = 0.0
            b = float(np.nanmean(ks))
            best = ("linear", (a, b))
        return best

    def fit(self, X: pd.DataFrame, y: pd.Series, importance_order: Optional[List[str]] = None) -> "QCAAdjustmentFitter":
        X = X.copy()
        y = y.astype(float).values
        n = len(y)
        self.Ybar = float(np.mean(y))
        k = y / self.Ybar

        if importance_order is None:
            if self.order is None:
                exclude = set(self.coord_cols)
                base = [c for c in X.columns if c not in exclude]
                importance_order = base
            else:
                importance_order = self.order

        prod_hat = np.ones(n)
        for idx, col in enumerate(importance_order):
            if idx == 0:
                kij = k
            else:
                kij = k / prod_hat

            xs, ks = self._portion_means(X[col].values, kij)
            form_name, params = self._choose_best_form(xs, ks)
            self.models[col] = OneFactorModel(col, form_name, params)

            pred_func = next(rf for rf in REG_FORMS if rf.name == form_name).predict
            prod_hat *= np.clip(pred_func(X[col].values, params), 1e-6, None)

        # Geographic coordinate residual
        if all(c in X.columns for c in self.coord_cols):
            kij_geo = k / prod_hat
            N = X[self.coord_cols[0]].values
            E = X[self.coord_cols[1]].values
            A = np.vstack([
                np.ones_like(N),
                N, E,
                N**2, E**2, N*E
            ]).T
            coeffs, *_ = np.linalg.lstsq(A, kij_geo, rcond=None)
            self.models["GEO"] = OneFactorModel("GEO", "poly2D", tuple(coeffs))
        else:
            self.models["GEO"] = OneFactorModel("GEO", "poly2D", tuple([1,0,0,0,0,0]))

        return self

    def k_hat(self, name: str, x: np.ndarray) -> np.ndarray:
        m = self.models[name]
        if m.form == "poly2D":
            N = x[:, 0]; E = x[:, 1]
            c0, c1, c2, c3, c4, c5 = m.params
            return c0 + c1*N + c2*E + c3*N**2 + c4*E**2 + c5*N*E
        else:
            rf = next(rf for rf in REG_FORMS if rf.name == m.form)
            return rf.predict(x, m.params)


# --------------------- Distance, Weights, and Prediction ---------------------

def sorting_quantile_importance(X: pd.DataFrame, y: np.ndarray, cols: List[str]) -> Dict[str, float]:
    imp = {}
    Ybar = float(np.mean(y))
    for c in cols:
        bins = ten_quantile_bins(X[c].values)
        means = np.array([y[bins == b].mean() if np.any(bins == b) else Ybar for b in range(10)])
        var = np.sum((means - Ybar)**2)
        imp[c] = float(var)
    return imp


def factor_weights(X: pd.DataFrame, y: np.ndarray, cols: List[str], method: int, geo_weight: float = 3.0) -> Dict[str, float]:
    imp = sorting_quantile_importance(X, y, cols)
    vals = np.array(list(imp.values()))
    if method in (1, 3):
        w = np.sqrt(vals + 1e-12)
    else:
        w = vals
    w = w / (np.min(w) + 1e-12)
    weights = {c: float(wi) for c, wi in zip(imp.keys(), w)}
    if "X5" in X.columns and "X6" in X.columns:
        weights["X5"] = geo_weight
        weights["X6"] = geo_weight
    return weights


def case_distance(x_t: np.ndarray, x_j: np.ndarray, std: np.ndarray, weights: np.ndarray) -> float:
    num = np.sum(weights * ((x_t - x_j) / std) ** 2)
    den = np.sum(weights)
    return float(np.sqrt(num / (den + 1e-12)))


def case_weight(distance: float, radius: float) -> float:
    return float(np.exp(- (distance / radius) ** 2))


class QCARegressor:
    def __init__(self, weight_method: int = 1, radius: float = 1.5, geo_weight: float = 3.0):
        self.weight_method = weight_method
        self.radius = radius
        self.geo_weight = geo_weight
        self.adjuster = QCAAdjustmentFitter()
        self.weights: Dict[str, float] = {}
        self.X_cases: Optional[pd.DataFrame] = None
        self.y_cases: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_order: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QCARegressor":
        X = X.copy()
        y = y.astype(float).values
        non_geo = [c for c in X.columns if c not in ("X5", "X6")]
        imp = sorting_quantile_importance(X, y, non_geo)
        order = sorted(non_geo, key=lambda c: imp[c], reverse=True)

        self.adjuster.fit(X, pd.Series(y), importance_order=order)

        self.weights = factor_weights(X, y, non_geo, method=self.weight_method, geo_weight=self.geo_weight)
        self.feature_order = list(X.columns)
        self.std_ = X[self.feature_order].astype(float).std(axis=0).replace(0, 1.0).values

        self.X_cases = X[self.feature_order].astype(float).reset_index(drop=True)
        self.y_cases = y.copy()
        return self

    def _k_vector(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        out = {}
        for col in [c for c in X.columns if c not in ("X5", "X6")]:
            out[col] = self.adjuster.k_hat(col, X[col].values)
        if ("X5" in X.columns) and ("X6" in X.columns):
            geo_in = X[["X5", "X6"]].values
            out["GEO"] = self.adjuster.k_hat("GEO", geo_in)
        else:
            out["GEO"] = np.ones(len(X))
        return out

    def predict_row(self, x_t: np.ndarray) -> float:
        assert self.X_cases is not None and self.y_cases is not None and self.std_ is not None
        Xc = self.X_cases.values
        yc = self.y_cases
        std = self.std_
        weights_vec = np.array([self.weights.get(c, 1.0) for c in self.feature_order])

        df_t = pd.DataFrame([x_t], columns=self.feature_order)
        k_t = self._k_vector(df_t)
        k_cases = self._k_vector(self.X_cases)

        Kj_prod = np.ones(len(yc))
        for key, vt in k_t.items():
            vj = k_cases[key]
            denom = np.clip(vj, 1e-9, None)
            Kj_prod *= np.clip(vt[0] / denom, 1e-6, 1e6)

        Y_adj = yc * Kj_prod

        D = np.array([case_distance(x_t, Xc[j], std, weights_vec) for j in range(len(yc))])
        F = np.array([case_weight(D[j], self.radius) for j in range(len(yc))])
        if np.all(F == 0):
            return float(np.mean(Y_adj))
        return float(np.sum(F * Y_adj) / np.sum(F))

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        X_new = X_new[self.feature_order].astype(float)
        preds = np.array([self.predict_row(X_new.values[i]) for i in range(len(X_new))])
        return preds


# --------------------------- CLI / Demo ---------------------------

def demo_train_evaluate(local_excel: Optional[str] = None, radius_list=(1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 100.0)) -> None:
    X, y = load_uci_real_estate(local_excel)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=42)

    rows = []
    best = (None, np.inf, None)
    for wm in (1, 2, 3, 4):
        for r in radius_list:
            model = QCARegressor(weight_method=wm, radius=r, geo_weight=3.0).fit(Xtr, pd.Series(ytr))
            yp = model.predict(Xte)
            rmse = float(np.sqrt(np.mean((yte - yp) ** 2)))
            mae = float(np.mean(np.abs(yte - yp)))
            r2 = 1.0 - float(np.sum((yte - yp) ** 2) / np.sum((yte - np.mean(yte)) ** 2))
            rows.append(dict(weight_method=wm, radius=r, RMSE=rmse, MAE=mae, R2=r2))
            if rmse < best[1]:
                best = (model, rmse, dict(weight_method=wm, radius=r))

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.to_csv("./qca_grid_scores.csv", index=False)

    plt.figure()
    df_pivot = df.pivot(index="radius", columns="weight_method", values="RMSE")
    df_pivot.plot(marker="o")
    plt.title("QCA RMSE vs radius & weight_method (lower is better)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig("./qca_rmse_grid.png", dpi=150)
    plt.close()

    best_model = best[0]
    yp_best = best_model.predict(Xte)
    pd.DataFrame({"y_true": yte, "y_pred": yp_best}).to_csv("./qca_predictions_best.csv", index=False)

    meta = {
        "best_params": best[2],
        "weights": best_model.weights,
        "models": {k: dict(form=v.form, params=list(v.params)) for k, v in best_model.adjuster.models.items()},
        "Ybar": best_model.adjuster.Ybar,
        "feature_order": best_model.feature_order,
    }
    with open("./qca_fitted_models.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Artifacts written to ./:")
    print(" - qca_grid_scores.csv")
    print(" - qca_rmse_grid.png")
    print(" - qca_predictions_best.csv")
    print(" - qca_fitted_models.json")


def main():
    import argparse
    p = argparse.ArgumentParser(description="QCA/CBR for UCI Real Estate Valuation")
    p.add_argument("--excel", type=str, default=None, help="Path to 'Real estate valuation data set.xlsx'")
    p.add_argument("--demo", action="store_true", help="Run a quick holdout evaluation & grid over radius/weight_method")
    args = p.parse_args()

    warnings.filterwarnings("ignore")

    if args.demo:
        demo_train_evaluate(args.excel)
    else:
        X, y = load_uci_real_estate(args.excel)
        model = QCARegressor(weight_method=1, radius=1.5, geo_weight=3.0).fit(X, y)
        yhat = model.predict(X)
        pd.DataFrame({"y_true": y, "y_pred": yhat}).to_csv("./qca_predictions_insample.csv", index=False)

        meta = {
            "weights": model.weights,
            "models": {k: dict(form=v.form, params=list(v.params)) for k, v in model.adjuster.models.items()},
            "Ybar": model.adjuster.Ybar,
            "feature_order": model.feature_order,
        }
        with open("./qca_fitted_models.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("Fitted QCA on full dataset. Artifacts:")
        print(" - qca_predictions_insample.csv")
        print(" - qca_fitted_models.json")


if __name__ == "__main__":
    main()
