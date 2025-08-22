# -*- coding: utf-8 -*-
"""
Evaluate 14 base models (original features) + 3 meta models (L1 features) on the TEST set.
Outputs (to experiments/level2/):
  - base_bootstrap_test.csv
  - meta_bootstrap_test.csv
  - combined_bootstrap_test.csv
  - combined_auc_ranking_test.csv
  - combined_bootstrap_summary_test.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# ---------------- paths ----------------
ROOT    = Path(r"D:\sh-gpc")
LV1_DIR = ROOT / "experiments" / "level1"
LAB_DIR = ROOT / "experiments" / "labels"
ART_DIR = ROOT / "experiments" / "artifacts"
MODEL_DIR = ROOT / "experiments" / "models"
OUT_DIR = ROOT / "experiments" / "level2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- files ---
BASE_X_TEST = ROOT / "X_ogtest_s.csv"
BASE_Y_TEST = LAB_DIR / "y_ogtest_s.csv"
if not BASE_Y_TEST.exists():
    BASE_Y_TEST = ROOT / "y_ogtest_s.csv"  # fallback

X_TRAIN_L1 = LV1_DIR / "L1_train_X.csv"
Y_TRAIN_L1 = LAB_DIR / "y_tomtrain_s.csv"
X_TEST_L1  = LV1_DIR / "L1_test_X.csv"
Y_TEST_L1  = LAB_DIR / "y_ogtest_s.csv"
GP_SEL_CSV = ART_DIR / "gp_selected_test.csv"

# ---------------- config ----------------
RANDOM_STATE = 1412
np.random.seed(RANDOM_STATE)

N_BOOT_BASE  = 20          # base models bootstrap
N_BOOT_META  = 20          # meta models bootstrap
SUB_FRAC_BASE = 1.00       # base models sampling fraction
SUB_FRAC_META = 0.30       # meta models sampling fraction

# GP runtime
CLIP_RANGE = 10.0
GP_FIXED_THRESH = 0.5

# ---------------- utils ----------------
def compute_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray) -> Dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = np.nan
    return dict(
        AUC=auc,
        Accuracy=float(accuracy_score(y_true, y_pred)),
        Precision=float(precision_score(y_true, y_pred, zero_division=0)),
        Recall=float(recall_score(y_true, y_pred, zero_division=0)),
        F1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

def _metrics(y_true: np.ndarray,
                       y_pred_hard: np.ndarray,
                       y_sigmoid: np.ndarray) -> Dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_sigmoid))
    except Exception:
        auc = np.nan
    return dict(
        AUC=auc,
        Accuracy=float(accuracy_score(y_true, y_pred_hard)),
        Precision=float(precision_score(y_true, y_pred_hard, zero_division=0)),
        Recall=float(recall_score(y_true, y_pred_hard, zero_division=0)),
        F1=float(f1_score(y_true, y_pred_hard, zero_division=0)),
    )

# ---- TabNet safe loader / predictor ----
def load_tabnet_model():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        z = MODEL_DIR / "tabnet_model.zip"
        if not z.exists():
            raise FileNotFoundError(f"Missing TabNet zip: {z}")
        m = TabNetClassifier()
        m.load_model(str(z))  
        return m
    except Exception as e:
        raise RuntimeError(f"TabNet load failed: {e}")

def proba_tabnet(model, X: pd.DataFrame) -> np.ndarray:
    import numpy as np
    Xc = X.copy()
    num = Xc.select_dtypes(include=[np.number]).columns
    if Xc.isna().any().any():
        Xc[num] = Xc[num].fillna(0)
    arr = Xc[num].to_numpy()
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = 0
        Xc[num] = arr
    Xc[num] = Xc[num].astype(np.float32)
    return model.predict_proba(Xc.values)[:, 1]

# ---------------- base models loader ----------------
BASE_NAMES = [
    "lda","lr","probit","knn","svc","nb","dt","rf","xgb","lgbm","cat","ngb","mlp","tabnet"
]

def load_base_models() -> Dict[str, object]:
    models = {}
    for name in BASE_NAMES:
        try:
            if name == "tabnet":
                models[name] = load_tabnet_model()
            else:
                path = MODEL_DIR / f"{name}.joblib"
                if not path.exists():
                    print(f"[WARN] missing model: {path}")
                    continue
                models[name] = joblib.load(path)
        except Exception as e:
            print(f"[WARN] failed loading {name}: {e}")
    return models

# ---------------- single-model bootstrap (original features) ----------------
def bootstrap_base(models: Dict[str, object],
                   X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    out = []
    Xr = X_test.reset_index(drop=True)
    yr = y_test.reset_index(drop=True).astype(int)

    def pred_hard(name, model, Xs):
        if name == "tabnet":
            p = proba_tabnet(model, Xs)
            return (p >= 0.5).astype(int)
        if name == "probit":
            return model.predict(Xs).astype(int)
        y = model.predict(Xs)
        return (np.asarray(y).reshape(-1) > 0.5).astype(int)

    for name, model in models.items():
        print(f"\n[BASE] {name} bootstrap {N_BOOT_BASE}x ...")
        frac = 0.5 if name == "knn" else SUB_FRAC_BASE
        n = max(1, int(frac * len(Xr)))
        for i in range(1, N_BOOT_BASE + 1):
            try:
                idx = resample(np.arange(len(Xr)), replace=True, n_samples=n,
                               random_state=RANDOM_STATE + i)
                Xs = Xr.iloc[idx]; ys = yr.iloc[idx]
                if len(np.unique(ys)) < 2:  
                    raise ValueError("Only one class present in y_true")
                y_pred = pred_hard(name, model, Xs)
                met = compute_metrics(ys, y_pred)
                out.append({"Family":"base","Model":name,"Iteration":i, **met})
                print(f"  ✓ {name} {i}/{N_BOOT_BASE}")
            except Exception as e:
                print(f"  ⚠ {name} {i} failed: {e}")

    df = pd.DataFrame(out)
    df.to_csv(OUT_DIR / "base_bootstrap_test.csv", index=False, encoding="utf-8")
    return df

# ---------------- meta training + bootstrap (L1 features) ----------------
# Optional XGB
HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
    print("[WARN] xgboost not installed; meta_XGB will be skipped.")

# GP expression runtime
import ast
NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "tanh": np.tanh, "sigmoid": lambda z: 1/(1+np.exp(-z)),
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where,
    "pow": np.power
}
def compile_gp_callable(expr_raw: str, p: int):
    expr = expr_raw.strip().rstrip(";").replace("^", "**")
    code = compile(ast.parse(expr, mode="eval"), "<gp_expr>", "eval")
    def _f(Xdf: pd.DataFrame):
        env = {}
        for j in range(p):
            env[f"X{j}"] = Xdf.iloc[:, j].to_numpy(dtype=float)
            env[f"x{j}"] = env[f"X{j}"]
        env.update(NP_FUNCS)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            raw = eval(code, {"__builtins__": {}}, env)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        raw[~np.isfinite(raw)] = 0.0
        raw = np.clip(raw, -CLIP_RANGE, CLIP_RANGE)
        y_pred  = (raw > GP_FIXED_THRESH).astype(int)
        y_proba = 1.0 / (1.0 + np.exp(-raw))
        return y_pred, y_proba
    return _f

def train_meta_and_bootstrap(X_tr_L1, y_tr_L1, X_te_L1, y_te_L1) -> pd.DataFrame:
    recs = []

    # meta_LR
    meta_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000).fit(X_tr_L1, y_tr_L1)
    def pair_lr(X): return meta_lr.predict(X).astype(int), None

    # meta_XGB
    meta_xgb = None
    if HAS_XGB:
        meta_xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1
        ).fit(X_tr_L1, y_tr_L1)
        def pair_xgb(X): return meta_xgb.predict(X).astype(int), None

    # meta_GP
    sel = pd.read_csv(GP_SEL_CSV)
    if sel.empty or "expression" not in sel.columns:
        raise ValueError(f"No expression found in {GP_SEL_CSV}")
    gp_expr = str(sel.loc[0, "expression"])
    gp_fn = compile_gp_callable(gp_expr, p=X_tr_L1.shape[1])
    print("[META][GP] expression:", gp_expr)
    def pair_gp(X): return gp_fn(X)  # returns (y_pred_hard, y_proba_sigmoid)

    # bootstrap on L1 test
    Xr = X_te_L1.reset_index(drop=True)
    yr = y_te_L1.reset_index(drop=True).astype(int)

    def do_boot(name, pair_fn, use_gp=False):
        for i in range(1, N_BOOT_META + 1):
            try:
                n = max(1, int(SUB_FRAC_META * len(Xr)))
                idx = resample(np.arange(len(Xr)), replace=True, n_samples=n,
                               random_state=RANDOM_STATE + 2000 + i)
                Xs = Xr.iloc[idx]; ys = yr.iloc[idx]
                if len(np.unique(ys)) < 2:
                    raise ValueError("Only one class present in y_true")
                y_pred, y_proba = pair_fn(Xs)
                if use_gp:
                    met = _metrics(ys, y_pred, y_proba)
                else:
                    met = compute_metrics(ys, y_pred)
                recs.append({"Family":"meta","Model":name,"Iteration":i, **met})
                print(f"  ✓ {name} {i}/{N_BOOT_META}")
            except Exception as e:
                print(f"  ⚠ {name} {i} failed: {e}")

    print("\n[META] bootstrap ...")
    do_boot("meta_LR",  pair_lr,  use_gp=False)
    if HAS_XGB:
        do_boot("meta_XGB", pair_xgb, use_gp=False)
    do_boot("meta_GP",  pair_gp,  use_gp=True)

    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / "meta_bootstrap_test.csv", index=False, encoding="utf-8")
    return df

# ---------------- main ----------------
def main():
    # --- load base test (original features) ---
    if not BASE_X_TEST.exists() or not BASE_Y_TEST.exists():
        raise FileNotFoundError(f"Missing base test files: {BASE_X_TEST} / {BASE_Y_TEST}")
    X_test = pd.read_csv(BASE_X_TEST)
    y_test = pd.read_csv(BASE_Y_TEST).squeeze("columns")

    # --- load base models and evaluate on test ---
    base_models = load_base_models()
    if not base_models:
        raise RuntimeError("No base models loaded from experiments/models/.")
    df_base = bootstrap_base(base_models, X_test, y_test)

    # --- load L1 and train/eval meta on L1 test ---
    if not (X_TRAIN_L1.exists() and Y_TRAIN_L1.exists() and X_TEST_L1.exists() and Y_TEST_L1.exists()):
        raise FileNotFoundError("Missing L1 files for meta: check level1 and labels folders.")
    X_tr_L1 = pd.read_csv(X_TRAIN_L1)
    y_tr_L1 = pd.read_csv(Y_TRAIN_L1).squeeze("columns").astype(int)
    X_te_L1 = pd.read_csv(X_TEST_L1)
    y_te_L1 = pd.read_csv(Y_TEST_L1).squeeze("columns").astype(int)

    df_meta = train_meta_and_bootstrap(X_tr_L1, y_tr_L1, X_te_L1, y_te_L1)

    # --- combine + ranking ---
    df_all = pd.concat([df_base, df_meta], ignore_index=True)
    df_all.to_csv(OUT_DIR / "combined_bootstrap_test.csv", index=False, encoding="utf-8")

    summary = (df_all
               .groupby(["Family","Model"])[["AUC","Accuracy","Precision","Recall","F1"]]
               .agg(["mean","std"]))
    summary.to_csv(OUT_DIR / "combined_bootstrap_summary_test.csv", encoding="utf-8")

    ranking = (df_all.groupby("Model")["AUC"]
               .mean().sort_values(ascending=False).reset_index())
    ranking.to_csv(OUT_DIR / "combined_auc_ranking_test.csv", index=False, encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n[SUMMARY]\n", summary.round(4))
        print("\n[RANKING by mean AUC]\n", ranking)

if __name__ == "__main__":
    main()
