
# scripts/01_train_base.py
# -*- coding: utf-8 -*-
"""
Train 14 base models on sampled data:
  - Train on (X_tomtrain_s, y_tomtrain_s)
  - Validate on (X_val_s, y_val_s)
Outputs (validation only):
  - experiments/predictions/val/{model}_pred.csv
  - experiments/predictions/val/{model}_proba.csv
  - experiments/metrics/val_metrics.csv  (AUC, ACC, PREC, REC, F1)

Notes:
  - No OOF here (do it later).
  - No test predictions here (bootstrap later).
  - Optional libraries (xgboost/lightgbm/catboost/ngboost/tabnet) are loaded safely;
    if not installed, the corresponding model is skipped with a warning.
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ---------- optional imports (safe) ----------
OPT = {}
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    OPT["tabnet"] = True
except Exception:
    OPT["tabnet"] = False

try:
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from ngboost.scores import LogScore
    OPT["ngb"] = True
except Exception:
    OPT["ngb"] = False

try:
    import xgboost as xgb
    OPT["xgb"] = True
except Exception:
    OPT["xgb"] = False

try:
    import lightgbm as lgb
    OPT["lgbm"] = True
except Exception:
    OPT["lgbm"] = False

try:
    import catboost as cat
    OPT["cat"] = True
except Exception:
    OPT["cat"] = False


# ---------- paths (sampled datasets) ----------
DATA_X_TRAIN = "X_tomtrain_s.csv"
DATA_Y_TRAIN = "y_tomtrain_s.csv"

DATA_X_VAL = "X_val_s.csv"
DATA_Y_VAL = "y_val_s.csv"

# Test files are reserved for future bootstrap step (NOT used here)
DATA_X_TEST  = "X_ogtest_s.csv"
DATA_Y_TEST  = "y_ogtest_s.csv"

OUT_ROOT = "experiments"
VAL_PRED_DIR = os.path.join(OUT_ROOT, "predictions", "val")
METRIC_DIR = os.path.join(OUT_ROOT, "metrics")
for d in [VAL_PRED_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 1412

# ---------- model save dir (unchanged) ----------
MODELS_DIR = os.path.join(OUT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

import joblib
def _save_model(name, model):
    try:
        if name == "tabnet":
            # TabNet requires a base path WITHOUT .zip; it will write base + ".zip"
            base = os.path.join(MODELS_DIR, "tabnet_model")
            model.save_model(base)
            print(f"[Save] {name} -> {base}.zip")
        else:
            path = os.path.join(MODELS_DIR, f"{name}.joblib")
            joblib.dump(model, path)
            print(f"[Save] {name} -> {path}")
    except Exception as e:
        print(f"[WARN] save {name} failed: {e}")



# ---------- helpers ----------
def to_1d(y):
    if isinstance(y, (pd.DataFrame, pd.Series)):
        return y.values.ravel().astype(int)
    return np.asarray(y).ravel().astype(int)

def save_val_preds(model_name, y_pred, y_proba):
    pd.DataFrame(y_pred, columns=[f"{model_name}_pred"]).to_csv(
        os.path.join(VAL_PRED_DIR, f"{model_name}_pred.csv"), index=False
    )
    pd.DataFrame(y_proba, columns=[f"{model_name}_proba"]).to_csv(
        os.path.join(VAL_PRED_DIR, f"{model_name}_proba.csv"), index=False
    )

def compute_metrics(y_true, y_pred, y_proba=None):
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_pred))
    except Exception:
        out["auc"] = np.nan
    out["accuracy"]  = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    return out

def run_and_collect(name, fit_fun, pred_proba_fun, X_tr, y_tr, X_val, y_val):
    """Train on train; eval on val; save preds/metrics and SAVE MODEL once."""
    print(f"==> Training {name}")
    model = fit_fun(X_tr, y_tr)

    # Save the fitted model immediately (single-pass)
    _save_model(name, model)

    # --- TabNet: use wrapped pred to avoid device/shape issues ---
    if name == "tabnet":
        y_val_proba = proba_tabnet(model, X_val)
        y_val_pred  = pred_tabnet(model, X_val)
    else:
        y_val_proba = pred_proba_fun(model, X_val)
        y_val_pred  = (y_val_proba >= 0.5).astype(int) if name == "probit" else model.predict(X_val)

    save_val_preds(name, y_val_pred, y_val_proba)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
    return name, val_metrics


# ---------- load data ----------
X_tomtrain = pd.read_csv(DATA_X_TRAIN)
y_tomtrain = pd.read_csv(DATA_Y_TRAIN)
X_val = pd.read_csv(DATA_X_VAL)
y_val = pd.read_csv(DATA_Y_VAL)

y_tr_all = to_1d(y_tomtrain)
y_val = to_1d(y_val)

# (Intentionally NOT reading test set here to avoid accidental leakage)
# X_test = pd.read_csv(DATA_X_TEST)
# y_test = pd.read_csv(DATA_Y_TEST)


# ---------- model wrappers (14 models) ----------
def fit_lda(X, y):
    m = LinearDiscriminantAnalysis()
    m.fit(X, y)
    return m
def proba_lda(m, X): return m.predict_proba(X)[:, 1]

def fit_lr(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(random_state=RANDOM_STATE))
    ])
    pipe.fit(X, y)
    return pipe
def proba_lr(m, X): return m.predict_proba(X)[:, 1]

def fit_nb_iso(X, y):
    base = GaussianNB()
    iso = CalibratedClassifierCV(base, cv=2, method="isotonic")
    iso.fit(X, y)
    return iso
def proba_nb(m, X): return m.predict_proba(X)[:, 1]

def fit_svc_linear(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svc", SVC(random_state=RANDOM_STATE,probability=True, kernel="linear",max_iter=10000))
    ])
    pipe.fit(X, y)
    return pipe
def proba_svc(m, X): return m.predict_proba(X)[:, 1]

def fit_knn(X, y):
    m = KNeighborsClassifier(n_jobs=-1)
    m.fit(X, y)
    return m
def proba_knn(m, X): return m.predict_proba(X)[:, 1]

def fit_dt(X, y):
    m = DecisionTreeClassifier(random_state=RANDOM_STATE)
    m.fit(X, y)
    return m
def proba_dt(m, X): return m.predict_proba(X)[:, 1]

def fit_rf(X, y):
    m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X, y)
    return m
def proba_rf(m, X): return m.predict_proba(X)[:, 1]

def fit_xgb(X, y):
    if not OPT["xgb"]:
        raise ImportError("xgboost is not installed; skipped.")
    m = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X, y)
    return m
def proba_xgb(m, X): return m.predict_proba(X)[:, 1]

def fit_lgbm(X, y):
    if not OPT["lgbm"]:
        raise ImportError("lightgbm is not installed; skipped.")
    m = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X, y)
    return m
def proba_lgbm(m, X): return m.predict_proba(X)[:, 1]

def fit_cat(X, y):
    if not OPT["cat"]:
        raise ImportError("catboost is not installed; skipped.")
    m = cat.CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False)
    m.fit(X, y)
    return m
def proba_cat(m, X): return m.predict_proba(X)[:, 1]

def fit_mlp(X, y):
    m = MLPClassifier(random_state=RANDOM_STATE)
    m.fit(X, y)
    return m
def proba_mlp(m, X): return m.predict_proba(X)[:, 1]

# ---- NGB (robust with fallbacks) ----
def _to_1d_int(y):
    import numpy as np, pandas as pd
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    y = np.asarray(y).ravel().astype(int)
    return y

def fit_ngb_robust(X, y, random_state=1412):
    y = _to_1d_int(y)
    trials = [
        dict(natural_gradient=True,  n_estimators=400, learning_rate=0.03, minibatch_frac=0.5, col_sample=0.8),
        dict(natural_gradient=False, n_estimators=600, learning_rate=0.025, minibatch_frac=0.5, col_sample=0.8),
        dict(natural_gradient=False, n_estimators=800, learning_rate=0.02)  # simplest, smaller step
    ]
    last_err = None
    for cfg in trials:
        try:
            m = NGBClassifier(
                Dist=Bernoulli, Score=LogScore,
                random_state=random_state, verbose=False, **cfg
            )
            m.fit(X, y)
            print(f"[NGB] fitted with cfg={cfg}")
            return m
        except Exception as e:
            last_err = e
            print(f"[WARN] NGB config failed ({cfg}): {e}")
    raise RuntimeError(f"NGB failed after fallbacks: {last_err}")

def pred_ngb(m, X):
    # hard labels
    return (m.predict_proba(X)[:, 1] >= 0.5).astype(int)

def proba_ngb(m, X):
    return m.predict_proba(X)[:, 1]


# ---- TabNet (robust inference via save->load CPU fallback) ----

def _ensure_float32(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    out = df.copy()
    out[num] = out[num].astype(np.float32)
    return out

def _sanitize_df(df: pd.DataFrame, name: str):
    num = df.select_dtypes(include=[np.number]).columns
    if df.isna().any().any():
        print(f"[WARN] {name} has NaN; filling 0.")
        df[num] = df[num].fillna(0)
    arr = df[num].to_numpy()
    bad = ~np.isfinite(arr)
    if bad.any():
        print(f"[WARN] {name} has Inf/-Inf; setting 0.")
        arr[bad] = 0
        df[num] = arr

def fit_tabnet(X, y, Xv, yv, random_state=1412):
    # sanitize & cast to float32
    _sanitize_df(X, "X_train"); _sanitize_df(Xv, "X_val")
    X = _ensure_float32(X); Xv = _ensure_float32(Xv)
    y = _to_1d_int(y); yv = _to_1d_int(yv)

    # auto device
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    m = TabNetClassifier(device_name=device, seed=random_state)
    m.fit(
        X_train=X.values, y_train=y,
        eval_set=[(Xv.values, yv)],
        eval_metric=["auc"],
        max_epochs=100, patience=10,
        batch_size=65536, virtual_batch_size=2048,
        drop_last=False
    )
    return m

def proba_tabnet(m: TabNetClassifier, X: pd.DataFrame):
    Xc = X.copy()
    _sanitize_df(Xc, "X_for_tabnet_pred")
    Xc = _ensure_float32(Xc)
    try:
        return m.predict_proba(Xc.values)[:, 1]
    except Exception as e:
        print(f"[WARN] TabNet GPU predict failed: {e}. Falling back via save->load on CPU.")
        # save -> reload on CPU (this reliably avoids device mismatch)
        tmp_base = os.path.join("experiments", "artifacts", "tabnet_tmp_model")
        os.makedirs(os.path.dirname(tmp_base), exist_ok=True)
        try:
            # save_model expects path WITHOUT .zip; it will append .zip
            m.save_model(tmp_base)
            m_cpu = TabNetClassifier()
            m_cpu.load_model(tmp_base + ".zip")  # loads on CPU by default
            return m_cpu.predict_proba(Xc.values)[:, 1]
        except Exception as e2:
            raise RuntimeError(f"TabNet CPU reload also failed: {e2}")

def pred_tabnet(m, X):
    return (proba_tabnet(m, X) >= 0.5).astype(int)


# Probit (statsmodels) with a thin wrapper to expose predict_proba-like API
def fit_probit(X, y):
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    mod = sm.Probit(y, Xc)
    res = mod.fit(disp=0)
    class _ProbitWrapper:
        def __init__(self, res):
            self.res = res
        def predict(self, X_):
            Xc_ = sm.add_constant(X_, has_constant="add")
            p = self.res.predict(Xc_)
            return (p >= 0.5).astype(int)
        def predict_proba(self, X_):
            Xc_ = sm.add_constant(X_, has_constant="add")
            p = self.res.predict(Xc_)
            return np.column_stack([1 - p, p])
    return _ProbitWrapper(res)
def proba_probit(m, X): return m.predict_proba(X)[:, 1]


# ---------- orchestrate ----------
results = []

def try_run(name, fit_fn, proba_fn, needs_val_in_fit=False):
    """
    Train & evaluate one model.
    For TabNet we need validation inside fit() via eval_set, hence needs_val_in_fit=True.
    """
    try:
        if name == "tabnet" and needs_val_in_fit:
            mname, met = run_and_collect(
                name,
                lambda Xt, yt: fit_tabnet(Xt, yt, X_val, y_val),
                proba_tabnet,
                X_tomtrain, y_tr_all, X_val, y_val
            )
        else:
            mname, met = run_and_collect(
                name, fit_fn, proba_fn,
                X_tomtrain, y_tr_all, X_val, y_val
            )
        results.append({"model": mname, **met})
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")


# Order: 14 models
try_run("lda",     fit_lda,     proba_lda)
try_run("lr",      fit_lr,      proba_lr)
try_run("probit",  fit_probit,  proba_probit)
try_run("knn",     fit_knn,     proba_knn)
try_run("svc",     fit_svc_linear,     proba_svc)
try_run("nb",      fit_nb_iso,  proba_nb)
try_run("dt",      fit_dt,      proba_dt)
try_run("rf",      fit_rf,      proba_rf)
try_run("xgb",     fit_xgb,     proba_xgb)
try_run("lgbm",    fit_lgbm,    proba_lgbm)
try_run("cat",     fit_cat,     proba_cat)
try_run("ngb",     fit_ngb_robust,     proba_ngb)
try_run("mlp",     fit_mlp,     proba_mlp)

try_run("tabnet",  fit_tabnet,  proba_tabnet, needs_val_in_fit=True)

print("VAL shapes:", X_val.shape, y_val.shape)
print("Types:", X_val.dtypes.unique())

# ---------- save metrics ----------
if results:
    dfm = pd.DataFrame(results)
    dfm.to_csv(os.path.join(METRIC_DIR, "val_metrics.csv"), index=False)
    print("\nValidation metrics (sorted by AUC):")
    print(dfm.sort_values("auc", ascending=False).round(4).to_string(index=False))
else:
    print("No model finished successfully. Please check dependencies and data.")

# ---------- single-pass saving ----------
print("\n[Save] Single-pass: models already saved during training.")
