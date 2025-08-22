
# -*- coding: utf-8 -*-
"""

Build Level-1 (stacking) feature matrices from selected base models.

Inputs:
  - experiments/artifacts/selected_base_models.json  (from 01a; final kept models)
  - X_tomtrain_s.csv, y_tomtrain_s.csv
  - X_ogtest_s.csv        (test features; no labels to avoid leakage)

Outputs (features only, no labels):
  - experiments/level1/L1_train_X.csv   (OOF probabilities for meta-model training)
  - experiments/level1/L1_test_X.csv    (K-fold averaged test probabilities)
  - experiments/level1/folds.csv
  - experiments/level1/models_used.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- paths ----------------
DATA_X_TRAIN = "X_tomtrain_s.csv"
DATA_Y_TRAIN = "y_tomtrain_s.csv"
DATA_X_TEST  = "X_ogtest_s.csv"   

ART_DIR   = os.path.join("experiments", "artifacts")
L1_DIR    = os.path.join("experiments", "level1")
os.makedirs(L1_DIR, exist_ok=True)

SELECTED_JSON = os.path.join(ART_DIR, "selected_base_models.json")

RANDOM_STATE = 1412
N_SPLITS = 5

# -------------- optional deps -------------
OPT = {}
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

try:
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from ngboost.scores import LogScore
    OPT["ngb"] = True
except Exception:
    OPT["ngb"] = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    OPT["tabnet"] = True
except Exception:
    OPT["tabnet"] = False


# --------- TabNet thin wrapper ---------
class TabNetSkWrapper:
    """A sklearn-like wrapper over TabNetClassifier"""
    def __init__(self, seed=RANDOM_STATE, device="auto",
                 max_epochs=100, patience=10,
                 batch_size=65536, virtual_batch_size=2048):
        dev = "cuda"
        if device == "cpu":
            dev = "cpu"
        elif device == "auto":
            try:
                import torch
                if not torch.cuda.is_available():
                    dev = "cpu"
            except Exception:
                dev = "cpu"
        self._cfg = dict(seed=seed, device_name=dev)
        self._fit_cfg = dict(
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=2048,
            drop_last=False, eval_metric=["auc"]
        )
        self.model = TabNetClassifier(**self._cfg)

    @staticmethod
    def _ensure_float32(X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            num = X.select_dtypes(include=[np.number]).columns
            X[num] = X[num].astype(np.float32)
            if X.isna().any().any():
                X[num] = X[num].fillna(0)
            arr = X[num].to_numpy()
            bad = ~np.isfinite(arr)
            if bad.any():
                arr[bad] = 0
            return arr.astype(np.float32)
        arr = np.asarray(X, dtype=np.float32)
        arr[~np.isfinite(arr)] = 0
        return arr

    @staticmethod
    def _to_1d_int(y) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        return np.asarray(y).ravel().astype(int)

    def fit(self, X, y, eval_set=None):
        Xn = self._ensure_float32(X)
        yn = self._to_1d_int(y)
        eval_set_proc = None
        if eval_set is not None and len(eval_set) > 0:
            eval_set_proc = []
            for (Xe, ye) in eval_set:
                Xe_ = self._ensure_float32(Xe)
                ye_ = self._to_1d_int(ye)
                eval_set_proc.append((Xe_, ye_))
        self.model.fit(
            X_train=Xn, y_train=yn,
            eval_set=eval_set_proc if eval_set_proc is not None else [],
            **self._fit_cfg
        )
        return self

    def predict_proba(self, X):
        Xn = self._ensure_float32(X)
        try:
            p = self.model.predict_proba(Xn)
        except Exception as e:
            print(f"[WARN] TabNet GPU predict failed: {e}. Falling back to CPU reload.")
            base = os.path.join(L1_DIR, "tabnet_tmp_model")
            try:
                self.model.save_model(base)  # will create base+".zip"
                m_cpu = TabNetClassifier()   # default CPU
                m_cpu.load_model(base + ".zip")
                p = m_cpu.predict_proba(Xn)
            except Exception as e2:
                raise RuntimeError(f"TabNet CPU reload also failed: {e2}")
        p = np.asarray(p)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        elif p.ndim == 2 and p.shape[1] == 1:
            p = np.column_stack([1 - p[:, 0], p[:, 0]])
        return p


# --------- estimator factory (12 models excluding lda/probit) ----------
def make_estimator(name: str):
    name = name.strip().lower()

    # linear models
    if name == "lr":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(random_state=RANDOM_STATE))
        ])
    if name == "svc":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("svc", SVC(random_state=RANDOM_STATE, probability=True, kernel="linear", max_iter=10000))
        ])

    # neighbors / trees / forests
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    if name == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # naive bayes (isotonic calibrated)
    if name in ("nb", "iso"):
        base = GaussianNB()
        return CalibratedClassifierCV(base, cv=2, method="isotonic")

    # gradient boosting families
    if name == "xgb":
        if not OPT["xgb"]:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "lgbm":
        if not OPT["lgbm"]:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "cat":
        if not OPT["cat"]:
            raise ImportError("catboost not installed")
        return cat.CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False)
    if name == "ngb":
        if not OPT["ngb"]:
            raise ImportError("ngboost not installed")
        return NGBClassifier(
            Dist=Bernoulli, Score=LogScore,
            natural_gradient=False, n_estimators=600,
            learning_rate=0.025, minibatch_frac=0.5, col_sample=0.8,
            verbose=False, random_state=RANDOM_STATE
        )

    # neural net
    if name == "mlp":
        return MLPClassifier(random_state=RANDOM_STATE)

    # tabnet
    if name == "tabnet":
        if not OPT["tabnet"]:
            raise ImportError("pytorch-tabnet not installed")
        return TabNetSkWrapper(seed=RANDOM_STATE)

    # ignore lda/probit even if accidentally present
    if name in ("lda", "probit"):
        raise ValueError(f"Estimator '{name}' is excluded from Level-1.")

    raise ValueError(f"Unknown estimator key: {name}")


# --------- core: build Level-1 via KFold OOF ----------
def build_level1_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    base_model_names: List[str],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      L1_train_X: (n_train, m) OOF probs
      L1_test_X:  (n_test,  m) mean probs across folds
      folds_df:   (n_train, 1) fold assignment
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int)
    X_test  = pd.DataFrame(X_test).reset_index(drop=True)

    m = len(base_model_names)
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    L1_train_X = pd.DataFrame(np.zeros((n_train, m)), columns=[f"{k}_L1" for k in base_model_names])
    L1_test_X  = pd.DataFrame(np.zeros((n_test,  m)), columns=[f"{k}_L1" for k in base_model_names])
    folds_df   = pd.DataFrame(index=np.arange(n_train), columns=["fold"], dtype=int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for j, key in enumerate(base_model_names):
        print(f"==> L1 building: {key}")
        test_pred_accum = np.zeros(n_test, dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
            folds_df.loc[va_idx, "fold"] = fold

            try:
                model = make_estimator(key)
            except Exception as e:
                print(f"[WARN] skip model {key}: {e}")
                continue

            # TabNet: use eval_set；
            if key == "tabnet" and hasattr(model, "fit"):
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            else:
                model.fit(X_tr, y_tr)

            proba_va = model.predict_proba(X_va)
            if proba_va.ndim == 1:
                p1_va = proba_va
            else:
                p1_va = proba_va[:, -1]
            L1_train_X.iloc[va_idx, j] = p1_va

            proba_te = model.predict_proba(X_test)
            if proba_te.ndim == 1:
                p1_te = proba_te
            else:
                p1_te = proba_te[:, -1]
            test_pred_accum += p1_te / n_splits

        L1_test_X.iloc[:, j] = test_pred_accum

    return L1_train_X, L1_test_X, folds_df


def main():
    if not os.path.exists(SELECTED_JSON):
        raise FileNotFoundError(f"Missing selected models JSON: {SELECTED_JSON}")
    with open(SELECTED_JSON, "r", encoding="utf-8") as f:
        selected = json.load(f)

    if not isinstance(selected, list) or len(selected) == 0:
        raise RuntimeError("No selected base models found in selected_base_models.json")

    # out lda/probit
    selected = [k.strip().lower() for k in selected if k.strip().lower() not in ("lda", "probit")]

    X_tr = pd.read_csv(DATA_X_TRAIN)
    y_tr = pd.read_csv(DATA_Y_TRAIN).squeeze("columns")
    X_te = pd.read_csv(DATA_X_TEST)

    usable = []
    for k in selected:
        try:
            _ = make_estimator(k)  # probe
            usable.append(k)
        except Exception as e:
            print(f"[WARN] model '{k}' is not usable in this env: {e}")
    if len(usable) == 0:
        raise RuntimeError("No usable models from selection. Please install missing libs or adjust selection.")

    print(f"[L1] models used: {usable}")

    L1_train_X, L1_test_X, folds_df = build_level1_oof(
        X_tr, y_tr, X_te, base_model_names=usable,
        n_splits=N_SPLITS, random_state=RANDOM_STATE
    )


    L1_train_X.to_csv(os.path.join(L1_DIR, "L1_train_X.csv"), index=False)
    L1_test_X.to_csv(os.path.join(L1_DIR, "L1_test_X.csv"), index=False)
    folds_df.to_csv(os.path.join(L1_DIR, "folds.csv"), index=False)
    with open(os.path.join(L1_DIR, "models_used.json"), "w", encoding="utf-8") as f:
        json.dump(usable, f, indent=2, ensure_ascii=False)

    print(f"[L1] Done. Shapes: L1_train_X={L1_train_X.shape}, L1_test_X={L1_test_X.shape}")
    print(f"[L1] Files written to: {L1_DIR}")

if __name__ == "__main__":
    main()
