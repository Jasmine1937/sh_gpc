

# -*- coding: utf-8 -*-
"""

Pipeline:
1) Load selected base models (after diversity) -> experiments/artifacts/selected_base_models.json
2) Rebuild Level-1 dataset with 5-fold CV (OOF for train, mean-of-fold-proba for test)
3) Rank L1 features by SHAP-LR; remove from low to high, until 2 models remain
4) For each subset size k (k = full...2), train & evaluate on TEST:
   - meta-LR 
   - meta-XGB 
   - meta-GP ; write .txt for Java GP; can auto-run java or print cmd

Outputs under experiments/ablation:
  - L1_train_X.csv, L1_test_X.csv (rebuilt)
  - shap_lr_ranking.csv
  - results_all.csv (all subsets × meta models, sorted by AUC desc)

ROOT default: D:\sh-gpc
"""

import os, re, ast, json, subprocess, warnings, traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------- Paths & Config -------------------
ROOT = Path(r"D:\sh-gpc")

X_TRAIN_CSV = ROOT / "X_tomtrain_s.csv"
Y_TRAIN_CSV = ROOT / "y_tomtrain_s.csv"
X_TEST_CSV  = ROOT / "X_ogtest_s.csv"
Y_TEST1     = ROOT / "experiments" / "labels" / "y_ogtest_s.csv"
Y_TEST2     = ROOT / "y_ogtest_s.csv"

SEL_JSON    = ROOT / "experiments" / "artifacts" / "selected_base_models.json"

ABL_DIR     = ROOT / "experiments" / "ablation"
L1_DIR      = ABL_DIR
GP_IN_DIR   = ABL_DIR / "gp_inputs"
GP_OUT_DIR  = ABL_DIR / "gp_outputs"
for d in [ABL_DIR, GP_IN_DIR, GP_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1412
N_SPLITS = 5

# Java GP settings
RUN_JAVA   = False  # <-
JAVA_CP    = str(ROOT / "src")  # classpath
JAVA_CLASS = "sh_gpc"

# GP eval settings
CLIP_RANGE      = 10.0
GP_FIXED_THRESH = 0.5   
EPS = 1e-8

np.random.seed(RANDOM_STATE)

# ------------------- Optional libs (safe import) -------------------
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ------------------- Base model factory (match 01_train_base) -------------------
def make_model(name: str):
    """Return (model, proba_supported: bool). Names must match those produced in selection."""
    n = name.strip().lower()
    if n == "lda":
        return LinearDiscriminantAnalysis(), True
    if n == "lr":
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(random_state=RANDOM_STATE))])
        return pipe, True
    if n == "svc":
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("svc", SVC(random_state=RANDOM_STATE, probability=True, kernel="linear", max_iter=10000))])
        return pipe, True
    if n == "knn":
        return KNeighborsClassifier(n_jobs=-1), True
    if n == "nb":
        base = GaussianNB()
        iso  = CalibratedClassifierCV(base, cv=2, method="isotonic")
        return iso, True
    if n == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE), True
    if n == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), True
    if n == "xgb":
        if not OPT["xgb"]:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1), True
    if n == "lgbm":
        if not OPT["lgbm"]:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1), True
    if n == "cat":
        if not OPT["cat"]:
            raise ImportError("catboost not installed")
        return cat.CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False), True
    if n == "ngb":
        if not OPT["ngb"]:
            raise ImportError("ngboost not installed")
        # robust-ish config
        return NGBClassifier(Dist=Bernoulli, Score=LogScore,
                             natural_gradient=False,
                             n_estimators=600, learning_rate=0.025,
                             minibatch_frac=0.5, col_sample=0.8,
                             random_state=RANDOM_STATE, verbose=False), True
    if n == "mlp":
        return MLPClassifier(random_state=RANDOM_STATE), True
    if n == "tabnet":
        if not OPT["tabnet"]:
            raise ImportError("tabnet not installed")
        # device auto inside fit
        return TabNetClassifier(seed=RANDOM_STATE), True
    
    raise ValueError(f"Unknown/unsupported model name for CV: {name}")

def predict_proba1(model, X: pd.DataFrame) -> np.ndarray:
    try:
        proba = model.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=float)
    except Exception:
        # decision_function -> sigmoid
        dec = model.decision_function(X).astype(float)
        return 1.0 / (1.0 + np.exp(-dec))

# ------------------- Build Level-1 via 5-fold CV -------------------
def build_level1_cv(selected: List[str], X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, n_splits: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return L1_train_X, L1_test_X with columns {model}_L1; OOF for train, mean-of-fold proba for test.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    L1_train = pd.DataFrame(index=X_train.index)
    L1_test  = pd.DataFrame(index=X_test.index)

    for name in selected:
        try:
            mdl, _ = make_model(name)
        except Exception as e:
            print(f"[WARN] skip {name}: {e}")
            continue

        oof = np.zeros(len(X_train), dtype=float)
        test_fold_preds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            Xva      = X_train.iloc[va_idx]

            # TabNet 支持 eval_set
            if name.lower() == "tabnet" and OPT["tabnet"]:
                Xt = Xtr.values.astype(np.float32)
                yt = ytr.values.astype(int)
                Xv = Xva.values.astype(np.float32)
                yv = y_train.iloc[va_idx].values.astype(int)
                mdl.fit(X_train=Xt, y_train=yt,
                        eval_set=[(Xv, yv)],
                        eval_metric=["auc"],
                        max_epochs=100, patience=10,
                        batch_size=65536, virtual_batch_size=2048,
                        drop_last=False)
                oof[va_idx] = mdl.predict_proba(Xv)[:, 1]
                test_fold_preds.append(mdl.predict_proba(X_test.values.astype(np.float32))[:, 1])
            else:
                mdl.fit(Xtr, ytr)
                oof[va_idx] = predict_proba1(mdl, Xva)
                test_fold_preds.append(predict_proba1(mdl, X_test))

        L1_train[f"{name}_L1"] = oof
        L1_test[f"{name}_L1"]  = np.mean(np.vstack(test_fold_preds), axis=0)

    L1_train.to_csv(L1_DIR / "L1_train_X.csv", index=False)
    L1_test.to_csv(L1_DIR / "L1_test_X.csv", index=False)
    print(f"[L1] Rebuilt and saved: {L1_DIR/'L1_train_X.csv'}, {L1_DIR/'L1_test_X.csv'}")
    return L1_train, L1_test

# ------------------- SHAP-LR ranking -------------------
def shap_lr_rank(L1_train_X: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Return feature names ranked by importance (desc). Fallback to |coef|.
    """
    try:
        import shap
        comb = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
        comb.fit(L1_train_X, y_train)
        try:
            expl = shap.LinearExplainer(comb, L1_train_X, feature_dependence="independent")
            shap_vals = expl.shap_values(L1_train_X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            I = np.mean(np.abs(shap_vals), axis=0)
        except Exception:
            I = np.abs(comb.coef_.ravel())
        imp = pd.DataFrame({"feature": L1_train_X.columns, "importance": I})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(ABL_DIR / "shap_lr_ranking.csv", index=False)
        print(f"[RANK] SHAP-LR ranking saved -> {ABL_DIR/'shap_lr_ranking.csv'}")
        return imp["feature"].tolist()
    except Exception as e:
        print(f"[WARN] SHAP-LR failed, fallback to |coef|: {e}")
        comb = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
        comb.fit(L1_train_X, y_train)
        I = np.abs(comb.coef_.ravel())
        imp = pd.DataFrame({"feature": L1_train_X.columns, "importance": I})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(ABL_DIR / "shap_lr_ranking.csv", index=False)
        return imp["feature"].tolist()

# ------------------- Meta models & metrics -------------------
def metrics_pack(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None, hard_auc: bool) -> Dict:

    auc = roc_auc_score(y_true, y_pred if hard_auc or (y_score is None) else y_score)
    return dict(
        AUC=auc,
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0)
    )

def fit_meta_lr(Xtr: pd.DataFrame, ytr: pd.Series):
    m = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
    m.fit(Xtr, ytr)
    return m

def fit_meta_xgb(Xtr: pd.DataFrame, ytr: pd.Series):
    if not OPT["xgb"]:
        raise ImportError("xgboost not installed for meta-XGB")
    m = xgb.XGBClassifier(
        random_state=RANDOM_STATE, n_estimators=400,
        learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9,
        n_jobs=-1
    )
    m.fit(Xtr, ytr)
    return m

# ------------------- GP glue (Java) -------------------
NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "tanh": np.tanh, "sigmoid": lambda z: 1/(1+np.exp(-z)),
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where,
    "pow": np.power
}

def write_gp_txt(Xy: pd.DataFrame, path: Path):
    """
    Xy: columns = [X..., y]  (X at left; y last)
    header: "<p> 0 0 0 <n>"
    """
    p = Xy.shape[1] - 1
    n = Xy.shape[0]
    header = f"{p} 0 0 0 {n}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        Xy.to_csv(f, header=False, index=False, sep=" ")

def run_java_gp(input_txt: Path, output_txt: Path):
    cmd = ["java", "-cp", JAVA_CP, JAVA_CLASS, str(input_txt)]
    print("[JAVA]", " ".join(cmd), f"> {output_txt}")
    with open(output_txt, "w", encoding="utf-8") as fout:
        subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, check=False)

def normalize_expr(s: str) -> str:
    return s.strip().rstrip(";").replace("^", "**")

def parse_gp_output(txt_path: Path) -> List[Tuple[int, str]]:
    """
      GEN,<gen>,<expr>
      Generation=N / Best Individual: <expr>
    """
    if not txt_path.exists():
        return []
    lines = txt_path.read_text("utf-8", errors="ignore").splitlines()
    out = []
    pat_csv = re.compile(r'^\s*GEN\s*,\s*(\d+)\s*,\s*(.+)$')
    pat_gen = re.compile(r'^\s*Generation\s*=\s*(\d+)\b', re.IGNORECASE)
    pat_best = re.compile(r'^\s*Best Individual:\s*(.+)$', re.IGNORECASE)
    cur = None
    fallback = 0
    for ln in lines:
        m = pat_csv.match(ln)
        if m:
            out.append((int(m.group(1)), normalize_expr(m.group(2))))
            continue
        m = pat_gen.search(ln)
        if m:
            cur = int(m.group(1)); continue
        m = pat_best.match(ln)
        if m:
            expr = normalize_expr(m.group(1))
            if cur is None:
                out.append((fallback, expr)); fallback += 1
            else:
                out.append((cur, expr)); cur = None
    keep = {}
    for g, e in out:
        keep.setdefault(g, e)
    return [(g, keep[g]) for g in sorted(keep.keys())]

def compile_gp_callable_one_based(expr_raw: str, col_names: List[str]):
    expr = normalize_expr(expr_raw)
    code = compile(ast.parse(expr, mode="eval"), "<gp_expr>", "eval")
    idxs = [int(m.group(1)) for m in re.finditer(r"\b[Xx](\d+)\b", expr)]
    if idxs and max(idxs) > len(col_names):
        raise ValueError(f"GP expr refers X{max(idxs)} but only {len(col_names)} features.")
    def _f(X_df: pd.DataFrame):
        env = {}
        for j, col in enumerate(col_names, start=1):
            arr = X_df[[col]].to_numpy(dtype=float).ravel()
            env[f"X{j}"] = arr; env[f"x{j}"] = arr
        env.update(NP_FUNCS)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            raw = eval(code, {"__builtins__": {}}, env)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        raw[~np.isfinite(raw)] = 0.0
        raw = np.clip(raw, -CLIP_RANGE, CLIP_RANGE)
        return raw
    return _f, expr

# ------------------- Main ablation -------------------
def main():
    # 0) Load selection & data
    if not SEL_JSON.exists():
        raise FileNotFoundError(f"Missing selection: {SEL_JSON}")
    selected = json.loads(SEL_JSON.read_text("utf-8"))
    if not isinstance(selected, list) or len(selected) < 2:
        raise RuntimeError("Need at least 2 selected models for ablation.")

    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(Y_TRAIN_CSV).squeeze("columns").astype(int)
    Xte = pd.read_csv(X_TEST_CSV)

    if Y_TEST1.exists():
        yte = pd.read_csv(Y_TEST1).squeeze("columns").astype(int)
    elif Y_TEST2.exists():
        yte = pd.read_csv(Y_TEST2).squeeze("columns").astype(int)
    else:
        raise FileNotFoundError("y_ogtest_s.csv not found (checked experiments/labels/ and root).")

    # 1) Rebuild L1 via 5-fold CV
    L1_train_X, L1_test_X = build_level1_cv(selected, Xtr, ytr, Xte, n_splits=N_SPLITS)

    # 2) SHAP-LR ranking (desc), then ablation from low->high (i.e., reverse order)
    rank_desc = shap_lr_rank(L1_train_X, ytr)  # high -> low
    rank_asc  = rank_desc[::-1]                # low -> high (first removed)
    print("[RANK] High->Low:", rank_desc)
    print("[ABLATION ORDER] Low->High:", rank_asc)

    # 3) Iterative ablation: keep k features (k from full to 2), removing the earliest in rank_asc
    removal_order = rank_asc.copy()
    kept_features = rank_desc.copy()

    records = []

    for k in range(len(kept_features), 1, -1):
        # Subset columns: take the LAST k of (rank_desc), which equals removing the first (len-rank_desc - k) low-importance ones
        to_keep = rank_desc[:k]  # top-k by importance
        Xtr_k = L1_train_X[to_keep].copy()
        Xte_k = L1_test_X[to_keep].copy()

        print(f"\n[ABL] k={k}, keep={to_keep}")

        # -------- meta-LR --------
        try:
            meta_lr = fit_meta_lr(Xtr_k, ytr)
            y_pred_lr = meta_lr.predict(Xte_k)
            # AUC with HARD labels for non-GP models
            rec = metrics_pack(yte, y_pred_lr, None, hard_auc=True)
            rec.update({"SubsetK": k, "Meta": "meta_LR", "Kept": ";".join(to_keep)})
            records.append(rec)
        except Exception as e:
            print(f"[WARN] meta-LR (k={k}) failed: {e}")

        # -------- meta-XGB --------
        try:
            meta_xgb = fit_meta_xgb(Xtr_k, ytr)
            y_pred_xgb = meta_xgb.predict(Xte_k)
            rec = metrics_pack(yte, y_pred_xgb, None, hard_auc=True)
            rec.update({"SubsetK": k, "Meta": "meta_XGB", "Kept": ";".join(to_keep)})
            records.append(rec)
        except Exception as e:
            print(f"[WARN] meta-XGB (k={k}) failed: {e}")

        # -------- meta-GP --------
        try:
            Xy_train = pd.concat([Xtr_k.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
            gp_in = GP_IN_DIR / f"L1_train_k{k}.txt"
            write_gp_txt(Xy_train, gp_in)

            gp_out = GP_OUT_DIR / f"gp_k{k}.txt"
            if RUN_JAVA:
                run_java_gp(gp_in, gp_out)
            else:
                print(f"[CMD] Manual Run：\n  cd {ROOT}\n  java -cp {JAVA_CP} {JAVA_CLASS} {gp_in} > {gp_out}\n  （then run this script again for evaluation）")

            exprs = parse_gp_output(gp_out)
            if not exprs:
                raise RuntimeError(f"No expressions parsed for k={k}.")
            gen, expr = exprs[-1]

            gp_fn, expr_clean = compile_gp_callable_one_based(expr, to_keep)
            raw_test = gp_fn(Xte_k)  # raw
            pre_test = 1.0 / (1.0 + np.exp(-raw_test))
            y_pred_gp = (raw_test > GP_FIXED_THRESH).astype(int)

            rec = metrics_pack(yte, y_pred_gp, pre_test, hard_auc=False) 
            rec.update({"SubsetK": k, "Meta": "meta_GP", "Kept": ";".join(to_keep), "GP_gen": int(gen), "GP_expr": expr_clean})
            records.append(rec)
        except Exception as e:
            print(f"[WARN] meta-GP (k={k}) failed: {e}")
            traceback.print_exc()

    df_all = pd.DataFrame(records)
    if not df_all.empty:
        df_all = df_all.sort_values("AUC", ascending=False).reset_index(drop=True)
        out_csv = ABL_DIR / "results_all.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"\n[RESULTS] Saved -> {out_csv}")
        print(df_all.head(15).to_string(index=False))
    else:
        print("[RESULTS] No results collected. Check errors above.")

if __name__ == "__main__":
    main()
