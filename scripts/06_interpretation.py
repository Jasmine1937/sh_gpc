
# -*- coding: utf-8 -*-
"""

Perform SHAP decision visualization for the L-1 models
-Feed the L1 vector of into the optimal GP expression, showing the substitution process and final output.
-Save the results to experiments/explain/.


"""

import os
import re
import ast
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------- paths ----------------
ROOT = Path(r"D:\sh-gpc")
DATA_X_TRAIN = ROOT / "X_tomtrain_s.csv"
DATA_Y_TRAIN = ROOT / "y_tomtrain_s.csv"

DATA_X_TEST  = ROOT / "X_ogtest_s.csv"
PREF_Y_TEST1 = ROOT / "experiments" / "labels" / "y_ogtest_s.csv"
PREF_Y_TEST2 = ROOT / "y_ogtest_s.csv"

L1_TEST_X    = ROOT / "experiments" / "level1" / "L1_test_X.csv"
SEL_JSON     = ROOT / "experiments" / "artifacts" / "selected_base_models.json"
GP_SEL_CSV   = ROOT / "experiments" / "artifacts" / "gp_selected_test.csv"

MODELS_DIR   = ROOT / "experiments" / "models"
OUT_DIR      = ROOT / "experiments" / "explain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- config ----------------
SAMPLE_IDX      = 0        # index of sample
RANDOM_STATE    = 1412
BG_SIZE         = 200      
CLIP_RANGE      = 10.0     
GP_FIXED_THRESH = 0.5      
EPS = 1e-8                 
np.random.seed(RANDOM_STATE)

# ---------------- GP env ----------------
NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "tanh": np.tanh, "sigmoid": lambda z: 1/(1+np.exp(-z)),
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where,
    "pow": np.power
}

def _load_y_test():
    if PREF_Y_TEST1.exists():
        return pd.read_csv(PREF_Y_TEST1).squeeze("columns").astype(int)
    if PREF_Y_TEST2.exists():
        return pd.read_csv(PREF_Y_TEST2).squeeze("columns").astype(int)
    raise FileNotFoundError(f"Cannot find y_ogtest_s.csv at {PREF_Y_TEST1} or {PREF_Y_TEST2}")


def load_base_model(name: str):
    name = name.strip().lower()
    if name == "tabnet":
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            m = TabNetClassifier()
            base = str(MODELS_DIR / "tabnet_model")  # .zip 自动附加
            m.load_model(base + ".zip")
            return m
        except Exception as e:
            print(f"[WARN] load tabnet failed: {e}")
            return None
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"[WARN] model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] load {name} failed: {e}")
        return None

def predict_proba1(model, X: pd.DataFrame) -> np.ndarray:
    try:
        proba = model.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=float)
    except Exception:
        try:
            dec = model.decision_function(X).astype(float)
            return 1.0 / (1.0 + np.exp(-dec))
        except Exception as e:
            raise RuntimeError(f"model has no proba/decision_function: {e}")

def model_logit_output(model, X: pd.DataFrame) -> np.ndarray:
    try:
        p = predict_proba1(model, X)
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))
    except Exception:
        dec = model.decision_function(X).astype(float)
        return dec

def resolve_l1_columns(selected, l1_columns):
    cols = list(l1_columns)
    lower_map = {c.lower(): c for c in cols}

    def _find_one(m):
        m = m.lower()
        cands = [m, f"{m}_l1", f"{m}_oof", f"{m}_proba", f"{m}_prob", f"{m}_predict", f"{m}_pred"]
        for cand in cands:
            if cand in lower_map:
                return lower_map[cand]
        for lc, orig in lower_map.items():
            if m in lc:
                return orig
        return None

    resolved, missing = [], []
    for m in selected:
        hit = _find_one(m)
        if hit is None: missing.append(m)
        else: resolved.append(hit)

    print("\n[L1 columns available]:")
    print(", ".join(cols))
    if missing:
        print(f"[WARN] L1 column not found for: {missing}")

    if resolved:
        mapping_df = pd.DataFrame({
            "Selected model": [m for m in selected if _find_one(m) is not None],
            "Matched L1 column": resolved
        })
        print("\n[Selected → L1 matched columns]:")
        print(mapping_df.to_string(index=False))
    return resolved

# ------------ SHAP： KernelExplainer------------
def shap_explain_one_logit(model, X_bg: pd.DataFrame, x_row: pd.Series, feature_names, model_name: str):
    import shap
    save_path = OUT_DIR / f"decision_logit_{model_name}_idx{SAMPLE_IDX}.png"


    f = lambda X: model_logit_output(model, pd.DataFrame(X, columns=feature_names))

    Xbg = X_bg.sample(min(BG_SIZE, len(X_bg)), random_state=RANDOM_STATE)
    x1  = x_row.values.reshape(1, -1)

    expl = shap.KernelExplainer(f, Xbg.to_numpy())
    shap_vec = expl.shap_values(x1, nsamples="auto")
    if isinstance(shap_vec, list):
        shap_vec = np.asarray(shap_vec)[0]
    shap_vec = np.asarray(shap_vec).reshape(-1)
    base_val = expl.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])

    # decision plot（logit）
    plt.figure(figsize=(10, 4))
    try:
        shap.decision_plot(
            base_val, shap_vec,
            features=x_row.values,
            feature_names=feature_names,
            show=False
        )
    except Exception:
        shap.decision_plot(base_val, shap_vec, show=False)
    plt.title(f"{model_name} — SHAP decision (logit)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.show()

    return float(base_val), shap_vec

# ------------ GP：1-based（X1..Xk） ------------
def compile_gp_callable_one_based(expr_raw: str, col_names):
    expr = expr_raw.strip().rstrip(";").replace("^", "**")
    code = compile(ast.parse(expr, mode="eval"), "<gp_expr>", "eval")
    idxs = [int(m.group(1)) for m in re.finditer(r"\b[Xx](\d+)\b", expr)]
    if idxs and max(idxs) > len(col_names):
        raise ValueError(f"GP expression references X{max(idxs)} but only {len(col_names)} L1 features are available.")

    def _f(xrow_df: pd.DataFrame):
        env = {}
        for j, col in enumerate(col_names, start=1):
            arr = xrow_df[[col]].to_numpy(dtype=float).ravel()
            env[f"X{j}"] = arr
            env[f"x{j}"] = arr
        env.update(NP_FUNCS)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            raw = eval(code, {"__builtins__": {}}, env)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        raw[~np.isfinite(raw)] = 0.0
        raw = np.clip(raw, -CLIP_RANGE, CLIP_RANGE)

        raw_val  = float(raw.ravel()[0])
        prob_val = 1.0 / (1.0 + np.exp(-raw_val))
        pred_val = int(raw_val > GP_FIXED_THRESH)
        return raw_val, prob_val, pred_val

    return _f, expr  


def show_gp_substitution(expr_str: str, xi_vals: dict, decimals=6) -> str:

    def _repl(m):
        k = f"X{int(m.group(1))}"
        v = xi_vals.get(k, np.nan)
        return f"{v:.{decimals}f}"
    expr_sub = re.sub(r"\bX(\d+)\b", _repl, expr_str)
    return expr_sub

# ---------------- main ----------------
def main():
    print(f"[paths] ROOT = {ROOT}")

    if not SEL_JSON.exists():
        raise FileNotFoundError(f"Missing: {SEL_JSON}")
    selected = json.loads(SEL_JSON.read_text(encoding="utf-8"))
    if not isinstance(selected, list) or len(selected) == 0:
        raise RuntimeError("selected_base_models.json is empty.")
    print("[Selected base models]:", selected)

    X_train = pd.read_csv(DATA_X_TRAIN)
    y_train = pd.read_csv(DATA_Y_TRAIN).squeeze("columns").astype(int)
    X_test  = pd.read_csv(DATA_X_TEST)
    y_test  = _load_y_test()

    if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(X_test):
        raise IndexError(f"SAMPLE_IDX out of range: {SAMPLE_IDX} (len={len(X_test)})")
    xrow = X_test.iloc[SAMPLE_IDX, :]
    y_true = int(y_test.iloc[SAMPLE_IDX])


    bg = X_train.sample(min(len(X_train), 2000), random_state=RANDOM_STATE)

    for name in selected:
        print(f"\n==> SHAP for base model: {name}")
        mdl = load_base_model(name)
        if mdl is None:
            print(f"[WARN] skip {name} (model not loaded).")
            continue
        try:
            _base, _sv = shap_explain_one_logit(
                mdl, bg, xrow,
                feature_names=X_test.columns.tolist(),
                model_name=name
            )
            if len(_sv) == X_test.shape[1]:
                df_contrib = pd.DataFrame({
                    "feature": X_test.columns,
                    "value": xrow.values,
                    "shap_value_logit": _sv
                })
                df_contrib.to_csv(OUT_DIR / f"decision_logit_{name}_idx{SAMPLE_IDX}.csv",
                                  index=False, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] SHAP failed for {name}: {e}")

    if not L1_TEST_X.exists():
        raise FileNotFoundError(f"Missing L1_test_X: {L1_TEST_X}")
    L1_test = pd.read_csv(L1_TEST_X)
    cols_in_order = resolve_l1_columns(selected, L1_test.columns)
    if len(cols_in_order) == 0:
        raise RuntimeError("No overlap between L1_test_X columns and selected base models (even with suffix/substring matching).")

    L1_row_df = L1_test.loc[[SAMPLE_IDX], cols_in_order].copy()
    xi_vals = {f"X{i}": float(L1_row_df.iloc[0, i-1]) for i in range(1, len(cols_in_order)+1)}

    mapping_df = pd.DataFrame({
        "Xi": [f"X{i}" for i in range(1, len(cols_in_order) + 1)],
        "L1_column": cols_in_order,
        "value": [xi_vals[f"X{i}"] for i in range(1, len(cols_in_order)+1)]
    })
    print("\n[GP mapping 1-based] Xi -> L1 columns (sample values):")
    print(mapping_df.to_string(index=False))

    if not GP_SEL_CSV.exists():
        raise FileNotFoundError(f"Missing GP selection: {GP_SEL_CSV}")
    gp_sel = pd.read_csv(GP_SEL_CSV)
    if gp_sel.empty or "expression" not in gp_sel.columns:
        raise RuntimeError(f"No expression column in {GP_SEL_CSV}")
    gp_expr_raw = str(gp_sel.loc[0, "expression"])
    gp_fn, gp_expr_clean = compile_gp_callable_one_based(gp_expr_raw, cols_in_order)

    gp_raw, gp_prob, gp_pred = gp_fn(L1_row_df)

    expr_sub = show_gp_substitution(gp_expr_clean, xi_vals, decimals=6)
    print("\n================ GP EVALUATION (with substitution) ================")
    print("Original g(X1, X2, ...):")
    print(gp_expr_clean)
    print("\nSubstitute Xi with L1 values (rounded):")
    print(expr_sub)
    print(f"\nResult: raw = {gp_raw:.6f}  ->  prob = {gp_prob:.6f}  ->  pred = {gp_pred}  (threshold={GP_FIXED_THRESH})")
    print("===================================================================")

    vals = L1_row_df.iloc[0].abs().sort_values(ascending=False)
    topk = min(10, len(vals))
    vals_plot = vals.iloc[:topk][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.barh(vals_plot.index, vals_plot.values)
    ax.set_title(f"L1 feature magnitudes (top-{topk}) — sample #{SAMPLE_IDX}")
    ax.set_xlabel("Absolute value")
    plt.tight_layout()
    fig_path = OUT_DIR / f"L1_sample{SAMPLE_IDX}_top{topk}.png"
    plt.savefig(fig_path, dpi=220)
    plt.show()

    out_rec = {
        "sample_index": int(SAMPLE_IDX),
        "y_true": int(int(pd.Series(y_test).iloc[SAMPLE_IDX])),
        "selected_base_models": selected,
        "L1_columns_used": cols_in_order,
        "L1_values": {k: float(L1_row_df.iloc[0][k]) for k in cols_in_order},
        "gp_expression": gp_expr_clean,
        "gp_expression_substituted": expr_sub,
        "gp_raw": float(gp_raw),
        "gp_prob": float(gp_prob),
        "gp_pred": int(gp_pred),
        "gp_threshold": float(GP_FIXED_THRESH),
        "plots": {
            "L1_bar": str(fig_path.name),
            "per_model_decision_png": [f"decision_logit_{m}_idx{SAMPLE_IDX}.png" for m in selected]
        }
    }
    (OUT_DIR / f"sample{SAMPLE_IDX}_baseSHAP_logit_and_GP.json").write_text(
        json.dumps(out_rec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OUT] Records written: {OUT_DIR / f'sample{SAMPLE_IDX}_baseSHAP_logit_and_GP.json'}")
    print(f"[OUT] Plots saved under: {OUT_DIR}")

if __name__ == "__main__":
    main()



import warnings
import numpy as np
import pandas as pd
from pathlib import Path


warnings.filterwarnings("ignore", category=FutureWarning)


ROOT         = Path(r"D:\sh-gpc")
ART_DIR      = ROOT / "experiments" / "artifacts"
L1_TEST_CSV  = ROOT / "experiments" / "level1" / "L1_test_X.csv"
SEL_JSON     = ART_DIR / "selected_base_models.json"
GP_SEL_CSV   = ART_DIR / "gp_selected_test.csv"
MODELS_DIR   = ROOT / "experiments" / "models"
OUT_DIR      = ROOT / "experiments" / "explain"
OUT_DIR.mkdir(parents=True, exist_ok=True)


N_SOBOL     = 50_000   # Monte Carlo 样本数
RANDOM_SEED = 1412
USE_TOTAL_ORDER_FOR_TOP = True  # 选“贡献最大”用 ST（总效应）；False 则用一阶 S
BG_SIZE     = 500      # SHAP 背景样本量
MAX_SHOW    = 1000     # SHAP beeswarm 显示最多样本
UNIT_TO_L1  = True     # True: 单位超立方 -> 直接作为 GP 的 X1..Xk；False: 同 True（两者对 GP 一样）
CLIP_RANGE  = 10.0     # 与之前一致：GP raw 裁剪
EPS         = 1e-8


NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "tanh": np.tanh, "sigmoid": lambda z: 1/(1+np.exp(-z)),
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where,
    "pow": np.power
}


def load_base_model(name: str):
    name = name.strip().lower()
    if name == "tabnet":
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            m = TabNetClassifier()
            base = str(MODELS_DIR / "tabnet_model")  # .zip 自动附加
            m.load_model(base + ".zip")
            return m
        except Exception as e:
            print(f"[WARN] load tabnet failed: {e}")
            return None
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"[WARN] model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] load {name} failed: {e}")
        return None

def predict_proba1(model, X: pd.DataFrame) -> np.ndarray:
    """概率输出，若无则用 decision_function 过 sigmoid。"""
    try:
        proba = model.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=float)
    except Exception:
        try:
            dec = model.decision_function(X).astype(float)
            return 1.0 / (1.0 + np.exp(-dec))
        except Exception as e:
            raise RuntimeError(f"model has no proba/decision_function: {e}")

def compile_gp_vectorized_one_based(expr_raw: str, k_vars: int):
    """
    返回 f(U)：
      - U: (N, k) 数组，列对应 X1..Xk，数值域为 [0,1]
      - 输出 raw、prob（sigmoid(raw)）
    """
    expr = expr_raw.strip().rstrip(";").replace("^", "**")
    code = compile(ast.parse(expr, mode="eval"), "<gp_expr>", "eval")
    idxs = [int(m.group(1)) for m in re.finditer(r"\b[Xx](\d+)\b", expr)]
    if idxs and max(idxs) > k_vars:
        raise ValueError(f"GP expression references X{max(idxs)} but only {k_vars} inputs provided.")

    def _f(U: np.ndarray):
        # U: (N, k)
        N, k = U.shape
        env = {}
        for j in range(1, k + 1):
            env[f"X{j}"] = U[:, j-1]
            env[f"x{j}"] = U[:, j-1]
        env.update(NP_FUNCS)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            raw = eval(code, {"__builtins__": {}}, env)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        raw[~np.isfinite(raw)] = 0.0
        raw = np.clip(raw, -CLIP_RANGE, CLIP_RANGE)
        prob = 1.0 / (1.0 + np.exp(-raw))
        return raw, prob

    return _f, expr

# ---------------- Sobol analysis ----------------
def sobol_indices(fun_prob, k: int, N=50_000, seed=0):
    """
    fun_prob: callable(U)-> prob，U: (N, k) in [0,1]^k
    返回 (S, ST)
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (N, k))
    B = rng.uniform(0, 1, (N, k))


    _, YA_prob = fun_prob(A)
    _, YB_prob = fun_prob(B)
    varY = np.var(np.concatenate([YA_prob, YB_prob]), ddof=1)
    if varY < EPS:

        return np.zeros(k), np.zeros(k)

    S  = np.zeros(k)
    ST = np.zeros(k)
    for i in range(k):
        C = A.copy()
        C[:, i] = B[:, i]
        _, YC_prob = fun_prob(C)

        S[i]  = np.mean(YB_prob * (YC_prob - YA_prob)) / varY
        ST[i] = 0.5 * np.mean((YA_prob - YC_prob) ** 2) / varY

    return S, ST

# ---------------- SHAP beeswarm ----------------
def shap_beeswarm_for_model(base_name: str,
                            model,
                            X_all: pd.DataFrame,
                            out_png: Path,
                            bg_size=500,
                            max_show=1000):
    import shap
    cols = X_all.columns.tolist()

    Xs = X_all.sample(min(len(X_all), max_show), random_state=RANDOM_SEED)
    Xbg = X_all.sample(min(len(X_all), bg_size), random_state=RANDOM_SEED)


    use_kernel = False
    explainer = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xs)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    except Exception as e:
        print(f"[WARN] TreeExplainer failed for {base_name}: {e} -> fallback Kernel")
        use_kernel = True

    if use_kernel:
        f = lambda X: predict_proba1(model, pd.DataFrame(X, columns=cols))
        explainer = shap.KernelExplainer(f, Xbg.to_numpy())
        shap_vals = explainer.shap_values(Xs.to_numpy(), nsamples="auto")
        if isinstance(shap_vals, list):
            shap_vals = np.asarray(shap_vals)[0]

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals, Xs, show=False)
    plt.title(f"SHAP beeswarm — {base_name}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.show()


def main():
    np.random.seed(RANDOM_SEED)
    if not SEL_JSON.exists():
        raise FileNotFoundError(f"Missing: {SEL_JSON}")
    selected = json.loads(SEL_JSON.read_text(encoding="utf-8"))
    if not isinstance(selected, list) or len(selected) == 0:
        raise RuntimeError("selected_base_models.json is empty.")
    if not L1_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing L1_test_X: {L1_TEST_CSV}")

    L1_test = pd.read_csv(L1_TEST_CSV)


    def _find_col(m, cols):
        m = m.lower()
        cands = [m, f"{m}_l1", f"{m}_oof", f"{m}_proba", f"{m}_prob", f"{m}_predict", f"{m}_pred"]
        lower_map = {c.lower(): c for c in cols}
        for cand in cands:
            if cand in lower_map: return lower_map[cand]
        for lc, orig in lower_map.items():
            if m in lc: return orig
        return None

    cols = []
    miss = []
    for m in selected:
        c = _find_col(m, L1_test.columns)
        if c is None: miss.append(m)
        else: cols.append(c)
    if len(cols) == 0:
        raise RuntimeError("No overlap between L1 columns and selected base models.")
    if miss:
        print(f"[WARN] no L1 column for: {miss}")
    print("[L1 columns used (order→X1..Xk)]:", cols)

    k = len(cols)


    if not GP_SEL_CSV.exists():
        raise FileNotFoundError(f"Missing GP selection: {GP_SEL_CSV}")
    gp_sel = pd.read_csv(GP_SEL_CSV)
    if gp_sel.empty or "expression" not in gp_sel.columns:
        raise RuntimeError(f"No expression column in {GP_SEL_CSV}")
    gp_expr_raw = str(gp_sel.loc[0, "expression"])

    gp_fun, gp_expr_clean = compile_gp_vectorized_one_based(gp_expr_raw, k_vars=k)

    #  Sobol
    def fun_prob(U):
        raw, prob = gp_fun(U)
        return raw, prob

    print(f"[SOBOL] N={N_SOBOL}, k={k}, expr={gp_expr_clean}")
    S, ST = sobol_indices(fun_prob, k=k, N=N_SOBOL, seed=RANDOM_SEED)

    labels = [f"X{i+1} ({selected[i]})" for i in range(k)]
    y = np.arange(k)
    h = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    ax.barh(y - h/2, S,  h, label="First-order $S_i$")
    ax.barh(y + h/2, ST, h, label="Total-order $S_{Ti}$")
    for i, v in enumerate(S):
        ax.text(v + 0.01, i - h/2, f"{v:.3f}", va='center', fontsize=9)
    for i, v in enumerate(ST):
        ax.text(v + 0.01, i + h/2, f"{v:.3f}", va='center', fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Sobol index")
    ax.set_xlim(0, 1)
    ax.set_title("Sobol sensitivity of GP meta on L1 inputs")
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    sobol_png = OUT_DIR / "gp_meta_sobol.png"
    plt.savefig(sobol_png, dpi=220)
    plt.show()

    pd.DataFrame({"Xi": [f"X{i+1}" for i in range(k)],
                  "base_model": selected[:k],
                  "S": S, "ST": ST}).to_csv(OUT_DIR / "gp_meta_sobol.csv", index=False)


    idx_top = int(np.argmax(ST if USE_TOTAL_ORDER_FOR_TOP else S))
    top_model_name = selected[idx_top]
    print(f"[TOP] by {'ST' if USE_TOTAL_ORDER_FOR_TOP else 'S'}: X{idx_top+1} -> {top_model_name}")


    base_model = load_base_model(top_model_name)
    if base_model is None:
        print(f"[WARN] skip beeswarm: cannot load model {top_model_name}")
        return


    DATA_X_TEST  = ROOT / "X_ogtest_s.csv"
    if not DATA_X_TEST.exists():
        raise FileNotFoundError(f"Missing raw test features: {DATA_X_TEST}")
    X_test_raw = pd.read_csv(DATA_X_TEST)

    bees_png = OUT_DIR / f"beeswarm_{top_model_name}.png"
    shap_beeswarm_for_model(top_model_name, base_model, X_test_raw, bees_png,
                            bg_size=BG_SIZE, max_show=MAX_SHOW)

    print(f"[DONE] Sobol : {sobol_png.name}, beeswarm: {bees_png.name}, path: {OUT_DIR}")

if __name__ == "__main__":
    main()



