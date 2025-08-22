
# -*- coding: utf-8 -*-
"""
Evaluate GP expressions on the TEST set:
- Classification threshold = Youden index on outputs, or a fixed threshold
- Clip outputs, filter by min_auc and max_std


Inputs:
  - experiments/artifacts/"output.txt"  (trained by Java)
  - experiments/level1/"L1_test_X.csv"
  - experiments/labels/"y_ogtest_s.csv"

Outputs:
  - experiments/artifacts/"gp_candidates_test.csv"
  - experiments/artifacts/"gp_selected_test.csv"
  - experiments/artifacts/"gp_debug_parse.json"

Root fixed at D:\sh-gpc
"""

import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)


ROOT = Path(r"D:\sh-gpc")
ART_DIR = ROOT / "experiments" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

GP_OUTPUT = ART_DIR / "output.txt"   # GP traning output from Java 
X_TEST    = ROOT / "experiments" / "level1" / "L1_test_X.csv"
Y_TEST    = ROOT / "experiments" / "labels" / "y_ogtest_s.csv"

CAND_CSV  = ART_DIR / "gp_candidates_test.csv"
SEL_CSV   = ART_DIR / "gp_selected_test.csv"
DEBUG_JSON= ART_DIR / "gp_debug_parse.json"


# ----------------Config (adjustable as needed----------------
# ——Complexity threshold——
TAU_T = 200      # max node
TAU_D = 20       # max depth
EPS_TIE = 0.0    

# ——logic——
CLIP_RANGE    = 10.0     
MIN_AUC       = 0.55     
MAX_STD       = 100.0    
FIXED_THRESH  = None

# One-SE 
Z_ONE_SE = 1.0   # suggest 1.0；A more conservative 1.64 or 1.96 is preferre


NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "tanh": np.tanh, "sigmoid": lambda z: 1/(1+np.exp(-z)),
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where,
    "pow": np.power
}

# =========================
# 1) resolve output.txt
# =========================
def normalize_expr(s: str) -> str:
    s = s.strip().rstrip(';')
    s = s.replace('^', '**')
    return s

def parse_gp_output(txt_path: Path) -> List[Tuple[int, str]]:

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing GP output: {txt_path}")

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[Tuple[int, str]] = []

    pat_gen_csv  = re.compile(r'^\s*GEN\s*,\s*(\d+)\s*,\s*(.+)$')
    pat_gen_line = re.compile(r'^\s*Generation\s*=\s*(\d+)\b', re.IGNORECASE)
    pat_best     = re.compile(r'^\s*Best Individual:\s*(.+)$', re.IGNORECASE)

    current_gen = None
    fallback_gen = 0

    for ln in lines:
        m_csv = pat_gen_csv.match(ln)
        if m_csv:
            g = int(m_csv.group(1))
            expr = normalize_expr(m_csv.group(2))
            out.append((g, expr))
            continue

        m_gen = pat_gen_line.search(ln)
        if m_gen:
            current_gen = int(m_gen.group(1))
            continue

        m_best = pat_best.match(ln)
        if m_best:
            expr = normalize_expr(m_best.group(1))
            if current_gen is None:
                out.append((fallback_gen, expr))
                fallback_gen += 1
            else:
                out.append((current_gen, expr))
                current_gen = None

    keep = {}
    for g, e in out:
        keep.setdefault(g, e)
    rows = [(g, keep[g]) for g in sorted(keep.keys())]
    return rows

# =========================
# 2) Complexity：T, D, O, |U|
# =========================
class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.T = 0      
        self.O = 0      
        self.U = set()  

    def generic_visit(self, node):
        self.T += 1
        super().generic_visit(node)

    def visit_BinOp(self, node):
        self.O += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if re.match(r'^[Xx]\d+$', node.id):
            self.U.add(node.id.upper())
        self.generic_visit(node)

def compute_depth(n) -> int:
    if not isinstance(n, ast.AST):
        return 0
    ch = list(ast.iter_child_nodes(n))
    if not ch:
        return 1
    return 1 + max(compute_depth(c) for c in ch)

def compute_complexity(expr: str):
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return (10**9, 10**9, 10**9, 10**9)
    vis = ComplexityVisitor()
    vis.visit(tree)
    D = compute_depth(tree)
    return (vis.T, D, vis.O, len(vis.U))

# =========================
# 3) evaluate expression
# =========================
def eval_expr_on_X(expr: str, X: pd.DataFrame) -> np.ndarray:

    env: Dict[str, object] = {}
    p = X.shape[1]
    for j in range(p):
        env[f"X{j}"] = X.iloc[:, j].to_numpy(dtype=float)
        env[f"x{j}"] = env[f"X{j}"]  
    env.update(NP_FUNCS)

    try:
        code = compile(ast.parse(expr, mode='eval'), "<expr>", "eval")
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            out = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return np.full(X.shape[0], np.nan, dtype=float)

    out = np.asarray(out, dtype=float).reshape(-1)
    out[~np.isfinite(out)] = np.nan
    return out

def evaluate_like_yours(y_true: np.ndarray,
                        y_score: np.ndarray,
                        clip_range: float = CLIP_RANGE,
                        min_auc: float = MIN_AUC,
                        max_std: float = MAX_STD,
                        fixed_threshold: float | None = FIXED_THRESH) -> Dict | None:

    s = np.copy(y_score)
    s[~np.isfinite(s)] = 0.0
    s = np.clip(s, -clip_range, clip_range)

    mean_out = float(np.mean(s))
    std_out  = float(np.std(s))
    if std_out > max_std:
        return None

    try:
        auc_val = float(roc_auc_score(y_true, s))
    except Exception:
        return None
    if auc_val < min_auc:
        return None

    if fixed_threshold is not None:
        best_thresh = float(fixed_threshold)
    else:
        fpr, tpr, thresholds = roc_curve(y_true, s)
        youden = tpr - fpr
        best_thresh = float(thresholds[int(np.argmax(youden))])

    y_pred = (s > best_thresh).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred))

    return dict(
        auc=auc_val, accuracy=acc, recall=rec, precision=prec, f1=f1,
        best_threshold=best_thresh, out_mean=mean_out, out_std=std_out
    )

def auc_se_hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> float:

    if not (0.0 < auc < 1.0) or n_pos <= 0 or n_neg <= 0:
        return float("inf")
    q1 = auc / (2 - auc)
    q2 = (2 * auc * auc) / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def main():
    print(f"[paths] ROOT = {ROOT}")


    if not X_TEST.exists() or not Y_TEST.exists():
        raise FileNotFoundError("Missing L1_test_X.csv or y_ogtest_s.csv")

    X = pd.read_csv(X_TEST)
    y = pd.read_csv(Y_TEST).squeeze("columns").to_numpy(dtype=int)
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X_test={len(X)} vs y_test={len(y)}")


    cand = parse_gp_output(GP_OUTPUT)
    if not cand:
        raise RuntimeError(f"No expressions parsed from: {GP_OUTPUT}")

    DEBUG_JSON.write_text(
        json.dumps({"parsed": [{"generation": g, "expression": e} for g, e in cand]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


    rows = []
    kept_any = False
    for gen, expr in cand:
        T, D, O, Usize = compute_complexity(expr)
        y_score = eval_expr_on_X(expr, X)                 
        metrics = evaluate_like_yours(y, y_score)         
        if metrics is None:
            rows.append({
                "generation": gen, "expression": expr,
                "T": T, "D": D, "O": O, "U": Usize,
                "test_auc": np.nan, "test_accuracy": np.nan,
                "test_recall": np.nan, "test_precision": np.nan, "test_f1": np.nan,
                "test_out_mean": np.nan, "test_out_std": np.nan,
                "test_best_threshold": np.nan,
                "feasible": (T <= TAU_T) and (D <= TAU_D)
            })
            continue

        kept_any = True
        rows.append({
            "generation": gen, "expression": expr,
            "T": T, "D": D, "O": O, "U": Usize,
            "test_auc": metrics["auc"],
            "test_accuracy": metrics["accuracy"],
            "test_recall": metrics["recall"],
            "test_precision": metrics["precision"],
            "test_f1": metrics["f1"],
            "test_out_mean": metrics["out_mean"],
            "test_out_std": metrics["out_std"],
            "test_best_threshold": metrics["best_threshold"],
            "feasible": (T <= TAU_T) and (D <= TAU_D)
        })

    if not kept_any:
        print("[WARN] All candidates filtered out by std/auc thresholds; will still write candidates CSV for inspection.")

    df = pd.DataFrame(rows).sort_values(["generation"]).reset_index(drop=True)
    df.to_csv(CAND_CSV, index=False, encoding="utf-8")
    print(f"[INFO] wrote candidates(test) -> {CAND_CSV}")

    F = df[df["feasible"]].copy()
    if F.empty:
        print("[WARN] No expression satisfies (T<=tau_T, D<=tau_D). Falling back to ignore complexity thresholds.")
        F = df.copy()


    F = F.dropna(subset=["test_auc"]).copy()
    if F.empty:
        print("[WARN] No valid test_auc; selecting nothing.")
        sel = F  
        sel.to_csv(SEL_CSV, index=False, encoding="utf-8")
        return


    best_idx = F["test_auc"].idxmax()
    best_row = F.loc[best_idx]
    best_auc = float(best_row["test_auc"])


    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    se_best = auc_se_hanley_mcneil(best_auc, n_pos, n_neg)
    if not np.isfinite(se_best):
        se_best = 0.0  

    tau = best_auc - Z_ONE_SE * se_best


    F1 = F[F["test_auc"] >= tau].copy()
    if F1.empty:  
        F1 = F.loc[[best_idx]].copy()

    #T -> D -> O -> U -> generation
    F1 = F1.sort_values(["T", "D", "O", "U", "generation"],
                        ascending=[True, True, True, True, True])
    sel = F1.iloc[0:1].copy()
    sel.to_csv(SEL_CSV, index=False, encoding="utf-8")

    print(f"[SELECT] eval=TEST | tau_T={TAU_T}, tau_D={TAU_D}, OneSE z={Z_ONE_SE:.2f}")
    print(f"[SELECT] best_auc={best_auc:.6f}, se_best={se_best:.6f}, tau(one-SE)={tau:.6f}, "
          f"candidates_in_band={len(F1)} / feasible={len(F)}")
    print("[SELECT] chosen (simple among statistically-tied):")
    print(sel[["generation", "test_auc", "T", "D", "O", "U", "test_best_threshold", "expression"]].to_string(index=False))

if __name__ == "__main__":
    main()
    
import re
import pandas as pd
from pathlib import Path

SEL_CSV = Path(r"D:\sh-gpc\experiments\artifacts\gp_selected_test.csv")

def pretty_simplify_selected(sel_csv=SEL_CSV):
    import sympy as sp

    df = pd.read_csv(sel_csv)
    if df.empty or "expression" not in df.columns:
        raise ValueError(f"No expression found in {sel_csv}")


    expr_raw = str(df.loc[0, "expression"]).strip().rstrip(";").replace("^", "**")

    names = sorted(set(re.findall(r"[Xx]\d+", expr_raw)),
                   key=lambda s: int(re.findall(r"\d+", s)[0]))
    local = {}
    for v in names:
        sym = sp.symbols(v.upper(), real=True)  
        local[v] = sym
        local[v.upper()] = sym                  

    #  sympify -> simplify
    expr_sym = sp.sympify(expr_raw, locals=local)
    simplified_expr = sp.simplify(expr_sym)

    print(sp.pretty(simplified_expr, use_unicode=True))

    # LaTeX：
    try:
        latex_str = sp.latex(simplified_expr)
    except Exception:
        latex_str = ""

    return simplified_expr, latex_str

simplified_expr, simplified_latex = pretty_simplify_selected()
