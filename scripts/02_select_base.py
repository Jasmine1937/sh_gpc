
# scripts/01a_select_base.py
# -*- coding: utf-8 -*-
"""
EDP selection (Effectiveness -> Diversity -> Pruning) with SIMPLE diversity thresholds.
Not include lda,probit

Inputs (from 01_train_base.py):
  - experiments/metrics/val_metrics.csv  (columns include: model, auc/f1/accuracy/precision/recall)
  - experiments/predictions/val/{model}_proba.csv
  - experiments/predictions/val/{model}_pred.csv  (if missing, we threshold proba at 0.5)
  - y_val_s.csv  (ground-truth labels for validation)

Outputs:
  - experiments/artifacts/effectiveness_pass.json
  - experiments/artifacts/diversity_matrix_phi.csv
  - experiments/artifacts/diversity_pass.json
  - experiments/artifacts/shap_importances.csv
  - experiments/artifacts/selection_report.json
  - experiments/artifacts/selected_base_models.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------- paths & config -----------------
METRICS_CSV = os.path.join("experiments", "metrics", "val_metrics.csv")
VAL_DIR     = os.path.join("experiments", "predictions", "val")
Y_VAL_FILE  = "y_val_s.csv"

ARTIFACT_DIR = os.path.join("experiments", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SCORE_NAME = "auc"   # which column in val_metrics.csv to use as s(h)
RANDOM_STATE = 1412

# ----------------- I/O helpers -----------------
def _read_val_labels() -> np.ndarray:
    y = pd.read_csv(Y_VAL_FILE)
    return y.values.ravel().astype(int)

def _read_scores() -> pd.DataFrame:
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Missing metrics CSV: {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    if "model" not in df.columns or SCORE_NAME not in df.columns:
        raise ValueError(f"`val_metrics.csv` must contain 'model' and '{SCORE_NAME}' columns.")
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df[SCORE_NAME] = pd.to_numeric(df[SCORE_NAME], errors="coerce")
    return df

def _read_proba(model: str) -> np.ndarray:
    f = os.path.join(VAL_DIR, f"{model}_proba.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing proba for model '{model}': {f}")
    s = pd.read_csv(f).squeeze("columns")
    return s.to_numpy().astype(float)

def _read_pred(model: str) -> np.ndarray:
    f = os.path.join(VAL_DIR, f"{model}_pred.csv")
    if not os.path.exists(f):
        p = _read_proba(model)
        return (p >= 0.5).astype(int)
    s = pd.read_csv(f).squeeze("columns")
    return s.to_numpy().astype(int)

# ----------------- step 1: effectiveness -----------------
def _effectiveness_filter(scores: pd.DataFrame) -> Tuple[List[str], Dict[str, float], Dict]:
    col = SCORE_NAME
    scores = scores.copy()
    s_valid = scores.loc[np.isfinite(scores[col]), col].to_numpy(dtype=float)

    info = {"metric": col}
    if s_valid.size == 0:
        mu = float("nan"); sigma = float("nan"); tau_eff = float("-inf")
        kept = scores["model"].tolist()
        info["note"] = "all scores are NaN; keeping all by default"
    else:
        mu = float(np.mean(s_valid))
        sigma = float(np.sqrt(np.mean((s_valid - mu) ** 2)))
        tau_eff = mu - sigma
        keep_mask = (scores[col] >= tau_eff) & np.isfinite(scores[col])
        kept = scores.loc[keep_mask, "model"].tolist()
        if len(kept) == 0:
            top1 = scores.loc[np.isfinite(scores[col])].sort_values(col, ascending=False)["model"].iloc[0]
            kept = [top1]
            info["note"] = "no model >= tau_eff; kept top-1 as fallback"

    info.update({
        "mu": mu, "sigma": sigma, "tau_eff": tau_eff,
        "kept": kept, "dropped": scores.loc[~scores["model"].isin(kept), "model"].tolist()
    })
    with open(os.path.join(ARTIFACT_DIR, "effectiveness_pass.json"), "w") as f:
        json.dump(info, f, indent=2)

    score_map = {}
    for _, r in scores.iterrows():
        v = r[col]
        score_map[r["model"]] = float(v) if np.isfinite(v) else float("-inf")
    return kept, score_map, info

# ----------------- diversity: (MCC) matrix -----------------
def _phi_from_preds(a: np.ndarray, b: np.ndarray) -> float:
    # counts: A=11, B=10, C=01, D=00
    A = int(np.sum((a == 1) & (b == 1)))
    B = int(np.sum((a == 1) & (b == 0)))
    C = int(np.sum((a == 0) & (b == 1)))
    D = int(np.sum((a == 0) & (b == 0)))
    denom = (A + B) * (A + C) * (C + D) * (B + D)
    return 0.0 if denom == 0 else float((A * D - B * C) / np.sqrt(denom))

def _pairwise_phi(proba_dict: Dict[str, np.ndarray], pred_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    models = list(proba_dict.keys())
    n = len(models)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = models[i], models[j]
            rho = _phi_from_preds(pred_dict[mi].astype(int), pred_dict[mj].astype(int))
            M[i, j] = M[j, i] = rho
    df = pd.DataFrame(M, index=models, columns=models)
    df.to_csv(os.path.join(ARTIFACT_DIR, "diversity_matrix_phi.csv"))
    return df

def _score_from_probs(y_true: np.ndarray, probs: np.ndarray, metric: str) -> float:

    hard = (probs >= 0.5).astype(int)
    try:
        if metric == "auc":
            return float(roc_auc_score(y_true, hard))
        elif metric == "f1":
            return float(f1_score(y_true, hard, zero_division=0))
        elif metric == "accuracy":
            return float(accuracy_score(y_true, hard))
        elif metric == "precision":
            return float(precision_score(y_true, hard, zero_division=0))
        elif metric == "recall":
            return float(recall_score(y_true, hard, zero_division=0))
        else:
            raise ValueError(f"Unsupported metric for combination check: {metric}")
    except Exception:
        return float("-inf")

# ----------------- step 2: diversity filter -----------------
def _diversity_filter_simple(
    kept_models: List[str],
    score_map: Dict[str, float],
    proba_dict: Dict[str, np.ndarray],
    pred_dict: Dict[str, np.ndarray],
    y_val: np.ndarray,
    scores_df: pd.DataFrame,
    rho_mat: pd.DataFrame
) -> Tuple[List[str], Dict]:
    """
    Tiered thresholds:
      delta = 0.97 / 0.95 / 0.92 depending on mu_s (>=0.95 / [0.85,0.95) / <0.85)
      epsilon = 0.002 / 0.005 / 0.010 accordingly
      adaptive: delta <- 0.5*delta + 0.5*Q90(phi)  
    Greedy keep by descending standalone score; drop a model if it is too similar
    to any selected one AND fails the tiny-gain test.
    """
    # mean validation score over EFFECTIVENESS-KEPT models
    mu_s_arr = scores_df.loc[scores_df["model"].isin(kept_models), SCORE_NAME].to_numpy(dtype=float)
    mu_s = float(np.nanmean(mu_s_arr)) if mu_s_arr.size > 0 else float("nan")

    # tiered thresholds
    if np.isfinite(mu_s) and mu_s >= 0.95:
        delta = 0.97; eps_improve = 0.002
    elif np.isfinite(mu_s) and mu_s >= 0.85:
        delta = 0.95; eps_improve = 0.005
    else:
        delta = 0.92; eps_improve = 0.010

    # adaptive softening by Q90 of phi
    phis = rho_mat.values[np.triu_indices_from(rho_mat.values, 1)]
    if phis.size > 0:
        q90 = float(np.quantile(phis, 0.90))
        delta = 0.5 * delta + 0.5 * min(0.999, q90)
    else:
        q90 = float("nan")

    # greedy selection by descending standalone score
    order = sorted(kept_models, key=lambda m: score_map.get(m, float("-inf")), reverse=True)
    selected = []
    drops = []  # records of pruned pairs

    for m in order:
        keep = True
        for s in selected:
            rho = float(rho_mat.loc[m, s])
            if rho >= delta:
                # complementarity small-gain test: average probabilities of the pair
                avg_prob = (proba_dict[m] + proba_dict[s]) / 2.0
                comb_score = _score_from_probs(y_val, avg_prob, SCORE_NAME)
                best_pair = max(score_map.get(m, float("-inf")), score_map.get(s, float("-inf")))
                if comb_score <= best_pair + eps_improve:
                    keep = False
                    drops.append({
                        "pair": [m, s],
                        "rho": rho,
                        "comb_score": comb_score,
                        "best_pair": best_pair,
                        "eps": eps_improve,
                        "dropped": m,
                        "kept": s
                    })
                    break
        if keep:
            selected.append(m)

    info = {
        "mu_s": mu_s, "delta": float(delta), "q90_phi": q90,
        "eps_improve": float(eps_improve),
        "dropped_pairs": drops,
        "selected_after_diversity": selected
    }
    with open(os.path.join(ARTIFACT_DIR, "diversity_pass.json"), "w") as f:
        json.dump(info, f, indent=2)
    return selected, info

# ----------------- step 3: SHAP pruning -----------------
def _shap_prune(selected_models: List[str], proba_dict: Dict[str, np.ndarray], y_val: np.ndarray) -> Tuple[List[str], pd.DataFrame, Dict]:
    X_meta = np.column_stack([proba_dict[m] for m in selected_models])
    X_df = pd.DataFrame(X_meta, columns=selected_models)

    comb = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)
    comb.fit(X_df, y_val)

    try:
        import shap
        explainer = shap.LinearExplainer(comb, X_df, feature_dependence="independent")
        shap_vals = explainer.shap_values(X_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        I = np.mean(np.abs(shap_vals), axis=0)
        source = "shap_linear"
    except Exception:
        I = np.abs(comb.coef_.ravel())
        source = "abs_coef_fallback"

    imp = pd.DataFrame({"model": selected_models, "importance": I})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(ARTIFACT_DIR, "shap_importances.csv"), index=False)

    S = float(imp["importance"].sum())
    if S <= 0:
        final_keep = [imp.loc[0, "model"]]
        tau = float(imp.loc[0, "importance"]); k = 1
    else:
        cum = imp["importance"].cumsum()
        k = int(np.searchsorted(cum, 0.80 * S) + 1)
        k = max(1, min(k, len(imp)))
        tau = float(imp.loc[k - 1, "importance"])
        final_keep = imp.loc[imp["importance"] >= tau, "model"].tolist()

    info = {
        "importance_source": source,
        "cum_percent": 0.80,
        "k": k, "tau_shap": tau, "S": S,
        "kept": final_keep
    }
    return final_keep, imp, info

# ----------------- main orchestration -----------------
def main():
    # read metrics and labels
    scores = _read_scores()
    scores = scores[~scores["model"].isin(["lda", "probit"])].reset_index(drop=True)
    y_val = _read_val_labels()

    # step 1: effectiveness
    kept_eff, score_map, eff_info = _effectiveness_filter(scores)
    if len(kept_eff) == 0:
        print("[Selection] No model passed the effectiveness filter.")
        with open(os.path.join(ARTIFACT_DIR, "selected_base_models.json"), "w") as f:
            json.dump([], f)
        return

    # load predictions/probabilities for effectiveness-kept models
    proba_dict, pred_dict = {}, {}
    for m in kept_eff:
        try:
            proba_dict[m] = _read_proba(m)
            pred_dict[m]  = _read_pred(m)
        except Exception as e:
            print(f"[WARN] Missing predictions for model {m}: {e}")

    # ensure alignment
    avail = [m for m in kept_eff if (m in proba_dict and len(proba_dict[m]) == len(y_val))]
    if len(avail) < 1:
        raise RuntimeError("No model has usable validation predictions/probabilities.")
    kept_eff = avail

    # step 2: diversity (phi) with SIMPLE thresholds
    rho_mat = _pairwise_phi(proba_dict, pred_dict)
    kept_div, div_info = _diversity_filter_simple(
        kept_eff, score_map, proba_dict, pred_dict, y_val, scores, rho_mat
    )

    # step 3: SHAP pruning
    kept_final, imp_df, shap_info = _shap_prune(kept_div, proba_dict, y_val)

    # persist final selection and report
    with open(os.path.join(ARTIFACT_DIR, "selected_base_models.json"), "w") as f:
        json.dump(kept_final, f, indent=2)

    report = {
        "config": {
            "score_name": SCORE_NAME,
            "random_state": RANDOM_STATE
        },
        "effectiveness": eff_info,
        "diversity": div_info | {"matrix_csv": os.path.join(ARTIFACT_DIR, "diversity_matrix_phi.csv")},
        "shap": shap_info,
        "final_selection": kept_final
    }
    with open(os.path.join(ARTIFACT_DIR, "selection_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("\n[Selection] Effectiveness kept:", kept_eff)
    print("[Selection] After diversity:", kept_div)
    print("[Selection] Final (SHAP-pruned):", kept_final)
    print(f"[Selection] Artifacts written to: {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
