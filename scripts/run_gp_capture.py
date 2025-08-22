# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 23:17:25 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-
import os, sys, re, csv, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 仓库根目录
JAVA_DIR = ROOT / "java"
BIN_DIR  = JAVA_DIR / "bin"
SRC_DIR  = JAVA_DIR / "src"

X_TRAIN  = ROOT / "experiments" / "level1" / "L1_train_X.csv"
Y_TRAIN  = ROOT / "experiments" / "labels" / "y_tomtrain.csv"

WORK_TXT = ROOT / "experiments" / "level1" / "L1_train_XY.txt"   # 中间文件（无表头）
OUT_DIR  = ROOT / "experiments" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LOG  = OUT_DIR / "gp_raw_log.txt"
OUT_CSV  = OUT_DIR / "gp_expressions.csv"

JAVA_MAIN = "sh_gpc"   # 与 sh_gpc.java 的类名一致

def ensure_compiled():
    """如果还没编译过，就编译 sh_gpc.java"""
    if (BIN_DIR / f"{JAVA_MAIN}.class").exists():
        return
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    java_files = [str(p) for p in SRC_DIR.glob("*.java")]
    if not java_files:
        raise FileNotFoundError(f"No .java files in {SRC_DIR}")
    cmd = ["javac", "-encoding", "UTF-8", "-d", str(BIN_DIR)] + java_files
    print("[PY] Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def make_xy_txt():
    """
    将 L1_train_X.csv 与 y_tomtrain.csv 合并为 txt：
    第一行: "<p> 0 0 0 <n>"
    其后每行: p 个特征 + 1 个标签y（空格分隔，无表头）
    """
    import pandas as pd
    X = pd.read_csv(X_TRAIN)
    y = pd.read_csv(Y_TRAIN).squeeze("columns")

    if len(X) != len(y):
        raise ValueError(f"length mismatch: X={len(X)} vs y={len(y)}")

    # 拼接，y 放最后一列
    df = X.copy()
    df["y"] = y.astype(int).values  # 确保 0/1 整型

    p = X.shape[1]   # 特征数
    n = X.shape[0]   # 样本数

    # 写入自定义头 + 数据（空格分隔）
    WORK_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(WORK_TXT, "w", encoding="utf-8") as f:
        f.write(f"{p} 0 0 0 {n}\n")
        # 继续把数据行以空格分隔写入
        df.to_csv(f, sep=" ", index=False, header=False, float_format="%.10g")

    print(f"[PY] Wrote (with header): {WORK_TXT}  | header='{p} 0 0 0 {n}'")

def run_java_and_capture():
    cmd = ["java", "-cp", str(BIN_DIR), JAVA_MAIN, str(WORK_TXT)]
    print("[PY] Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    best_expr = []
    cur_gen = None
    pat_gen  = re.compile(r"^Generation\s*=\s*(\d+)\b")
    pat_best = re.compile(r"^Best Individual:\s*(.*)$")

    for ln in proc.stdout:
        lines.append(ln.rstrip("\n"))
        m1 = pat_gen.search(ln)
        if m1:
            cur_gen = int(m1.group(1))
            continue
        m2 = pat_best.search(ln)
        if m2 and cur_gen is not None:
            expr = m2.group(1).strip()
            best_expr.append((cur_gen, expr))
            cur_gen = None  # 防止一代多次匹配

    code = proc.wait()
    RAW_LOG.write_text("\n".join(lines), encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"Java exited with {code}. See {RAW_LOG}")

    # 去重并排序
    keep = {}
    for g, e in best_expr:
        keep.setdefault(g, e)
    rows = [(g, keep[g]) for g in sorted(keep.keys())]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "expression"])
        for g, e in rows:
            w.writerow([g, e])

    print(f"[PY] Wrote expressions -> {OUT_CSV}")
    print(f"[PY] Raw log -> {RAW_LOG}")

if __name__ == "__main__":
    ensure_compiled()
    make_xy_txt()
    run_java_and_capture()
