# SH-GPC

Reproducible code and artifacts for the **SH-GPC** pipeline: effectiveness–diversity–pruning (EDP) selection of base classifiers, Level‑1 stacking features, and a GP meta‑classifier with principled selection and interpretability tools.

## 1. Repository Layout

```
sh-gpc/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ .gitignore
├─ scripts/
│  ├─ 01_train_base.py  
│  ├─ 02_select_base.py
│  ├─ 03_build_level1_dataset.py
│  ├─ 04_gp_select_new.py
│  ├─ 05_evaluation.py
│  ├─ 06_interpretation.py
│  └─ 07_ablation.py
├─ experiments/      #files will be generated automatically except for labels/             
│  ├─ artifacts/
│  ├─ metrics/
│  ├─ predictions/
│  ├─ level1/
│  ├─ level2/
│  ├─ models/
│  └─ explain/
│  └─ labels/   #put 'y_ogtest_s.csv'& 'y_tomtrain_s.csv' here
└─ data/                        
   ├─ X_tomtrain_s.csv
   ├─ y_tomtrain_s.csv
   ├─ X_val_s.csv
   ├─ y_val_s.csv
   ├─ X_ogtest_s.csv
   └─ y_ogtest_s.csv
└─ java/
   ├─ src
   │   ├─sh_gpc.java
```
> Keep `experiments/**` out of version control. See `.gitignore` below.

---

## 2. Environment

- Python ≥ 3.10
- Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `statsmodels`, `shap`
- Extra/optional (guarded in code; models are skipped if the library is missing): `xgboost`, `lightgbm`, `catboost`, `ngboost`, `pytorch-tabnet` (requires PyTorch).  

Install:
```bash
pip install -r requirements.txt
```

> Install a compatible `torch` first (CUDA/CPU), then install `pytorch-tabnet`.

---

## 3. Quick Start

```bash
make setup
make train      # 01: train 14 base classifiers on train; evaluate on val; save models
make select     # 02: EDP selection → selected_base_models.json
make L1         # 03: build Level-1 OOF train and averaged test features
make L2:gp      # 04: evaluate/select GP expressions
make eval       # 05: bootstrap evaluation (base + meta) on TEST
make interpret  # 06: SHAP (logit) + GP substitution tracing
make ablate     # 07: ablation over L1 subsets and meta-classifier
```

### Windows path note
If any script assumes a project root (e.g., `ROOT = "D:\\sh-gpc"`), either run from the repo root or override that variable via CLI/env. The default paths in this repo write to `experiments/**` under the repo root.

---

## 4. Pipeline Details

### 1) Train base classifiers → writes validation predictions/metrics and saves fitted models

The sample dataset has already been preprocessed, including data cleaning, feature engineering, and the application of sampling techniques. Please first extract the dataset located under `data\` and obtain the training, validation, and test sets (with features and labels already separated). Place these files in the `sh_gpc\` directory.

After preparation, run:
`python scripts/01_train_base.py`

A process similar to the figure below will be observed.

&emsp;<img width="344" height="35" alt="734b500c169ab8205c0bdebfc4d98ed" src="https://github.com/user-attachments/assets/6a22cd35-fcfa-436e-9d79-73b92f66c336" />

Outputs:
- `experiments/predictions/val/{model}_proba.csv`, `{model}_pred.csv`  
- `experiments/metrics/val_metrics.csv` (AUC/ACC/PREC/REC/F1)  
- `experiments/models/{model}.joblib` (TabNet saves as `tabnet_model.zip`)
Make sure all the outpus are saved in the corresponding files.

### 2) EDP selection (Effectiveness → Diversity → Pruning)  
`python scripts/02_select_base.py`

&emsp;<img width="457" height="109" alt="1f094808df22ac3044b984925edc431" src="https://github.com/user-attachments/assets/73319fff-d3cb-4144-b419-facad7a16f16" />

Artifacts:
- `experiments/artifacts/effectiveness_pass.json`  
- `experiments/artifacts/diversity_matrix_phi.csv`  
- `experiments/artifacts/selected_base_models.json`

### 3) Build Level‑1 (stacking) dataset
`python scripts/03_build_level1_dataset.py`

&emsp;<img width="423" height="89" alt="1183aa01f9bf5e1a035b1980ca39017" src="https://github.com/user-attachments/assets/649a0843-1783-40e1-a979-8a7ecdb88649" />  

Outputs:
- `experiments/level1/L1_train_X.csv` (OOF)
- `experiments/level1/L1_test_X.csv` (K‑fold averaged)
- `experiments/level1/folds.csv`, `models_used.json`


### 4) GP meta‑selection  

- Java JDK (e.g., 17+) on your PATH  
**Running GP (Java) before `04_gp_select_new.py`**

Before executing `scripts/04_gp_select_new.py`, train GP on Java and produce a console log file (`output.txt`). The workflow:


#### a) Prepare the GP input (`L1_train_XY.txt`)

Start from the Level-1 files:
- `experiments/level1/L1_train_X.csv`  (OOF features)
- `experiments/labels/y_tomtrain_s.csv` (binary labels 0/1)

Concatenate them **row-wise** so that the label is the **last** column (call the merged file `L1_train_XY.csv`). Then convert it to **space-separated TXT** with this exact header as the **first line**:

```
<P> 0 0 0 <N>
```

- `P` = number of features (columns in `L1_train_X.csv`)
- `N` = number of rows (samples)

After the header, each subsequent line is a sample:
```
x1 x2 ... xP y
```

**Requirements**
- Space-separated, no header line for the data section
- `y` is integer 0/1
- No missing values or NaNs
- Save as `experiments/level1/L1_train_XY.txt`

*Example header (14 features, 12,345 rows):*
```
14 0 0 0 12345
```


#### b) Compile the Java GP engine
The main class is `sh_gpc` (and sources live in `java/src/`):

```bash
# Create output directory for class files (if not exists)
mkdir -p java/bin

# Compile all sources to java/bin

```bash
javac sh_gpc.java
```


#### c) Run GP and capture the log

From the project root:

```bash
# If you compiled into java/bin:
java -cp java/bin sh_gpc experiments/level1/L1_train_XY.txt > output.txt

```

When the run completes, place the produced `output.txt` under:
```
experiments/artifacts/output.txt
```


#### d) Select GP expressions 

Now run:
```bash
python scripts/04_gp_select_new.py
```

This script will parse `experiments/artifacts/output.txt`, evaluate the candidate expressions, and write selection artifacts (e.g., `gp_candidates_test.csv`, `gp_selected_test.csv`) into `experiments/artifacts/`.


**Notes**

- Keep the **column order** in `L1_train_X.csv` consistent; GP typically refers to variables as `X1..XP` (1-based).
- If Java cannot find the main class, double-check the `-cp` (classpath) and that `sh_gpc.class` exists in `java/bin`.


- Parse GP outputs, evaluate continuous scores (with clipping), select via One‑SE + “simplest among ties”.  
- Writes `gp_candidates_test.csv`, `gp_selected_test.csv` in `experiments/artifacts/`.

### 5) Comprehensive evaluation (bootstrap) 
`python scripts/05_evaluation.py`  
- Bootstraps base models (original features) and meta‑models (L1 features).  
- For GP: AUC uses sigmoid(raw); others use hard labels.  
- Outputs under `experiments/level2/` and ranking tables.

### 6) Interpretability 
`python scripts/06_interpretation.py`  
- SHAP decision plots in **logit** space for selected base learners.  
- 1‑based `X1..Xk` substitution from `L1_test_X` into the selected GP expression.

### 7) Ablation
`python scripts/07_ablation.py`  
- Ranks L1 features (SHAP‑LR or |coef| fallback) and runs meta‑LR/meta‑XGB/meta‑GP across shrinking subsets; writes to `experiments/ablation/`.

---

## 5. Reproducibility

- Random seed fixed at `1412` across scripts.
- Optional libs are guarded to **skip** their models if not installed (the pipeline still runs).


---

## 6. Troubleshooting

- **ConvergenceWarning (sklearn/statsmodels)**: harmless. To hide globally, add at the top of scripts:  
  ```python
  import warnings
  from sklearn.exceptions import ConvergenceWarning
  warnings.filterwarnings("ignore", category=ConvergenceWarning)
  warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
  ```
- **TabNet save path**: it saves to `experiments/models/tabnet_model.zip` using its own API.  
