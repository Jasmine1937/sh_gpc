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
├─ experiments/      #files will be generated automatically            
│  ├─ artifacts/
│  ├─ metrics/
│  ├─ predictions/
│  ├─ level1/
│  ├─ level2/
│  ├─ models/
│  └─ explain/
│  └─ labels/   # → create this folder manually
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
```bash
python scripts/01_train_base.py
```

A process similar to the figure below will be observed.

&emsp;<img width="344" height="35" alt="734b500c169ab8205c0bdebfc4d98ed" src="https://github.com/user-attachments/assets/6a22cd35-fcfa-436e-9d79-73b92f66c336" />

Outputs (files will be generated automatically):
- `experiments/predictions/val/{model}_proba.csv`, `{model}_pred.csv`  
- `experiments/metrics/val_metrics.csv` (AUC/ACC/PREC/REC/F1)  
- `experiments/models/{model}.joblib` (TabNet saves as `tabnet_model.zip`)


### 2) EDP selection (Effectiveness → Diversity → Pruning)  
Run: 

```bash
python scripts/02_select_base.py
```

A selection process similar to the figure below will be observed. Selected base classifiers are saved in `selected_base_models.json`. 

&emsp;<img width="457" height="109" alt="1f094808df22ac3044b984925edc431" src="https://github.com/user-attachments/assets/73319fff-d3cb-4144-b419-facad7a16f16" />

Outputs:
- `experiments/artifacts/effectiveness_pass.json`  
- `experiments/artifacts/diversity_matrix_phi.csv`  
- `experiments/artifacts/selected_base_models.json`

### 3) Build Level‑1 (stacking) dataset
Run: 
```bash
python scripts/03_build_level1_dataset.py`
```

A process similar to the figure below will be observed. 

&emsp;<img width="423" height="89" alt="1183aa01f9bf5e1a035b1980ca39017" src="https://github.com/user-attachments/assets/649a0843-1783-40e1-a979-8a7ecdb88649" />  

Outputs:
- `experiments/level1/L1_train_X.csv` (OOF)
- `experiments/level1/L1_test_X.csv` (K‑fold averaged)
- `experiments/level1/folds.csv`, `models_used.json`

Then put `y_tomtrain_s.csv` and `y_ogtest_s` to `experiments/labels/`.

### 4) Train Meta‑GP and selection  

- Java JDK (e.g., 17+) on your PATH  
**Running GP (Java) before `04_gp_select_new.py`**

Before executing `scripts/04_gp_select_new.py`, train GP on Java and produce a console log file (`output.txt`). The workflow:


#### a) Prepare the GP input (`L1_train_XY.txt`)

Start from the Level-1 files:
- `experiments/level1/L1_train_X.csv`  (features)
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
```

```bash
# Compile all sources to java/bin
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
- Outputs:
     - `experiments/artifacts/gp_candidates_test.csv`
     - `experiments/artifacts/gp_selected_test.csv`


### 5) Comprehensive evaluation (bootstrap) 
Run:
```bash
python scripts/05_evaluation.py
```

A process similar to the figure below will be observed. 

<img width="259" height="347" alt="22c21d5ce4726f36f4218bfe17ec396" src="https://github.com/user-attachments/assets/3af34ff4-cd86-4b73-a67b-933876d8c11a" />


- Bootstraps base models (original features) and meta‑models (L1 features).   
- Outputs:
     - `experiments/level2/base_bootstrap_test.csv`
     - `experiments/level2/combined_auc_ranking_test.csv`
     - `experiments/level2/combined_bootstrap_summary_test.csv`
     - `experiments/level2/combined_bootstrap_test.csv`
     - `experiments/level2/meta_bootstrap_test.csv`
  


### 6) Interpretability 
Run: 
```bash
python scripts/06_interpretation.py
```

This part outputs the decision plots of all base classifiers as well as a beeswarm plot for the most contributive base classifier. In addition, it produces the Sobol analysis results for the simplified GP expression.


- SHAP decision plots in **logit** space for selected base classifiers.  
- 1‑based `X1..Xk` substitution from `L1_test_X` into the selected GP expression.
- Outputs:
- `experiments/explain/decision_{model}.png`
- `experiments/explain/beeswarm_{model}.png`
- `experiments/explain/gp_meta_sobol.png`
- `experiments/explain/gp_meta_sobol.csv`
- `experiments/explain/L1_sample0_top3.png`
- `experiments/explain/sample0_baseSHAP_logit_and_GP.json`


### 7) Ablation study
Repeat step 4) Train Meta‑GP and selection and check details in `scripts/07_ablation.py`.

- Ranks L1 features (SHAP‑LR or |coef| fallback) and runs meta‑LR/meta‑XGB/meta‑GP across shrinking subsets; writes to `experiments/ablation/`.

---

## 5. Statistical Tests
The statistical testing was conducted using external software, with the input data provided in `combined_bootstrap_summary_test.csv`. Specifically, the DeLong test was performed with MedCalc (version 20.216), while the Friedman test was carried out using SPSS (version 19).


## 6. Reproducibility

- Random seed fixed at `1412` across scripts.
- Optional libs are guarded to **skip** their models if not installed (the pipeline still runs).

---

## 7. Troubleshooting

- **ConvergenceWarning (sklearn/statsmodels)**: harmless. To hide globally, add at the top of scripts:  
  ```python
  import warnings
  from sklearn.exceptions import ConvergenceWarning
  warnings.filterwarnings("ignore", category=ConvergenceWarning)
  warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
  ```
- **TabNet save path**: it saves to `experiments/models/tabnet_model.zip` using its own API.  
