"""Shared configuration for the sepsis transportability study.

All paths can be redirected with the SEPSIS_ROOT environment variable so the
same code runs on the synthetic test data and on the real Challenge data.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SEPSIS_ROOT", str(Path(__file__).resolve().parents[1] / "work")))
DATA = ROOT / "data"
FEAT = ROOT / "features"
PRED = ROOT / "preds"
MODELS = ROOT / "models"
RES = ROOT / "results"
FIG = ROOT / "figures"
for _d in (DATA, FEAT, PRED, MODELS, RES, FIG):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261111
SITES = ["A", "B"]

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
DEMOG = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
CLIN = VITALS + LABS                      # 34 clinical variables
RAW_COLS = CLIN + DEMOG + ["SepsisLabel"]  # 41 columns, order of the PSV files
MONITORED = ["HR", "O2Sat", "SBP", "MAP", "DBP", "Resp"]  # routinely monitored
N_CLIN = len(CLIN)

ARMS = ["C", "V", "P", "VP"]
N_FOLDS = 5
INNER_HOLDOUT = 0.20       # patient-level early-stopping holdout
N_DRAWS_PRED = 5           # concealment draws averaged at prediction time (arm V)

# Gradient boosting configuration fixed in the analysis plan (version 1.0).
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    num_threads=2,
    verbosity=-1,
    force_col_wise=True,
    deterministic=True,
)
MAX_ROUNDS = 3000
EARLY_STOP = 50

N_BOOT = 1000
SENS_TARGET = 0.80         # patient-level sensitivity fixed in the development site
NB_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
AGE_BINS = [0, 50, 65, 80, 200]
AGE_LABELS = ["<50", "50-64", "65-79", ">=80"]
