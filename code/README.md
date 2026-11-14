# Measurement patterns versus physiology in early sepsis prediction

Code, fitted models and result files for the manuscript submitted to
*Intelligence-Based Medicine*.

## Data

PhysioNet/Computing in Cardiology Challenge 2019, public training sets A and B
(version 1.0.0, https://doi.org/10.13026/v64v-d857, CC BY 4.0). The data are
not redistributed here. `build_dataset.py` accepts either the original ZIP
archives of `.psv` files or the container archives described in the script,
and verifies every file's MD5 checksum when one is supplied.
`results/archive_sha256.txt` lists the SHA-256 of the archives used.

Expected after `build_dataset.py`: system A 20,336 patients / 790,215
patient-hours / 1,790 septic; system B 20,000 / 761,995 / 1,142.

## Requirements

Python 3.13 with numpy, pandas, scipy, scikit-learn 1.9, lightgbm 4.7,
statsmodels, pyarrow, matplotlib, pillow. The recurrent-network comparator
(`sens_gru.py`) needs TensorFlow 2.21 and was run in a separate environment.

Set `SEPSIS_ROOT` to a working directory (default `/home/claude/sepsis/work`)
and edit the `sys.path.insert` line at the top of each script if the code
lives elsewhere.

## Pipeline (run in this order)

| Step | Command | Output |
| --- | --- | --- |
| 1 | `python3 build_dataset.py <archives...>` | one parquet file per system |
| 2 | `python3 features.py` | feature matrices, patient table, folds |
| 3 | `python3 tests.py` | checks features and metrics against slow references |
| 4 | `python3 train.py external` | external models and predictions |
| 5 | `python3 train.py internal` | 5-fold out-of-fold local predictions |
| 6 | `python3 train.py pooled --arms VP` | two-site model |
| 7 | `python3 analysis_main.py` | primary results (`results/main_A.json`, `main_B.json`) |
| 8 | `python3 analysis_updating.py` | local recalibration |
| 9 | `python3 analysis_shap.py` | SHAP attributions |
| 10 | `python3 train.py external/internal --arms V --variant median` (and `strict`), `--arms VP,P,V --variant excl --exclude EtCO2`; `python3 sens_tuning.py`; `python3 sens_lr.py`; `sens_gru.py` | sensitivity predictions |
| 11 | `python3 analysis_sens.py`; `python3 sens_subsets.py`; `python3 analysis_extra.py` | `results/sensitivity.json`, `sens_subsets.json`, `alert_operating_points.json` |
| 12 | `python3 table1.py; python3 tables.py; python3 supp_tables.py; python3 figures.py` | tables and figures |
| 13 | `python3 build_manuscript.py; python3 assemble_full.py; python3 assemble_supp.py` | manuscript and supplement files |

`synth.py` writes synthetic data with the Challenge schema so that the whole
pipeline can be tested without patient data.

## Contents

- `config.py`: variables, seeds, the fixed LightGBM configuration.
- `features.py`: the four feature arms and the concealment procedure.
- `train.py`: fitting, early stopping, prediction.
- `metrics.py`: patient-level bootstrap, discrimination, calibration, net
  benefit, Challenge utility score, alerting.
- `models/`: fitted external models (LightGBM text format), one per system and arm.
- `results/`: every number reported in the manuscript.

Seeds are fixed. LightGBM runs with `deterministic=True`; small differences
between machines are possible. TensorFlow training is not bit-reproducible.
