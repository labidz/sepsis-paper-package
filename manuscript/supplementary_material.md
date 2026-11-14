# Supplementary material

Measurement patterns versus physiology in early sepsis prediction: a bidirectional external validation across two hospital systems

## Appendix A. Analysis plan and deviations

### A.1 Analysis plan (version 1.0)

The plan is reproduced as fixed; only cross-references were adapted.

Fixed on 11 November 2026, before the study data were opened. The plan was not registered externally. Any later deviation is listed at the end of this plan and reported in the supplement.

#### Aims

1. Measure the loss of performance when an hourly sepsis-prediction model developed in one hospital system is applied unchanged in the other, in both directions.
2. Separate how much of the predictive performance comes from physiological values and how much from the measurement process (which variables were measured, when and how often), and test whether each source transports.
3. Describe calibration, clinical utility and alert burden after transport, and test whether recalibration on limited local data restores them.
4. Describe performance by sex, age group and intensive care unit type.

#### Hypotheses

- **H1.** At the target site, the externally developed full model discriminates less well than a locally developed model evaluated on the same patients.
- **H2.** A model that sees only the measurement process, and no physiological value, discriminates better than a covariates-only model in internal validation.
- **H3.** The transport loss (local minus external AUROC) is larger for the process-only model than for the values-only model.

Each hypothesis is tested in both directions (A to B, B to A) and is considered supported only if the 95% confidence interval excludes zero in both. All other analyses are descriptive.

#### Data and outcome

- Source: public training sets A and B of the PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. All patients are included.
- Unit of prediction: the patient-hour, using only information recorded up to that hour.
- Outcome: the distributed SepsisLabel (positive from 6 hours before Sepsis-3 onset). Onset time is the first positive hour plus 6 hours. Patients positive at their first recorded hour are kept in the main analysis and excluded from lead-time estimates.

#### Feature arms

All arms include the covariates: age, sex, unit type, hours from hospital to unit admission, hours since unit admission.

| Arm | Values | Process | Content |
| --- | --- | --- | --- |
| C | No | No | Covariates only (reference) |
| V | Yes | No | Last value carried forward for 8 vital signs and 26 laboratory tests; trailing 6-hour and 24-hour mean, minimum and maximum for six routinely monitored vital signs; derived indices (shock index, pulse pressure, urea-to-creatinine ratio, SpO2-to-FiO2 ratio, SIRS count). Measurement status is concealed: a variable never yet measured receives a random draw from the development-site distribution of observed values, fixed per patient until a real value arrives; predictions average five draws |
| P | No | Yes | For each of the 34 variables: measured this hour, count so far, count in the trailing 24 hours, hours since last measurement; plus counts of distinct laboratory tests in the current hour, trailing 6 hours and trailing 24 hours |
| V+P | Yes | Yes | Union of V and P, with never-measured values left missing |

#### Model

Gradient-boosted decision trees (LightGBM, logistic loss) with one configuration fixed here for every arm and site: learning rate 0.05, 31 leaves, at least 200 patient-hours per leaf, column subsampling 0.5, row subsampling 0.8, L2 penalty 10, up to 3,000 trees with early stopping (50 rounds, log-loss) on a 20% patient-level holdout of the training patients. No resampling or class weighting.

#### Validation

- Internal: 5-fold cross-validation grouped by patient and stratified by sepsis status, separately in each site; out-of-fold predictions are pooled.
- External: the model developed on all patients of one site is applied unchanged to all patients of the other site.
- Reference at each target site: the locally developed model (that site's out-of-fold predictions).
- Secondary: a pooled two-site model, cross-validated.

#### Performance measures

All intervals come from 1,000 patient-level bootstrap resamples, paired for differences.

- Discrimination: AUROC (primary) and area under the precision-recall curve.
- Calibration: observed-to-expected ratio, calibration intercept and slope, integrated calibration index, Brier score, calibration curves.
- Clinical utility: net benefit across threshold probabilities of 0.5% to 10%; the Challenge utility score.
- Alerting: the threshold is fixed in the development site so that 80% of septic patients are alerted at or before onset. Reported at each site: sensitivity, proportion of non-septic patients alerted, positive predictive value, median lead time.
- Subgroups: sex, age (under 50, 50-64, 65-79, 80 or older), unit type.

#### Secondary and sensitivity analyses

1. Local updating: logistic recalibration of the external model on random samples of 100 to 5,000 target-site patients, evaluated in the remainder (50 repetitions).
2. Explanations: SHAP values for the full models, summarised by feature family and compared between sites.
3. Alternative concealment in arm V (median imputation; current values only).
4. Logistic regression as the learner for all four arms.
5. Excluding patients positive at their first recorded hour; excluding hours after onset.
6. Excluding variables recorded in under 1% of patients at either site.
7. Hyperparameter search for the full model, to show sensitivity to the fixed configuration.
8. A recurrent neural network on the raw hourly sequences as a deep-learning comparator.

### A.2 Clarifications, deviations and additions

The plan was fixed on 11 November 2026, before the study data were opened. It was not registered. The analysis followed the plan with these clarifications and additions.

1. Rarely recorded variables (sensitivity analysis 6). Only end-tidal carbon dioxide met the pre-specified criterion (recorded in under 1% of patients at either site; it was never recorded in system A). It was removed from all arms in that sensitivity analysis only; the primary models keep it.
2. Hyperparameter search (sensitivity analysis 7). Eight random configurations were fitted per development site. The configuration with the lowest holdout log-loss was cross-validated locally without nesting, which can only favour the local model.
3. Recurrent network (sensitivity analysis 8). The network (Appendix C.3) was trained on four cross-validation folds and evaluated on the fifth at both sites. It was not tuned systematically: the reported learning rate and batch size were chosen after an earlier run with a higher learning rate and larger batches trained unstably.
4. Analyses added after the data were seen, reported as exploratory: discrimination by time to onset (Table C.7, Fig. C.2); alerting at sensitivity targets from 50% to 90% (Table C.9); agreement of SHAP importance between the two models; alert measures within subgroups at the common threshold; re-derivation of the alert threshold in the local-updating samples.
5. Confidence intervals for calibration measures were computed from 2,000 quantile bins of the predictions within each bootstrap resample; point estimates of discrimination are exact.
6. The Shapley summary of the process share was not defined in the plan; it is reported as a descriptive summary, and H3 remains the pre-specified test. Re-setting the alert threshold on the full target cohort (Table 3) was also added after the plan.
7. The plan did not specify how the Challenge utility threshold was chosen for local models. In the first analysis it was chosen on the evaluation data, which favours the local model; it is now chosen, for each fold, on the out-of-fold predictions of the other folds (Table 3).
8. Net benefit after recalibration (plan aim 3) was added during internal review (Section 3.7).
9. Post hoc analyses added during internal review, reported as exploratory: the hypotheses and process shares by time to onset (Table C.10); removal of the ten variables whose patient-level availability differed more than twofold between systems (end-tidal CO2, base excess, bicarbonate, FiO2, pH, PaCO2, chloride, direct bilirubin, troponin I and partial thromboplastin time), together with the derived indices that use them and the laboratory-count features (Table C.2); removal of the process features of the variables that define organ dysfunction (platelets, total and direct bilirubin, creatinine, FiO2, pH, PaCO2, SaO2 and mean arterial pressure), of all measured-this-hour flags and of the laboratory counts, with their values kept (Table C.2); refitting of all arms with four further random seeds, which change the early-stopping holdout, the boosting randomness and the concealment draws but not the folds (Table C.2); and refitting of the full model on 500 to 5,000 target-site patients, from scratch or by further boosting of the external model (Table C.11).

## Appendix B. TRIPOD+AI checklist

| Item | Topic | Where reported |
| --- | --- | --- |
| 1 | Title | Title |
| 2 | Abstract | Abstract |
| 3a | Background and rationale | Section 1 |
| 3b | Target population and intended use | Sections 1 and 2.1 |
| 3c | Known health inequalities | Sections 2.8 and 4.5 (the dataset records sex and age only) |
| 4 | Objectives | Section 1, last paragraph; Section 2.1 |
| 5a | Data sources | Section 2.2 |
| 5b | Dates of data collection | Section 2.2 (not released with the dataset) |
| 6a | Setting and centres | Section 2.2 |
| 6b | Eligibility criteria | Section 2.2 (all patients in the public training sets) |
| 6c | Treatments received | Not available in the dataset |
| 7 | Data preparation | Section 2.4; Appendix C.1 and C.2 |
| 8a | Outcome definition | Section 2.3 |
| 8b | Subjective outcome assessment | Not applicable (automated label) |
| 8c | Blinding of outcome assessment | Section 2.3 |
| 9a | Choice of predictors | Section 2.4 |
| 9b | Predictor definitions and timing | Section 2.4; Appendix C.2 |
| 9c | Subjective predictor assessment | Not applicable |
| 10 | Sample size | Section 2.5; Table 1 |
| 11 | Missing data | Section 2.4; Appendix C.2 |
| 12a | Use of data for development and evaluation | Sections 2.5 and 2.6; Fig. 1 |
| 12b | Handling of predictors | Section 2.4; Appendix C.2 |
| 12c | Model type, building and internal validation | Sections 2.5 and 2.6; Appendix C.3 |
| 12d | Heterogeneity across clusters | Sections 2.6 and 3.2 (two systems, reported separately) |
| 12e | Performance measures | Section 2.7; Appendix C.4 |
| 12f | Model updating | Sections 2.8 and 3.7 |
| 12g | How predictions were calculated | Appendix C.3; code and fitted models in the supplementary package |
| 13 | Class imbalance | Section 2.5 (no resampling or reweighting) |
| 14 | Fairness | Sections 2.8 and 3.6; Table C.3 |
| 15 | Model output and thresholds | Section 2.7 |
| 16 | Differences between development and evaluation data | Table 1; Section 3.1; Table C.4; Fig. C.1 |
| 17 | Ethical approval | Section 2.9; Ethics statement |
| 18a | Funding | Funding statement |
| 18b | Conflicts of interest | Declaration of competing interest |
| 18c | Protocol | Appendix A.1 (analysis plan reproduced in full) |
| 18d | Registration | Section 2.1 (not registered) |
| 18e | Data sharing | Data availability |
| 18f | Code sharing | Data availability |
| 19 | Patient and public involvement | Section 2.9 (none) |
| 20a | Flow of participants | Sections 2.2 and 3.1; Table 1 (no exclusions) |
| 20b | Participant characteristics | Table 1; Table C.4 |
| 20c | Comparison of development and evaluation data | Table 1; Section 3.1 |
| 21 | Participants and events in each analysis | Table 1; Table C.8 |
| 22 | Full model specification | Fitted models and code in the supplementary package |
| 23a | Performance with confidence intervals, including subgroups | Tables 2 and 3; Table C.3; Figs 2 to 4 |
| 23b | Heterogeneity in performance | Tables 2 and 3 (by system); Table C.3 |
| 24 | Results of model updating | Section 3.7; Table C.5; Fig. 4 |
| 25 | Interpretation | Sections 4.1 to 4.4 |
| 26 | Limitations | Section 4.5 |
| 27a | Handling of poor or missing input data in use | Sections 2.1 and 2.4 (research models; missing inputs handled natively) |
| 27b | User interaction and expertise | Section 2.1 (not proposed for deployment) |
| 27c | Next steps | Section 4.5, last sentence |

## Appendix C. Supplementary methods and results

### C.1 Data verification and preparation

The 40,336 patient files (20,336 from system A and 20,000 from system B) were read from the PhysioNet open-data repository. The byte length and MD5 checksum of every file matched the repository listing. The data comprise 1,552,210 patient-hours, of which 27,916 are labelled positive, and 2,932 patients with sepsis; these counts agree with the published description of the dataset. The first recorded hour was later than the first hour in the unit for 36.9% of patients in A and 6.5% in B; hours since unit admission were used as recorded. No value was removed or altered, with one exception: FiO2 values above 1 were treated as percentages and divided by 100, and FiO2 was then bounded to 0.21 to 1.0 when the SpO2-to-FiO2 ratio was computed. Unit type was coded as medical, surgical or not recorded.

### C.2 Feature definitions

Covariates (5): age; sex; unit type; hours from hospital admission to unit admission; hours since unit admission.

Process features (139). For each of the 34 clinical variables: whether it was measured in the current hour; the number of measurements so far; the number in the trailing 24 hours; and hours since the last measurement (missing if never measured). In addition: the number of distinct laboratory variables measured in the current hour, in the trailing 6 hours and in the trailing 24 hours.

Value features (75). For each of the 34 clinical variables, the last value carried forward. For heart rate, oxygen saturation, systolic, mean and diastolic arterial pressure and respiratory rate: the mean, minimum and maximum of the carried-forward series over the trailing 6 and 24 hours. Derived indices: shock index (heart rate divided by systolic pressure); pulse pressure; urea-to-creatinine ratio; SpO2-to-FiO2 ratio; and the number of systemic inflammatory response criteria met (temperature above 38 or below 36 degrees Celsius; heart rate above 90 per minute; respiratory rate above 20 per minute or PaCO2 below 32 mm Hg; white cell count above 12 or below 4 thousand per microlitre).

Concealment in arm V. A value feature that is undefined because the variable has not yet been measured for that patient was replaced by a random draw from the observed values of that variable among the development patients used to fit the model (one draw per patient and variable, reused for the trailing statistics of that variable). Derived indices were computed after replacement. At prediction time five independent sets of draws were used and the predicted probabilities averaged. For a variable never observed in the development data the replacement was a constant. In arm VP undefined values were left missing and handled natively by the learner.

Every feature at hour t uses only data recorded up to and including hour t. The feature code was checked against slow reference implementations on 30 randomly chosen patients per site.

### C.3 Model fitting

Models were gradient-boosted decision trees (LightGBM 4.7, logistic loss) with the configuration given in Section 2.5, up to 3,000 trees and early stopping after 50 rounds without improvement in log-loss on a 20% patient-level holdout of the development patients (stratified by sepsis status). Table C.8 gives the number of fits, features, trees and training patient-hours. For external validation the model was developed on all patients of one system (80% for fitting, 20% for early stopping) and applied to all patients of the other. For internal validation the same procedure was applied within each of five training partitions. Fold assignment, holdout splits and random draws used fixed seeds. The fitted external models are provided as LightGBM text files.

Hyperparameter search (sensitivity analysis 7). Eight configurations per development site were drawn at random from learning rate {0.02, 0.05, 0.1}, leaves {15, 31, 63, 127}, minimum patient-hours per leaf {50, 200, 1,000}, column subsampling {0.3, 0.5, 0.8}, row subsampling {0.6, 0.8, 1.0} and L2 penalty {1, 10, 100}, and fitted to the full feature set. The configuration with the lowest log-loss on the development holdout was selected: for system A, learning rate 0.05, 15 leaves, 1,000 patient-hours per leaf, column subsampling 0.5, row subsampling 0.6, L2 penalty 10; for system B, 0.05, 63, 1,000, 0.5, 0.8 and 100.

Logistic regression (sensitivity analysis 4). L2-penalised logistic regression (scikit-learn, C = 1, L-BFGS) was fitted to the same development rows as the boosted model in every arm. Counts and hours were log-transformed, values were winsorised at the 0.5th and 99.5th training percentiles, hours since last measurement were capped at 168 with an indicator for a variable never measured, remaining missing values (arm VP only) were set to the training median, and all inputs were standardised.

Recurrent network (sensitivity analysis 8). One layer of 64 gated recurrent units, dropout 0.2 and a sigmoid output at every hour. Inputs at each hour were, for each of the 34 clinical variables, the standardised last value (0 if never measured), a flag for a measurement in that hour and the scaled time since the last measurement, plus the covariates. Training used Adam (learning rate 0.0005, gradient-norm clipping 1.0, batches of 64 patients) and early stopping on the log-loss of a 20% patient-level holdout (patience 5 epochs, at most 40).

### C.4 Performance measures and resampling

AUROC and AUPRC were computed across patient-hours. Calibration intercept and slope came from logistic regression of the outcome on the logit of the prediction (intercept with the slope fixed at 1). The integrated calibration index is the mean absolute difference between the prediction and a smoothed observed proportion, here a logistic model with a restricted cubic spline (five knots) of the logit of the prediction. Net benefit at threshold probability p is (true positives minus false positives multiplied by p/(1 - p)) divided by the number of patient-hours. The Challenge utility score was re-implemented and checked against the logic of the organisers' scoring code; its threshold was chosen on the development site's out-of-fold predictions.

For alerting, onset was taken as the first positive hour plus 6 hours. A septic patient counted as alerted if the prediction reached the threshold at any hour up to onset; a non-septic patient if it reached the threshold at any hour. Lead time is the interval between the first alert and onset and excludes patients whose label was positive at the first recorded hour.

Confidence intervals are percentile intervals from 1,000 bootstrap resamples of patients. The same resamples were used for all models evaluated at a site, so that every difference is paired.

### C.5 Supplementary tables

**Table C.1.** Two-site model (full feature set), five-fold cross-validation.

| Measure | System B | System A |
| --- | --- | --- |
| AUROC, two-site model | 0.861 (0.852 to 0.870) | 0.825 (0.816 to 0.833) |
| AUROC, difference from local model | -0.004 (-0.008 to -0.001) | -0.001 (-0.004 to 0.003) |
| AUROC, difference from external model | 0.076 (0.067 to 0.086) | 0.050 (0.044 to 0.056) |
| AUPRC, two-site model | 0.122 (0.107 to 0.134) | 0.115 (0.107 to 0.124) |
| Observed-to-expected ratio | 0.95 (0.89 to 1.01) | 0.95 (0.91 to 1.00) |
| Calibration slope | 1.00 (0.95 to 1.04) | 0.94 (0.90 to 0.98) |

**Table C.2.** Sensitivity analyses. Estimates are differences in AUROC (95% confidence intervals); transport loss is the local minus the external AUROC on the same patients. Differences were computed before rounding.

| Specification | Target | H1: transport loss of the full model (local minus external) | H2: process minus covariates, local | H3: excess loss of process over values | Process share of gain, local / external |
| --- | --- | --- | --- | --- | --- |
| Primary analysis | B | 0.081 (0.071 to 0.091) | 0.079 (0.069 to 0.092) | 0.032 (0.021 to 0.043) | 48% / 35% |
| Primary analysis | A | 0.051 (0.044 to 0.058) | 0.066 (0.057 to 0.074) | 0.032 (0.024 to 0.040) | 55% / 34% |
| Values arm: median imputation instead of random draws | B | 0.081 (0.071 to 0.091) | 0.079 (0.069 to 0.092) | 0.032 (0.022 to 0.043) | 43% / 29% |
| Values arm: median imputation instead of random draws | A | 0.051 (0.044 to 0.058) | 0.066 (0.057 to 0.074) | 0.020 (0.013 to 0.028) | 48% / 34% |
| Values arm: current values only (no trailing statistics) | B | 0.081 (0.071 to 0.091) | 0.079 (0.069 to 0.092) | 0.040 (0.029 to 0.052) | 52% / 35% |
| Values arm: current values only (no trailing statistics) | A | 0.051 (0.044 to 0.058) | 0.066 (0.057 to 0.074) | 0.027 (0.019 to 0.034) | 58% / 41% |
| End-tidal CO2 excluded from all arms | B | 0.076 (0.066 to 0.086) | 0.073 (0.062 to 0.085) | 0.029 (0.018 to 0.039) | 47% / 35% |
| End-tidal CO2 excluded from all arms | A | 0.047 (0.040 to 0.054) | 0.066 (0.057 to 0.074) | 0.027 (0.020 to 0.035) | 55% / 38% |
| Logistic regression as the learner (all arms) | B | 0.105 (0.091 to 0.118) | 0.163 (0.145 to 0.180) | 0.026 (0.011 to 0.042) | 61% / 58% |
| Logistic regression as the learner (all arms) | A | 0.060 (0.051 to 0.070) | 0.111 (0.099 to 0.123) | 0.028 (0.017 to 0.039) | 63% / 53% |
| Full model with hyperparameter search | B | 0.088 (0.077 to 0.098) | not available | not available | not available |
| Full model with hyperparameter search | A | 0.052 (0.044 to 0.059) | not available | not available | not available |
| Patients positive at first recorded hour excluded | B | 0.064 (0.053 to 0.074) | 0.089 (0.075 to 0.103) | 0.034 (0.022 to 0.047) | 48% / 30% |
| Patients positive at first recorded hour excluded | A | 0.051 (0.044 to 0.058) | 0.067 (0.058 to 0.076) | 0.030 (0.022 to 0.038) | 55% / 36% |
| Hours after onset excluded | B | 0.078 (0.067 to 0.089) | 0.065 (0.054 to 0.078) | 0.023 (0.012 to 0.035) | 45% / 36% |
| Hours after onset excluded | A | 0.059 (0.051 to 0.066) | 0.061 (0.053 to 0.070) | 0.037 (0.029 to 0.046) | 55% / 24% |
| Variables with more than twofold difference in availability removed (10 variables) | B | 0.082 (0.072 to 0.093) | 0.069 (0.058 to 0.082) | 0.024 (0.014 to 0.035) | 45% / 34% |
| Variables with more than twofold difference in availability removed (10 variables) | A | 0.055 (0.047 to 0.063) | 0.051 (0.042 to 0.059) | 0.039 (0.030 to 0.048) | 47% / 14% |
| Process features of organ-dysfunction variables, all measured-this-hour flags and laboratory counts removed | B | 0.084 (0.074 to 0.095) | 0.072 (0.061 to 0.085) | 0.038 (0.026 to 0.049) | 45% / 29% |
| Process features of organ-dysfunction variables, all measured-this-hour flags and laboratory counts removed | A | 0.050 (0.043 to 0.057) | 0.055 (0.046 to 0.062) | 0.036 (0.028 to 0.044) | 49% / 23% |
| Refitted with new random seeds: median (range) over 5 fits | B | 0.081 (0.075 to 0.090) | 0.081 (0.078 to 0.082) | 0.020 (0.015 to 0.032) | 48% (48% to 50%) / 41% (35% to 43%) |
| Refitted with new random seeds: median (range) over 5 fits | A | 0.057 (0.051 to 0.060) | 0.067 (0.065 to 0.069) | 0.027 (0.020 to 0.032) | 54% (53% to 55%) / 36% (32% to 40%) |
| Recurrent network (one test fold, 4,000 patients) | B | 0.050 (0.019 to 0.085); boosted model on the same patients 0.076 (0.056 to 0.099) | not applicable | not applicable | not applicable |
| Recurrent network (one test fold, 4,068 patients) | A | 0.130 (0.104 to 0.156); boosted model on the same patients 0.047 (0.033 to 0.061) | not applicable | not applicable | not applicable |

**Table C.3.** Subgroup performance of the full model. Alert measures use the local threshold for the local model and the development-site threshold for the external model.

| Target | Subgroup | Patients (septic), n | Local AUROC | External AUROC | External minus local | O:E local / external | Septic alerted, local / external, % | Non-septic alerted, local / external, % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Sex: female | 9,268 (495) | 0.867 (0.853 to 0.880) | 0.790 (0.771 to 0.808) | -0.077 (-0.093 to -0.061) | 0.95 / 0.91 | 79 / 56 | 24 / 16 |
| B | Sex: male | 10,732 (647) | 0.863 (0.852 to 0.875) | 0.780 (0.765 to 0.796) | -0.083 (-0.097 to -0.071) | 0.99 / 0.90 | 81 / 57 | 26 / 18 |
| B | Age: under 50 | 4,818 (276) | 0.857 (0.839 to 0.875) | 0.762 (0.734 to 0.787) | -0.095 (-0.120 to -0.073) | 0.97 / 0.87 | 83 / 54 | 26 / 16 |
| B | Age: 50-64 | 6,278 (351) | 0.870 (0.854 to 0.884) | 0.789 (0.768 to 0.808) | -0.081 (-0.099 to -0.063) | 0.95 / 0.93 | 76 / 54 | 25 / 16 |
| B | Age: 65-79 | 6,582 (386) | 0.868 (0.853 to 0.883) | 0.796 (0.776 to 0.817) | -0.072 (-0.089 to -0.055) | 0.95 / 0.88 | 81 / 62 | 26 / 18 |
| B | Age: 80 or older | 2,322 (129) | 0.864 (0.839 to 0.890) | 0.790 (0.758 to 0.821) | -0.075 (-0.102 to -0.048) | 1.13 / 1.03 | 80 / 52 | 26 / 18 |
| B | Unit: medical | 6,923 (390) | 0.859 (0.843 to 0.874) | 0.769 (0.749 to 0.791) | -0.090 (-0.109 to -0.071) | 1.02 / 0.83 | 77 / 58 | 29 / 23 |
| B | Unit: surgical | 6,982 (428) | 0.863 (0.847 to 0.878) | 0.786 (0.766 to 0.805) | -0.078 (-0.093 to -0.061) | 0.91 / 0.89 | 82 / 57 | 25 / 17 |
| B | Unit: not recorded | 6,095 (324) | 0.876 (0.861 to 0.892) | 0.802 (0.780 to 0.822) | -0.074 (-0.094 to -0.054) | 1.01 / 1.05 | 81 / 53 | 22 / 11 |
| A | Sex: female | 8,502 (698) | 0.827 (0.813 to 0.840) | 0.780 (0.765 to 0.794) | -0.047 (-0.058 to -0.036) | 0.92 / 0.79 | 80 / 86 | 34 / 61 |
| A | Sex: male | 11,834 (1,092) | 0.825 (0.815 to 0.836) | 0.772 (0.760 to 0.784) | -0.053 (-0.062 to -0.044) | 0.98 / 0.84 | 80 / 85 | 32 / 60 |
| A | Age: under 50 | 4,358 (379) | 0.832 (0.815 to 0.848) | 0.782 (0.762 to 0.799) | -0.051 (-0.066 to -0.037) | 0.98 / 0.78 | 81 / 87 | 34 / 66 |
| A | Age: 50-64 | 5,958 (512) | 0.828 (0.812 to 0.844) | 0.785 (0.769 to 0.802) | -0.043 (-0.056 to -0.029) | 0.96 / 0.87 | 80 / 85 | 31 / 57 |
| A | Age: 65-79 | 6,975 (643) | 0.825 (0.811 to 0.839) | 0.763 (0.747 to 0.780) | -0.062 (-0.074 to -0.050) | 0.95 / 0.84 | 80 / 84 | 32 / 58 |
| A | Age: 80 or older | 3,045 (256) | 0.814 (0.790 to 0.837) | 0.777 (0.753 to 0.800) | -0.037 (-0.058 to -0.015) | 0.93 / 0.75 | 79 / 89 | 36 / 66 |
| A | Unit: medical | 5,344 (576) | 0.795 (0.780 to 0.810) | 0.727 (0.710 to 0.744) | -0.068 (-0.082 to -0.054) | 1.03 / 1.00 | 81 / 84 | 43 / 70 |
| A | Unit: surgical | 5,470 (222) | 0.846 (0.822 to 0.870) | 0.810 (0.781 to 0.837) | -0.036 (-0.057 to -0.014) | 0.88 / 0.58 | 57 / 77 | 12 / 41 |
| A | Unit: not recorded | 9,522 (992) | 0.819 (0.807 to 0.830) | 0.783 (0.771 to 0.795) | -0.036 (-0.045 to -0.028) | 0.93 / 0.81 | 84 / 88 | 39 / 67 |

**Table C.4.** Availability of each clinical variable by hospital system.

| Variable | A: patients with a measurement, % | B: patients with a measurement, % | A: hours with a measurement, % | B: hours with a measurement, % |
| --- | --- | --- | --- | --- |
| HR | 100.0 | 100.0 | 92.3 | 87.9 |
| O2Sat | 99.9 | 100.0 | 88.0 | 85.9 |
| Temp | 98.8 | 99.8 | 33.8 | 33.9 |
| SBP | 98.7 | 99.9 | 84.8 | 86.1 |
| MAP | 100.0 | 99.5 | 89.8 | 85.2 |
| DBP | 63.7 | 99.9 | 51.9 | 86.1 |
| Resp | 99.9 | 99.8 | 90.2 | 78.9 |
| EtCO2 | 0.0 | 16.1 | 0.0 | 7.6 |
| BaseExcess | 62.2 | 2.8 | 10.4 | 0.2 |
| HCO3 | 97.4 | 2.1 | 8.1 | 0.2 |
| FiO2 | 58.9 | 29.1 | 14.2 | 2.3 |
| pH | 64.8 | 28.8 | 11.5 | 2.2 |
| PaCO2 | 61.9 | 28.9 | 8.8 | 2.2 |
| SaO2 | 39.2 | 25.6 | 5.0 | 1.9 |
| AST | 29.0 | 42.3 | 1.5 | 1.8 |
| BUN | 97.9 | 92.0 | 8.2 | 5.5 |
| Alkalinephos | 28.0 | 42.4 | 1.5 | 1.8 |
| Calcium | 81.4 | 92.2 | 5.0 | 6.8 |
| Chloride | 97.3 | 8.1 | 8.3 | 0.6 |
| Creatinine | 97.7 | 92.1 | 6.6 | 5.5 |
| Bilirubin_direct | 2.9 | 7.4 | 0.1 | 0.2 |
| Glucose | 98.0 | 94.1 | 12.2 | 22.2 |
| Lactate | 38.0 | 23.8 | 3.4 | 1.9 |
| Magnesium | 93.2 | 82.3 | 7.8 | 4.8 |
| Phosphate | 82.0 | 58.2 | 5.0 | 2.9 |
| Potassium | 97.9 | 92.8 | 10.9 | 7.7 |
| Bilirubin_total | 28.4 | 42.4 | 1.2 | 1.8 |
| TroponinI | 2.4 | 32.8 | 0.1 | 1.8 |
| Hct | 98.2 | 90.2 | 11.8 | 5.8 |
| Hgb | 97.5 | 90.3 | 8.8 | 5.9 |
| PTT | 77.9 | 22.0 | 4.8 | 1.0 |
| WBC | 96.9 | 90.0 | 7.5 | 5.3 |
| Fibrinogen | 12.6 | 9.7 | 0.8 | 0.6 |
| Platelets | 97.1 | 90.0 | 6.5 | 5.3 |

**Table C.5.** Logistic recalibration of the external full model on random samples of target-site patients (50 repetitions per sample size).

| Direction | Local patients used | Septic patients in sample, median | ICI before, percentage points | ICI after, median (2.5th to 97.5th percentile) | Repetitions improved, % | O:E after | Slope after | Septic alerted with locally re-set threshold, % | Non-septic alerted with locally re-set threshold, % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A to B | 100 | 6 | 0.37 | 0.80 (0.31 to 1.80) | 14 | 0.94 (0.44 to 2.61) | 0.86 (0.57 to 4.65) | 83 (53 to 96) | 52 (15 to 92) |
| A to B | 250 | 14 | 0.37 | 0.40 (0.24 to 1.18) | 38 | 0.99 (0.63 to 1.85) | 0.98 (0.64 to 1.60) | 81 (62 to 94) | 49 (23 to 76) |
| A to B | 500 | 27 | 0.38 | 0.37 (0.24 to 1.02) | 50 | 1.04 (0.63 to 1.49) | 0.93 (0.72 to 1.47) | 80 (60 to 91) | 45 (21 to 70) |
| A to B | 1,000 | 58 | 0.37 | 0.29 (0.24 to 0.58) | 84 | 0.96 (0.79 to 1.24) | 0.99 (0.78 to 1.23) | 80 (72 to 89) | 45 (34 to 65) |
| A to B | 2,500 | 145 | 0.38 | 0.27 (0.22 to 0.39) | 98 | 1.00 (0.82 to 1.19) | 0.98 (0.86 to 1.16) | 79 (74 to 85) | 45 (39 to 54) |
| A to B | 5,000 | 281 | 0.38 | 0.26 (0.20 to 0.32) | 98 | 1.01 (0.89 to 1.20) | 1.00 (0.91 to 1.09) | 80 (75 to 83) | 45 (40 to 50) |
| B to A | 100 | 8 | 0.63 | 0.59 (0.28 to 1.55) | 54 | 0.97 (0.70 to 2.74) | 0.90 (0.57 to 1.81) | 80 (58 to 97) | 51 (18 to 85) |
| B to A | 250 | 23 | 0.63 | 0.56 (0.23 to 1.20) | 58 | 0.90 (0.64 to 1.45) | 0.98 (0.69 to 1.45) | 80 (61 to 94) | 48 (21 to 76) |
| B to A | 500 | 46 | 0.63 | 0.31 (0.14 to 0.84) | 86 | 0.97 (0.74 to 1.32) | 1.00 (0.77 to 1.32) | 80 (65 to 89) | 48 (25 to 65) |
| B to A | 1,000 | 91 | 0.63 | 0.24 (0.11 to 0.46) | 100 | 0.97 (0.84 to 1.14) | 0.98 (0.85 to 1.16) | 80 (72 to 90) | 51 (32 to 66) |
| B to A | 2,500 | 218 | 0.63 | 0.20 (0.12 to 0.38) | 100 | 1.03 (0.90 to 1.18) | 1.00 (0.91 to 1.12) | 80 (72 to 84) | 51 (34 to 57) |
| B to A | 5,000 | 443 | 0.63 | 0.16 (0.11 to 0.25) | 100 | 1.00 (0.91 to 1.08) | 1.01 (0.93 to 1.07) | 80 (76 to 83) | 51 (40 to 55) |

**Table C.6.** Most influential features of the two full models by mean absolute SHAP value (2,000 randomly sampled patients per system).

| Rank | Model developed in A: feature (family), share % | Model developed in B: feature (family), share % |
| --- | --- | --- |
| 1 | FiO2 hours since (process), 9.2 | HospAdmTime (covariate), 7.6 |
| 2 | Temp last (value), 5.6 | Temp last (value), 6.1 |
| 3 | ICULOS (covariate), 4.5 | ICULOS (covariate), 5.4 |
| 4 | UnitType (covariate), 4.1 | Creatinine last (value), 2.8 |
| 5 | Creatinine last (value), 4.1 | EtCO2 hours since (process), 2.8 |
| 6 | Lactate hours since (process), 3.0 | BUN last (value), 2.8 |
| 7 | HospAdmTime (covariate), 3.0 | WBC last (value), 2.6 |
| 8 | WBC last (value), 2.6 | SIRS (value), 2.3 |
| 9 | Resp min24h (value), 2.6 | FiO2 hours since (process), 2.3 |
| 10 | O2Sat count (process), 2.4 | HR max6h (value), 2.1 |
| 11 | Temp count24h (process), 2.2 | nlab 6h (process), 1.9 |
| 12 | PTT last (value), 2.1 | Resp count24h (process), 1.9 |
| Family shares on development data (process / value / covariate), % | 35 / 52 / 13 | 33 / 52 / 16 |
| Family shares on the other system's data, % | 33 / 53 / 14 | 37 / 47 / 16 |

**Table C.7.** Exploratory analysis: AUROC for separating septic patient-hours in a window relative to onset from all hours of non-septic patients.

| Target | Hours relative to onset | Positive hours, n | C local / external | V local / external | P local / external | VP local / external |
| --- | --- | --- | --- | --- | --- | --- |
| B | -24 to -13 h | 8,771 | 0.748 / 0.702 | 0.814 / 0.774 | 0.781 / 0.726 | 0.818 / 0.769 |
| B | -12 to -7 h | 5,185 | 0.750 / 0.698 | 0.824 / 0.777 | 0.789 / 0.727 | 0.829 / 0.776 |
| B | -6 to -1 h | 6,811 | 0.773 / 0.669 | 0.848 / 0.773 | 0.834 / 0.738 | 0.872 / 0.794 |
| B | 0 to +3 h | 3,969 | 0.768 / 0.704 | 0.868 / 0.811 | 0.880 / 0.772 | 0.912 / 0.828 |
| A | -24 to -13 h | 14,306 | 0.779 / 0.762 | 0.827 / 0.792 | 0.846 / 0.796 | 0.855 / 0.803 |
| A | -12 to -7 h | 8,894 | 0.771 / 0.751 | 0.824 / 0.789 | 0.839 / 0.775 | 0.853 / 0.790 |
| A | -6 to -1 h | 10,740 | 0.772 / 0.735 | 0.828 / 0.787 | 0.840 / 0.760 | 0.857 / 0.790 |
| A | 0 to +3 h | 6,396 | 0.777 / 0.722 | 0.844 / 0.803 | 0.858 / 0.794 | 0.877 / 0.836 |

**Table C.8.** Model fitting summary for the primary analysis.

| Stage | Arm | Fits, n | Features, n | Trees at early stopping, median (range) | Training patient-hours, median | Septic patients in the development data, n |
| --- | --- | --- | --- | --- | --- | --- |
| external | C | 2 | 5 | 66 (65 to 68) | 620,340 | 1,790 (A) / 1,142 (B) |
| external | P | 2 | 144 | 116 (92 to 139) | 620,340 | 1,790 (A) / 1,142 (B) |
| external | V | 2 | 80 | 113 (110 to 116) | 620,340 | 1,790 (A) / 1,142 (B) |
| external | VP | 2 | 219 | 138 (136 to 141) | 620,340 | 1,790 (A) / 1,142 (B) |
| internal | C | 10 | 5 | 60 (35 to 82) | 497,212 | 1,432 (A) / 914 (B), median per fold |
| internal | P | 10 | 144 | 80 (56 to 106) | 497,212 | 1,432 (A) / 914 (B), median per fold |
| internal | V | 10 | 80 | 94 (83 to 167) | 497,212 | 1,432 (A) / 914 (B), median per fold |
| internal | VP | 10 | 219 | 124 (80 to 217) | 497,212 | 1,432 (A) / 914 (B), median per fold |
| pooled | VP | 5 | 219 | 155 (82 to 172) | 993,159 | 2,346, median per fold |

**Table C.9.** Exploratory analysis: alerting at other sensitivity targets. The threshold was set at the development site to alert the stated percentage of septic patients by onset and applied unchanged at the target site; the last two columns show the proportion of non-septic patients alerted when the threshold is set at the target site to reach the same sensitivity.

| Direction | Sensitivity target at development site, % | External site, development threshold: septic alerted, % | External site, development threshold: non-septic alerted, % | Non-septic alerted at the target sensitivity, external model, % | Non-septic alerted at the target sensitivity, local model, % |
| --- | --- | --- | --- | --- | --- |
| A to B | 50 | 33.7 | 2.7 | 11.0 | 5.4 |
| A to B | 60 | 38.8 | 4.6 | 20.9 | 9.1 |
| A to B | 70 | 46.3 | 8.7 | 31.8 | 15.9 |
| A to B | 80 | 56.5 | 17.1 | 45.3 | 25.5 |
| A to B | 90 | 72.4 | 34.6 | 67.1 | 42.1 |
| B to A | 50 | 55.0 | 15.8 | 12.7 | 7.5 |
| B to A | 60 | 66.9 | 26.1 | 19.7 | 13.1 |
| B to A | 70 | 77.4 | 40.3 | 29.9 | 20.9 |
| B to A | 80 | 85.5 | 60.4 | 50.8 | 32.5 |
| B to A | 90 | 94.6 | 78.7 | 67.7 | 49.4 |

**Table C.10.** Post hoc analysis by time to onset: hypotheses and process shares when septic patient-hours in each window are separated from all hours of non-septic patients (95% confidence intervals).

| Target | Hours relative to onset | H1: transport loss, full model | H2: process minus covariates, local | H3: excess loss of process over values | Process share of gain, local / external |
| --- | --- | --- | --- | --- | --- |
| B | -24 to -13 h | 0.050 (0.036 to 0.063) | 0.033 (0.020 to 0.048) | 0.014 (-0.001 to 0.029) | 26% / 14% |
| B | -12 to -7 h | 0.054 (0.040 to 0.067) | 0.038 (0.024 to 0.052) | 0.015 (0.001 to 0.031) | 27% / 18% |
| B | -6 to -1 h | 0.077 (0.066 to 0.089) | 0.061 (0.050 to 0.074) | 0.021 (0.009 to 0.034) | 43% / 36% |
| B | 0 to +3 h | 0.084 (0.074 to 0.095) | 0.113 (0.099 to 0.129) | 0.051 (0.038 to 0.065) | 54% / 35% |
| A | -24 to -13 h | 0.052 (0.045 to 0.061) | 0.067 (0.055 to 0.078) | 0.015 (0.006 to 0.024) | 62% / 55% |
| A | -12 to -7 h | 0.062 (0.054 to 0.071) | 0.068 (0.059 to 0.079) | 0.029 (0.020 to 0.039) | 59% / 32% |
| A | -6 to -1 h | 0.067 (0.058 to 0.075) | 0.068 (0.059 to 0.077) | 0.040 (0.031 to 0.048) | 57% / 25% |
| A | 0 to +3 h | 0.041 (0.034 to 0.048) | 0.081 (0.072 to 0.091) | 0.023 (0.015 to 0.032) | 57% / 46% |

**Table C.11.** Post hoc analysis: full models refitted on n target-site patients (5 repetitions per size), from scratch or by further boosting of the external model, evaluated on a fixed random half of the target-site patients that was never used for fitting. The reference rows give the unchanged external model and the cross-validated local model on the same evaluation patients.

| Target | Local patients used (septic, median) | New local model, AUROC median (range) | External model updated by further boosting, AUROC median (range) | Trees added, median |
| --- | --- | --- | --- | --- |
| B | Reference: external model unchanged | 0.782 | | |
| B | Reference: cross-validated local model (about 13,000 patients) | 0.866 | | |
| B | 500 (27) | 0.777 (0.758 to 0.795) | 0.794 (0.785 to 0.796) | 3 |
| B | 1,000 (53) | 0.799 (0.789 to 0.810) | 0.818 (0.810 to 0.828) | 48 |
| B | 2,500 (151) | 0.830 (0.824 to 0.839) | 0.830 (0.809 to 0.836) | 28 |
| B | 5,000 (302) | 0.847 (0.845 to 0.853) | 0.842 (0.835 to 0.849) | 38 |
| A | Reference: external model unchanged | 0.777 | | |
| A | Reference: cross-validated local model (about 13,000 patients) | 0.828 | | |
| A | 500 (44) | 0.734 (0.728 to 0.749) | 0.780 (0.779 to 0.785) | 9 |
| A | 1,000 (89) | 0.767 (0.758 to 0.786) | 0.791 (0.778 to 0.801) | 15 |
| A | 2,500 (216) | 0.800 (0.788 to 0.805) | 0.801 (0.800 to 0.804) | 38 |
| A | 5,000 (453) | 0.814 (0.810 to 0.822) | 0.805 (0.802 to 0.811) | 28 |

### C.6 Supplementary figures

**Fig. C.1** (Figure_S1). Availability of each clinical variable: percentage of patients with at least one measurement in system A (open squares) and system B (filled circles).

**Fig. C.2** (Figure_S2). Exploratory analysis: AUROC for separating septic patient-hours in each window relative to onset from all hours of non-septic patients, for the four feature arms; local models solid lines with filled markers, external models dashed lines with open markers.

**Fig. C.3** (Figure_S3). The 15 most influential features of each full model at its development site, as share of the total mean absolute SHAP value, by feature family (hatched: measurement process).

**Fig. C.4** (Figure_S4). AUROC (95% confidence interval) of the local (filled circles) and external (open squares) full models by sex, age group and unit type.
