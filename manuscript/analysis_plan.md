# Analysis plan (version 1.0)

Fixed on 11 November 2026, before the study data were opened. The plan was not registered externally. Any later deviation is listed at the end of this plan and reported in the supplement.

## Aims

1. Measure the loss of performance when an hourly sepsis-prediction model developed in one hospital system is applied unchanged in the other, in both directions.
2. Separate how much of the predictive performance comes from physiological values and how much from the measurement process (which variables were measured, when and how often), and test whether each source transports.
3. Describe calibration, clinical utility and alert burden after transport, and test whether recalibration on limited local data restores them.
4. Describe performance by sex, age group and intensive care unit type.

## Hypotheses

- **H1.** At the target site, the externally developed full model discriminates less well than a locally developed model evaluated on the same patients.
- **H2.** A model that sees only the measurement process, and no physiological value, discriminates better than a covariates-only model in internal validation.
- **H3.** The transport loss (local minus external AUROC) is larger for the process-only model than for the values-only model.

Each hypothesis is tested in both directions (A to B, B to A) and is considered supported only if the 95% confidence interval excludes zero in both. All other analyses are descriptive.

## Data and outcome

- Source: public training sets A and B of the PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. All patients are included.
- Unit of prediction: the patient-hour, using only information recorded up to that hour.
- Outcome: the distributed SepsisLabel (positive from 6 hours before Sepsis-3 onset). Onset time is the first positive hour plus 6 hours. Patients positive at their first recorded hour are kept in the main analysis and excluded from lead-time estimates.

## Feature arms

All arms include the covariates: age, sex, unit type, hours from hospital to unit admission, hours since unit admission.

| Arm | Values | Process | Content |
| --- | --- | --- | --- |
| C | No | No | Covariates only (reference) |
| V | Yes | No | Last value carried forward for 8 vital signs and 26 laboratory tests; trailing 6-hour and 24-hour mean, minimum and maximum for six routinely monitored vital signs; derived indices (shock index, pulse pressure, urea-to-creatinine ratio, SpO2-to-FiO2 ratio, SIRS count). Measurement status is concealed: a variable never yet measured receives a random draw from the development-site distribution of observed values, fixed per patient until a real value arrives; predictions average five draws |
| P | No | Yes | For each of the 34 variables: measured this hour, count so far, count in the trailing 24 hours, hours since last measurement; plus counts of distinct laboratory tests in the current hour, trailing 6 hours and trailing 24 hours |
| V+P | Yes | Yes | Union of V and P, with never-measured values left missing |

## Model

Gradient-boosted decision trees (LightGBM, logistic loss) with one configuration fixed here for every arm and site: learning rate 0.05, 31 leaves, at least 200 patient-hours per leaf, column subsampling 0.5, row subsampling 0.8, L2 penalty 10, up to 3,000 trees with early stopping (50 rounds, log-loss) on a 20% patient-level holdout of the training patients. No resampling or class weighting.

## Validation

- Internal: 5-fold cross-validation grouped by patient and stratified by sepsis status, separately in each site; out-of-fold predictions are pooled.
- External: the model developed on all patients of one site is applied unchanged to all patients of the other site.
- Reference at each target site: the locally developed model (that site's out-of-fold predictions).
- Secondary: a pooled two-site model, cross-validated.

## Performance measures

All intervals come from 1,000 patient-level bootstrap resamples, paired for differences.

- Discrimination: AUROC (primary) and area under the precision-recall curve.
- Calibration: observed-to-expected ratio, calibration intercept and slope, integrated calibration index, Brier score, calibration curves.
- Clinical utility: net benefit across threshold probabilities of 0.5% to 10%; the Challenge utility score.
- Alerting: the threshold is fixed in the development site so that 80% of septic patients are alerted at or before onset. Reported at each site: sensitivity, proportion of non-septic patients alerted, positive predictive value, median lead time.
- Subgroups: sex, age (under 50, 50-64, 65-79, 80 or older), unit type.

## Secondary and sensitivity analyses

1. Local updating: logistic recalibration of the external model on random samples of 100 to 5,000 target-site patients, evaluated in the remainder (50 repetitions).
2. Explanations: SHAP values for the full models, summarised by feature family and compared between sites.
3. Alternative concealment in arm V (median imputation; current values only).
4. Logistic regression as the learner for all four arms.
5. Excluding patients positive at their first recorded hour; excluding hours after onset.
6. Excluding variables recorded in under 1% of patients at either site.
7. Hyperparameter search for the full model, to show sensitivity to the fixed configuration.
8. A recurrent neural network on the raw hourly sequences as a deep-learning comparator.

## Deviations from this plan

See Appendix A of the supplementary material.
