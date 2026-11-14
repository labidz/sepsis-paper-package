# Measurement patterns versus physiology in early sepsis prediction: a bidirectional external validation across two hospital systems

**Authors:** [Given name Family name]^a,\*^, [Given name Family name]^b^ — *to be completed by you*

^a^ [Department, institution, full postal address, country]
^b^ [Department, institution, full postal address, country]

\* Corresponding author: [name, email address]

**Article type:** Full-length research article

## Highlights

- Sepsis models lost 0.05 to 0.08 AUROC when moved between two hospital systems
- A model blind to all physiological values matched a values-only model locally
- Measurement patterns gave half the local AUROC gain but transported worse
- One alert threshold missed 44% of sepsis cases or alerted 60% of non-cases
- Recalibration fixed calibration only; refitting needed thousands of local patients


## Abstract

**Background:** Sepsis early-warning models built by machine learning often lose accuracy in other hospitals. We tested whether reliance on the measurement process (which tests are done, and when) explains this.

**Methods:** Using hourly records of 40,336 ICU patients from two US hospital systems (PhysioNet Challenge 2019), we developed gradient-boosted models in each system and validated them in the other, against local cross-validation on the same patients. A factorial of feature sets separated physiological values (measurement status concealed) from process features (no values).

**Results:** Transport reduced full-model AUROC from 0.865 to 0.785 (loss 0.081, 95% CI 0.071 to 0.091) and from 0.826 to 0.775 (0.051, 0.044 to 0.058). A process-only model matched a values-only model locally but lost more on transport: 0.032 more in both directions in the primary fit (95% CIs 0.021 to 0.043, 0.024 to 0.040), median 0.020 and 0.027 across five fits. The excess persisted without post-onset hours or inconsistently recorded variables but was smaller, in one system uncertain, 13 to 24 h before onset. Process features supplied about half of the local gain over covariates but 32% to 43% after transport. A transported alert threshold cut sensitivity from 80% to 56% or raised alerts in non-septic patients from 26% to 60%. Recalibration restored calibration but not discrimination; refitting on 5,000 local patients reached AUROCs of 0.847 and 0.814 (local 0.866, 0.828).

**Conclusions:** Measurement patterns supplied half the local gain but transported worse than physiology. Sepsis models need local validation; reports should separate physiology from process.

**Keywords:** Sepsis; Machine learning; External validation; Informative missingness; Dataset shift; Clinical prediction model; Intensive care


## 1. Introduction

Sepsis contributes to about one in five deaths worldwide, and each hour of delay in treatment is associated with higher mortality [1-3]. Because the syndrome is hard to recognise early [4], many machine-learning models have been built to warn clinicians hours before onset, and retrospective studies report high discrimination [5-7].

That promise has not transferred reliably. A widely implemented proprietary model discriminated far less well at an external health system than its developer had reported, and its discrimination varied between hospitals of one network [8-10]. Algorithms entered in the PhysioNet/Computing in Cardiology Challenge 2019 lost much of their utility in a hidden third hospital system [11], and models moved between intensive care databases lose performance unless trained on several [12,13]. Prospective evaluations show benefit in some settings but not uniformly [14-16]. Why models fail to transport remains debated [17-21].

One explanation deserves a direct test. Electronic health records capture the care process as well as physiology: which tests are ordered and recorded, and when, reflects clinical suspicion, local practice and data capture [22-26]. Flexible learners can exploit such signals as shortcuts [27,28], some architectures are designed to use them [29,30], and risk models may then echo what clinicians already suspect [31]. Sepsis is especially exposed, because its operational definition includes clinician actions (cultures and antibiotics) [4], and the accuracy of a deployed model fell when it was restricted to data recorded before clinicians acted [32]. If measurement patterns carry much of the signal and differ between hospitals, they could explain both high internal accuracy and poor transport. To our knowledge, no study has separated the contribution of physiological values from that of the measurement process and tested the transport of each between hospital systems in both directions.

We therefore measured the loss of performance when an hourly sepsis model developed in one hospital system is applied unchanged in another, in both directions; partitioned predictive performance between physiological values and the measurement process with a factorial design of feature sets, and tested whether each source transports; and examined calibration, clinical utility, alert burden, subgroups and the effects of local recalibration and refitting.


## 2. Methods

### 2.1 Study design

This retrospective study developed and externally validated prediction models and is reported according to TRIPOD+AI [33] (checklist in Appendix B). The analysis plan, with three directional hypotheses, was written before the data were opened (Appendix A); it was not registered and has no independent timestamp. H1: at the target site, the externally developed full model discriminates less well than a locally developed model evaluated on the same patients. H2: a model that sees only the measurement process discriminates better than a covariates-only model. H3: the loss of discrimination on transport is larger for the process-only model than for the values-only model. Each hypothesis was tested in both directions and regarded as supported only if the 95% confidence interval excluded zero in both. The models estimate hourly risk to prompt clinical review in intensive care; they are research instruments and are not proposed for deployment.

### 2.2 Data source and participants

We used the public training sets of the PhysioNet/Computing in Cardiology Challenge 2019 [11,34,35]: de-identified hourly records of intensive care patients from two United States hospital systems with different electronic record systems, Beth Israel Deaconess Medical Center (system A) and Emory University Hospital (system B). Each record holds 40 variables per hour: eight vital signs, 26 laboratory values and six demographic or administrative items. System A held no patients older than 89, and B recorded ages of 90 or more as 100; calendar dates of collection are not released. All patients were included. Each file was verified against its repository checksum.

### 2.3 Outcome

The outcome was the hourly label provided by the Challenge organisers, who defined onset by Sepsis-3 criteria as the earlier of clinical suspicion of infection (blood culture and intravenous antibiotic orders) and a two-point rise in the Sequential Organ Failure Assessment score [4,11]. The label is positive from 6 h before onset, so each hourly prediction estimates the probability that onset lies within the next 6 h or has occurred. Records of septic patients end 3 h after onset. Labelling was automated and did not use our models.

### 2.4 Predictors and feature arms

Predictions were made every hour from the first recorded hour, using only information available up to that hour. Two sources of information were crossed in a 2 × 2 factorial (Fig. 1). All arms included five covariates: age, sex (a binary variable labelled gender in the dataset), unit type, hours from hospital to unit admission and hours since unit admission. Arm C used the covariates alone. Arm P (process) added 139 features that describe only when and how often each of the 34 clinical variables was measured, with no measured value (Appendix C.2). Arm V (values) added 75 features: the last value of each variable carried forward, trailing summaries of six routinely monitored vital signs and five derived indices, including a count of systemic inflammatory response criteria [36]. In arm V the measurement process was concealed: carrying values forward hides their age, and a variable not yet measured for a patient received a random draw from the development-set distribution of observed values, held fixed until a real measurement arrived; predictions averaged five draws. Arm VP combined both feature sets with unmeasured values left missing, as such models are usually built [37,38]. No patient or hour was excluded for missing data.

### 2.5 Model development

All arms used gradient-boosted decision trees (LightGBM [39]) with one configuration fixed in the analysis plan (learning rate 0.05, 31 leaves, at least 200 patient-hours per leaf, column subsampling 0.5, row subsampling 0.8, L2 penalty 10), so that differences between arms reflect information and not tuning. The number of trees was set by early stopping on a 20% patient-level holdout of the development patients (Appendix Table C.8). Classes were not reweighted or resampled, to preserve calibration [40]. Sample size was fixed by the available data; precision is conveyed by confidence intervals.

### 2.6 Validation

Internal performance was estimated by five-fold cross-validation grouped by patient and stratified by sepsis status within each site. For external validation, the model developed on all patients of one site was applied unchanged to every patient of the other, in both directions [41,42]. The transport loss at a target site was the performance of the locally developed model (that site's out-of-fold predictions) minus that of the external model on the same patients, which holds case mix constant [43]. The external model was fitted on more patients (about 16,000) than each cross-validated local model (about 13,000), so differences in training-set size do not favour the local model. A model trained on both sites was cross-validated as a secondary analysis.

### 2.7 Performance measures

The primary measure was the area under the receiver operating characteristic curve (AUROC) across patient-hours; the area under the precision-recall curve (AUPRC) was secondary [44]. Calibration was summarised by the observed-to-expected ratio (O:E), calibration intercept and slope, and the integrated calibration index [40,45]. Clinical utility was assessed by net benefit at threshold probabilities of 0.5% to 10% [46,47] and by the Challenge utility score [11], with its threshold chosen on the development site's out-of-fold predictions for external models and on the other cross-validation folds for local models. For alerting, the threshold was set at the development site to alert 80% of septic patients at or before onset and applied unchanged at the target site (measures in Table 3).

### 2.8 Statistical analysis

Confidence intervals (95%) came from 1,000 bootstrap resamples of patients [48], with the same resamples for every model at a site so that differences were paired. The contribution of each information source was expressed as the gain in AUROC over arm C averaged over both orders of entry (a Shapley decomposition for two sources); because arm C itself lost discrimination on transport, we report absolute contributions as well as shares and treat H3 as the test. Performance was examined by sex, age group and unit type; the dataset's single binary variable labelled gender cannot separate sex from gender, and there is no information on race or ethnicity [49]. Secondary analyses were logistic recalibration [50] of the external model on random samples of 100 to 5,000 target-site patients, and SHAP attributions [51] summarised by feature family. Sensitivity analyses varied the concealment method, the learner, the hyperparameters and the handling of early-onset patients, post-onset hours and a variable never recorded at one site (Appendix C). Post hoc analyses (Appendix A.2) examined time to onset, removed the ten variables whose availability differed more than twofold between systems or the process features of organ-dysfunction variables, refitted all models with four further seeds and refitted the full model on 500 to 5,000 local patients. Analyses used Python 3.13, LightGBM 4.7 and scikit-learn 1.9 [52].

### 2.9 Ethics

The study used only public, de-identified data released under a Creative Commons Attribution 4.0 licence; no further ethical approval or consent was required. Patients and the public were not involved in the design or conduct of the study.


## 3. Results

### 3.1 Cohorts

System A contributed 20,336 patients (790,215 patient-hours) and system B 20,000 patients (761,995 patient-hours) (Table 1). Sepsis occurred in 1,790 (8.8%) and 1,142 (5.7%) patients. Measurement practice differed markedly: system A recorded 40.8 laboratory results per patient-day against 23.6 in B, and several variables were recorded for most patients in one system and few in the other, such as bicarbonate (97% of patients in A, 2% in B), chloride (97% and 8%) and troponin I (2% and 33%); end-tidal carbon dioxide was never recorded in A (Appendix Table C.4, Fig. C.1).

### 3.2 Loss of discrimination on transport

In internal validation the full model (VP) reached an AUROC of 0.865 (95% CI 0.856 to 0.874) in system B and 0.826 (0.818 to 0.834) in system A (Table 2, Fig. 2). Applied unchanged in the other system, the model developed in A reached 0.785 (0.772 to 0.797) in B and the model developed in B reached 0.775 (0.766 to 0.784) in A. Against the local model on the same patients, the external model lost 0.081 (0.071 to 0.091) in B and 0.051 (0.044 to 0.058) in A, which supports H1. A model trained on both systems performed within 0.004 of the local model at each site (Appendix Table C.1).

### 3.3 Physiological values and measurement process

Without a single physiological value, the process-only model discriminated better than covariates alone, by 0.079 (0.069 to 0.092) in B and 0.066 (0.057 to 0.074) in A (H2), and as well as the values-only model (difference -0.004, -0.014 to 0.007, in B; 0.009, -0.0001 to 0.017, in A). On transport the process-only model lost more than the values-only model in both directions (0.099 against 0.067 in B; 0.068 against 0.036 in A), an excess loss of 0.032 (0.021 to 0.043) in B and 0.032 (0.024 to 0.040) in A (H3). The measurement process added 0.056 (0.048 to 0.065) to the local AUROC in B and 0.047 (0.040 to 0.053) in A, 48% and 55% of the gain over covariates; after transport it added 0.042 (0.032 to 0.052) and 0.025 (0.018 to 0.033), or 35% and 34% (Fig. 2c). The contribution of values rose after transport (from 0.059 to 0.077 in B and from 0.038 to 0.048 in A), partly because the covariates-only model also lost discrimination (Table 2).

The process signal depended on the time to onset (Appendix Tables C.7 and C.10, Fig. C.2). In B its share of the local gain rose from 26% at 13 to 24 h before onset to 54% after onset, and the excess transport loss from 0.014 (-0.001 to 0.029) to 0.051 (0.038 to 0.065); in A the share was 57% to 62% in every window and the excess loss 0.015 to 0.040, each interval excluding zero.

Process features carried 35% and 33% of the SHAP attribution of the full models developed in A and B; the most influential feature of the model developed in A was the time since FiO2 was last recorded (Appendix Table C.6, Fig. C.3).

### 3.4 Calibration and clinical utility

Local models were well calibrated (Table 3, Fig. 3). The external models over-predicted risk (O:E 0.91 in B and 0.82 in A) and their predictions were too extreme (calibration slope 0.82, 0.77 to 0.86, and 0.78, 0.75 to 0.81). Their integrated calibration index was close to that of the local model in B (0.37 against 0.32 percentage points) but not in A (0.63 against 0.38). External models retained a higher net benefit than alerting every patient or none at thresholds of 0.5% to 8%, but a lower one than the local models (2.2 against 3.7 per 1,000 patient-hours at a 5% threshold in B, 3.9 against 6.0 in A). The Challenge utility score fell from 0.417 to 0.283 in B and from 0.433 to 0.327 in A.

### 3.5 Alerting

With the threshold fixed at the development site to alert 80% of septic patients by onset, the model developed in A alerted only 56.5% (53.7 to 59.0) of septic patients in B. The model developed in B alerted 85.5% of septic patients in A but also 60.4% (59.7 to 61.0) of non-septic patients, against 25.5% at its development site (Fig. 4). Re-setting the threshold locally to restore 80% sensitivity left the external models alerting 45.3% and 50.8% of non-septic patients, against 25.5% and 32.5% for the local models. The pattern held at sensitivity targets from 50% to 90% (Appendix Table C.9). First alerts preceded onset by a median of 37 to 41 h, because at this sensitivity many alerts fire early in the stay.

### 3.6 Subgroups

Discrimination and transport loss were similar by sex and age group (Appendix Table C.3, Fig. C.4). Transport loss differed more by unit type: in A it was 0.068 in medical and 0.036 in surgical unit patients, and the external model over-predicted markedly in surgical patients (O:E 0.58).

### 3.7 Local updating

Logistic recalibration on 1,000 local patients (a median of 58 septic patients in B and 91 in A) brought the integrated calibration index of the external model to 0.29 and 0.24 percentage points (Appendix Table C.5), at or below that of the local models (0.32 and 0.38). Recalibration cannot change the ranking of patients, and it left net benefit unchanged (at a 5% threshold, 2.1 against 2.2 per 1,000 patient-hours before recalibration in B and 4.0 against 3.9 in A). Local refitting restored discrimination gradually (Appendix Table C.11): on held-out patients, further boosting of the external model on 1,000 local patients raised its AUROC from 0.782 to 0.818 in B and from 0.777 to 0.791 in A, and new models fitted on 5,000 local patients reached 0.847 and 0.814 (cross-validated local models 0.866 and 0.828).

### 3.8 Sensitivity analyses

H1 held in both directions in every sensitivity analysis, and H2 and H3 in every analysis that refitted all four arms (Appendix Table C.2). The transport loss of the full model ranged from 0.064 to 0.088 in B and 0.051 to 0.059 in A when early-onset patients or post-onset hours were excluded or hyperparameters were searched. The excess loss of the process-only model ranged from 0.020 to 0.040 across alternative concealment schemes, removal of end-tidal CO2, both exclusions and logistic regression as the learner. A recurrent network, evaluated on one fold, discriminated less well than the boosted model on the same patients (0.752 against 0.864 in B, 0.799 against 0.824 in A) and lost 0.050 and 0.130 on transport. In post hoc analyses, removing the ten variables whose availability differed more than twofold, with dependent features, left the transport loss (0.082 and 0.055) and the excess loss (0.024, 0.014 to 0.035, and 0.039, 0.030 to 0.048) in place. Across five fits with different seeds, the transport loss ranged from 0.075 to 0.090 in B and 0.051 to 0.060 in A, and the excess loss from 0.015 to 0.032 and 0.020 to 0.032, with every interval excluding zero. Removing process features of organ-dysfunction variables, measured-this-hour flags and laboratory counts gave excess losses of 0.038 (0.026 to 0.049) and 0.036 (0.028 to 0.044). Configurations with near-identical development log-loss differed in external AUROC by up to 0.032 in B and 0.011 in A, an instance of underspecification [53].


## 4. Discussion

### 4.1 Principal findings

Sepsis early-warning models lost 0.05 to 0.08 of AUROC when moved between two hospital systems, in both directions. A model that saw no physiological value, only when and how often variables were measured, matched a values-only model at the site where it was built. Measurement-process features accounted for about half of the local gain in AUROC over basic covariates but a third to two-fifths of the gain that survived transport, and process-only models lost more than values-only models in both directions and in every refit.

### 4.2 Interpretation

A large part of what such models learn reflects how and when data were recorded: test ordering, monitoring, data capture and outcome ascertainment, all of which differed between the systems. Some, such as bicarbonate and chloride recorded for nearly all patients in A and few in B, are implausible as clinical practice and point to differences in data extraction or variable mapping. Removing the ten variables whose availability differed most left the transport loss and an excess loss of process features in place (Section 3.8).

In B the process signal was concentrated near and after onset, where the label itself is defined by culture and antibiotic orders and by the tests that reveal organ dysfunction [4,11], so part of it echoes clinical recognition [31,32]. In A it was present a day before onset, consistent with testing intensity marking severity of illness. Thirteen to 24 h before onset, the excess transport loss of process features was about half as large as overall.

The measurement process is not the whole explanation. Values-only and even covariates-only models also lost discrimination, which points to differences in case mix, recording and outcome ascertainment [18,19,43]; covariates such as hours since unit admission also carry recording differences. Process features retained some value after transport, so removing them is not a remedy. A model trained on both systems matched the local models, in line with multi-site studies [12,13], but it could not be tested in a third system and may partly recognise which system a record came from.

### 4.3 Comparison with previous work

The size of the loss agrees with external validations of proprietary and academic sepsis models [8,10-12]. Earlier studies showed that which tests are run, and when, can predict outcomes as well as or better than their results [22,30] and that models built on clinician-initiated data may add little to clinical judgement [31]. Missing-data indicators can improve prediction [24,54], but missingness depends on record-keeping policies that can differ between locations even when patients do not, a dataset shift termed missingness shift [55]. Our study adds empirical evidence from two hospital systems: it partitions the signal between the two sources and shows, with paired comparisons in both directions, that the process component is the less portable one.

### 4.4 Implications

First, thresholds do not transport: the same threshold yielded 56% sensitivity in one system and alerts in 60% of non-septic patients in the other, which invites missed sepsis or alert fatigue [56-58]. Second, recalibration restored calibration but by design cannot restore discrimination and did not improve net benefit; refitting on local data recovered discrimination only with thousands of patients. Hospitals should validate, update and monitor a model locally before and during use [20,42,59]. Third, developers could report a process-only baseline, so that users can judge how much of a model's accuracy reflects local recording practice. Such reporting would complement TRIPOD+AI and PROBAST+AI [33,60]. Fourth, discrimination measured over all hours overstates clinical value when part of the signal comes from actions that clinicians have already taken [32]; reporting performance by time to onset makes this visible.

### 4.5 Strengths and limitations

Strengths include an analysis plan written before the data were opened, bidirectional validation, paired comparisons on identical patients, patient-level resampling, repeated refits and open code. The study also has limitations. It covers two academic United States systems and data collected before 2019, and it is retrospective. The outcome label depends on clinician actions whose timestamps are not released, so we could not restrict evaluation to hours before clinical recognition. Concealing the measurement process in the values-only arm is imperfect. Random draws discard the legitimate knowledge that a value is real, and trailing statistics may leak gaps in monitoring. Draws from the development distribution also carry that site's measurement pattern into the values-only model at the target site (bicarbonate values for 98% of patients in B came from A's distribution), so part of the values-only model's loss is itself process shift, which makes H3 conservative. The covariates-only baseline is not neutral, so process shares are descriptive. Confidence intervals do not include refitting variability, which we assessed separately. The dataset lacks race and ethnicity and, for a third to a half of patients, unit type. Other learners were limited to logistic regression and a briefly tuned recurrent network. The alert analysis considered only the first alert, and we did not study the effect of alerts on care or outcomes. The next steps are validation in further systems and time periods, and evaluation restricted to the hours before clinical recognition.


## 5. Conclusions

In two hospital systems, measurement-process features accounted for about half of the local gain in AUROC over basic covariates in sepsis early-warning models and transported less well than physiological values. Models moved between systems lost discrimination, calibration and a usable alert threshold; recalibration repaired calibration only, and refitting needed thousands of local patients to approach a locally developed model. Sepsis models should be validated locally before use, and their reports should separate what is learned from the patient from what is learned from the recording and care process.


## Tables and figures (inline in the journal file)

Table 1. Characteristics of the two cohorts.
| Characteristic | System A | System B |
| --- | --- | --- |
| Patients, n | 20,336 | 20,000 |
| Patient-hours, n | 790,215 | 761,995 |
| Recorded hours per patient, median (IQR) | 39 (25 to 47) | 38 (23 to 47) |
| Age, years, mean (SD) | 62.6 (16.2) | 60.6 (16.7) |
| Male sex, % | 58.2 | 53.7 |
| Unit type: medical / surgical / not recorded, % | 26.3 / 26.9 / 46.8 | 34.6 / 34.9 / 30.5 |
| Hours from hospital to unit admission, median (IQR) | 2.8 (0.0 to 34.1) | 8.6 (3.4 to 52.4) |
| Patients with sepsis, n (%) | 1,790 (8.8) | 1,142 (5.7) |
| Positive patient-hours, n (%) | 17,136 (2.17) | 10,780 (1.41) |
| Sepsis label positive at first recorded hour, n | 203 | 223 |
| Onset, hours after unit admission, median (IQR) | 43 (19 to 87) | 48 (22 to 94) |
| Vital signs recorded per hour, mean (of 8) | 5.31 | 5.51 |
| Hours with at least one laboratory result, % | 30.7 | 25.0 |
| Laboratory results per patient-day, mean | 40.8 | 23.6 |

Table 2. Discrimination of the four feature arms in local and external validation. Values are estimates (95% confidence intervals). Transport loss is the local minus the external AUROC on the same patients; differences were computed before rounding.
| Target site and feature arm | Local model: AUROC | External model: AUROC | Transport loss in AUROC | Local model: AUPRC | External model: AUPRC |
| --- | --- | --- | --- | --- | --- |
| **Evaluated in system B** (external model developed in A) | | | | | |
| C: covariates only | 0.750 (0.733 to 0.765) | 0.665 (0.648 to 0.683) | 0.085 (0.066 to 0.102) | 0.057 (0.050 to 0.064) | 0.045 (0.040 to 0.051) |
| V: values (process concealed) | 0.833 (0.822 to 0.843) | 0.765 (0.751 to 0.778) | 0.067 (0.054 to 0.080) | 0.095 (0.082 to 0.103) | 0.068 (0.059 to 0.077) |
| P: process (no values) | 0.829 (0.818 to 0.840) | 0.730 (0.715 to 0.743) | 0.099 (0.087 to 0.112) | 0.101 (0.089 to 0.112) | 0.052 (0.047 to 0.058) |
| VP: values + process | 0.865 (0.856 to 0.874) | 0.785 (0.772 to 0.797) | 0.081 (0.071 to 0.091) | 0.123 (0.109 to 0.133) | 0.067 (0.060 to 0.075) |
| **Evaluated in system A** (external model developed in B) | | | | | |
| C: covariates only | 0.741 (0.730 to 0.753) | 0.702 (0.690 to 0.714) | 0.040 (0.030 to 0.049) | 0.085 (0.079 to 0.091) | 0.079 (0.072 to 0.085) |
| V: values (process concealed) | 0.798 (0.789 to 0.807) | 0.762 (0.752 to 0.772) | 0.036 (0.029 to 0.043) | 0.106 (0.098 to 0.114) | 0.085 (0.078 to 0.092) |
| P: process (no values) | 0.807 (0.798 to 0.815) | 0.739 (0.728 to 0.749) | 0.068 (0.060 to 0.077) | 0.104 (0.096 to 0.112) | 0.079 (0.073 to 0.086) |
| VP: values + process | 0.826 (0.818 to 0.834) | 0.775 (0.766 to 0.784) | 0.051 (0.044 to 0.058) | 0.115 (0.106 to 0.124) | 0.088 (0.082 to 0.095) |

Table 3. Calibration, clinical utility and alerting of the full model (VP). Alert measures for external models use the threshold fixed at the development site. Net benefit is per 1,000 patient-hours. The utility threshold was chosen on the development site's out-of-fold predictions for external models and on the other cross-validation folds for local models.
| Measure | System B: local model | System B: external model (from A) | System A: local model | System A: external model (from B) |
| --- | --- | --- | --- | --- |
| Observed-to-expected ratio | 0.97 (0.91 to 1.04) | 0.91 (0.85 to 0.97) | 0.96 (0.91 to 1.00) | 0.82 (0.78 to 0.86) |
| Calibration intercept | -0.03 (-0.10 to 0.04) | -0.11 (-0.18 to -0.04) | -0.05 (-0.10 to 0.00) | -0.22 (-0.27 to -0.17) |
| Calibration slope | 0.94 (0.90 to 0.98) | 0.82 (0.77 to 0.86) | 0.93 (0.90 to 0.97) | 0.78 (0.75 to 0.81) |
| Integrated calibration index, percentage points | 0.32 (0.26 to 0.40) | 0.37 (0.29 to 0.47) | 0.38 (0.30 to 0.48) | 0.63 (0.55 to 0.73) |
| Brier score × 100 | 1.32 (1.25 to 1.40) | 1.38 (1.31 to 1.45) | 2.02 (1.94 to 2.10) | 2.10 (2.01 to 2.18) |
| Net benefit at a 2% threshold, per 1,000 patient-hours | 7.1 (6.4 to 7.7) | 4.9 (4.4 to 5.4) | 11.2 (10.5 to 11.9) | 9.4 (8.7 to 10.1) |
| Net benefit at a 5% threshold, per 1,000 patient-hours | 3.7 (3.3 to 4.2) | 2.2 (1.8 to 2.6) | 6.0 (5.5 to 6.5) | 3.9 (3.4 to 4.4) |
| Challenge utility score | 0.417 (0.392 to 0.442) | 0.283 (0.256 to 0.308) | 0.433 (0.415 to 0.451) | 0.327 (0.305 to 0.348) |
| Septic patients alerted at or before onset, % | 80.0 (77.7 to 82.2) | 56.5 (53.7 to 59.0) | 80.1 (78.3 to 81.9) | 85.5 (83.9 to 87.1) |
| Non-septic patients alerted, % | 25.5 (25.0 to 26.2) | 17.1 (16.5 to 17.6) | 32.5 (31.8 to 33.2) | 60.4 (59.7 to 61.0) |
| Positive predictive value of an alert, % | 15.9 (15.0 to 16.9) | 16.7 (15.5 to 17.8) | 19.2 (18.3 to 20.1) | 12.0 (11.5 to 12.6) |
| Non-septic patients alerted when the threshold is re-set to 80% sensitivity at the target site, % | same as above | 45.3 (44.7 to 46.1) | same as above | 50.8 (50.1 to 51.5) |

Fig. 1 (file Figure_1.png). Study design. (a) Feature arms: 2 x 2 factorial crossing physiological values with the measurement process. (b) Validation: at each target site the external model (developed on all patients of the other system) and the local model (five-fold cross-validation) are evaluated on the same patients.
Fig. 2 (Figure_2.png). Discrimination by feature arm. (a, b) AUROC with 95% CI of local (filled circles) and external (open squares) models; numbers give the transport loss (local minus external AUROC). (c) Gain in AUROC over each scenario's own covariates-only arm, attributed to values and to process (Shapley), with process share and 95% CI.
Fig. 3 (Figure_3.png). Calibration (log axes) and decision curves of the full model: local, external and two-site models. Calibration curves are logistic fits with a restricted cubic spline (five knots) of the logit of the prediction; most predicted risks are below 2%. Decision curves are per patient-hour.
Fig. 4 (Figure_4.png). Alerting at transported thresholds with an 80% sensitivity target at the development site (a, b), and integrated calibration index after recalibration on random samples of local patients, evaluated in the patients not sampled (c, d); bands show the 2.5th to 97.5th percentiles over 50 repetitions.
Appendix figures: Figure_S1.png (variable availability by site), Figure_S2.png (AUROC by time to onset, exploratory), Figure_S3.png (SHAP), Figure_S4.png (subgroups).


## CRediT authorship contribution statement

*To be completed by you. List each author with their roles, chosen from: Conceptualization, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing - original draft, Writing - review and editing.*

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. *(Confirm, or replace with your disclosures.)*

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. *(Confirm, or replace with your funding sources.)*

## Ethics statement

This study analysed only public, de-identified data released by the PhysioNet/Computing in Cardiology Challenge 2019 under a Creative Commons Attribution 4.0 licence. No patient was contacted and no identifiable information was available to the authors, so no further ethical approval or informed consent was required. *(Confirm that this matches your institution's policy on secondary analysis of public data.)*

## Data availability

The data are publicly available from PhysioNet (https://doi.org/10.13026/v64v-d857). The analysis code, the pre-specified analysis plan, the checksums of the source archives and all result files are provided as supplementary material. *(Add the link to a public code repository if you create one.)*

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) to write and run the analysis code, to produce the tables and figures, and to draft the manuscript, under the authors' direction. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. *(This statement must stay accurate: check the analysis and the text yourself before submitting, and edit the statement if your use differs.)*

## Acknowledgements

We thank the organisers of the PhysioNet/Computing in Cardiology Challenge 2019 and the contributing hospital systems for making the data openly available.

## Appendices. Supplementary material

Appendix A (analysis plan and deviations), Appendix B (TRIPOD+AI checklist) and Appendix C (supplementary methods, tables and figures) are provided in the Supplementary material tab of this document.


## References

[1] Rudd KE, Johnson SC, Agesa KM, Shackelford KA, Tsoi D, Kievlan DR, et al. Global, regional, and national sepsis incidence and mortality, 1990-2017: analysis for the Global Burden of Disease Study. Lancet 2020;395:200-11. https://doi.org/10.1016/S0140-6736(19)32989-7

[2] Seymour CW, Gesten F, Prescott HC, Friedrich ME, Iwashyna TJ, Phillips GS, et al. Time to treatment and mortality during mandated emergency care for sepsis. N Engl J Med 2017;376:2235-44. https://doi.org/10.1056/NEJMoa1703058

[3] Evans L, Rhodes A, Alhazzani W, Antonelli M, Coopersmith CM, French C, et al. Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock 2021. Intensive Care Med 2021;47:1181-247. https://doi.org/10.1007/s00134-021-06506-y

[4] Singer M, Deutschman CS, Seymour CW, Shankar-Hari M, Annane D, Bauer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315:801-10. https://doi.org/10.1001/jama.2016.0287

[5] Fleuren LM, Klausch TLT, Zwager CL, Schoonmade LJ, Guo T, Roggeveen LF, et al. Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. Intensive Care Med 2020;46:383-400. https://doi.org/10.1007/s00134-019-05872-y

[6] Nemati S, Holder A, Razmi F, Stanley MD, Clifford GD, Buchman TG. An interpretable machine learning model for accurate prediction of sepsis in the ICU. Crit Care Med 2018;46:547-53. https://doi.org/10.1097/CCM.0000000000002936

[7] Henry KE, Hager DN, Pronovost PJ, Saria S. A targeted real-time early warning score (TREWScore) for septic shock. Sci Transl Med 2015;7:299ra122. https://doi.org/10.1126/scitranslmed.aab3719

[8] Wong A, Otles E, Donnelly JP, Krumm A, McCullough J, DeTroyer-Cooley O, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Intern Med 2021;181:1065-70. https://doi.org/10.1001/jamainternmed.2021.2626

[9] Habib AR, Lin AL, Grant RW. The Epic Sepsis Model falls short—the importance of external validation. JAMA Intern Med 2021;181:1040-1. https://doi.org/10.1001/jamainternmed.2021.3333

[10] Lyons PG, Hofford MR, Yu SC, Michelson AP, Payne PRO, Hough CL, et al. Factors associated with variability in the performance of a proprietary sepsis prediction model across 9 networked hospitals in the US. JAMA Intern Med 2023;183:611-2. https://doi.org/10.1001/jamainternmed.2022.7182

[11] Reyna MA, Josef CS, Jeter R, Shashikumar SP, Westover MB, Nemati S, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Crit Care Med 2020;48:210-7. https://doi.org/10.1097/CCM.0000000000004145

[12] Moor M, Bennett N, Plecko D, Horn M, Rieck B, Meinshausen N, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. EClinicalMedicine 2023;62:102124. https://doi.org/10.1016/j.eclinm.2023.102124

[13] Rockenschaub P, Hilbert A, Kossen T, Elbers P, von Dincklage F, Madai VI, et al. The impact of multi-institution datasets on the generalizability of machine learning prediction models in the ICU. Crit Care Med 2024;52:1710-21. https://doi.org/10.1097/CCM.0000000000006359

[14] Adams R, Henry KE, Sridharan A, Soleimani H, Zhan A, Rawat N, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. Nat Med 2022;28:1455-60. https://doi.org/10.1038/s41591-022-01894-0

[15] Shimabukuro DW, Barton CW, Feldman MD, Mataraso SJ, Das R. Effect of a machine learning-based severe sepsis prediction algorithm on patient survival and hospital length of stay: a randomised clinical trial. BMJ Open Respir Res 2017;4:e000234. https://doi.org/10.1136/bmjresp-2017-000234

[16] Boussina A, Shashikumar SP, Malhotra A, Owens RL, El-Kareh R, Longhurst CA, et al. Impact of a deep learning sepsis prediction model on quality of care and survival. NPJ Digit Med 2024;7:14. https://doi.org/10.1038/s41746-023-00986-6

[17] Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of generalisability in clinical research and machine learning in health care. Lancet Digit Health 2020;2:e489-92. https://doi.org/10.1016/S2589-7500(20)30186-2

[18] Lasko TA, Strobl EV, Stead WW. Why do probabilistic clinical models fail to transport between sites. NPJ Digit Med 2024;7:53. https://doi.org/10.1038/s41746-024-01037-4

[19] Finlayson SG, Subbaswamy A, Singh K, Bowers J, Kupke A, Zittrain J, et al. The clinician and dataset shift in artificial intelligence. N Engl J Med 2021;385:283-6. https://doi.org/10.1056/NEJMc2104626

[20] Youssef A, Pencina M, Thakur A, Zhu T, Clifton D, Shah NH. External validation of AI models in health should be replaced with recurring local validation. Nat Med 2023;29:2686-7. https://doi.org/10.1038/s41591-023-02540-z

[21] Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med 2018;15:e1002683. https://doi.org/10.1371/journal.pmed.1002683

[22] Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. BMJ 2018;361:k1479. https://doi.org/10.1136/bmj.k1479

[23] Pivovarov R, Albers DJ, Sepulveda JL, Elhadad N. Identifying and mitigating biases in EHR laboratory tests. J Biomed Inform 2014;51:24-34. https://doi.org/10.1016/j.jbi.2014.03.016

[24] Sisk R, Lin L, Sperrin M, Barrett JK, Tom B, Diaz-Ordaz K, et al. Informative presence and observation in routine health data: a review of methodology for clinical risk prediction. J Am Med Inform Assoc 2021;28:155-66. https://doi.org/10.1093/jamia/ocaa242

[25] Groenwold RHH. Informative missingness in electronic health record systems: the curse of knowing. Diagn Progn Res 2020;4:8. https://doi.org/10.1186/s41512-020-00077-0

[26] Sharafoddini A, Dubin JA, Maslove DM, Lee J. A new insight into missing data in intensive care unit patient profiles: observational study. JMIR Med Inform 2019;7:e11605. https://doi.org/10.2196/11605

[27] Geirhos R, Jacobsen JH, Michaelis C, Zemel R, Brendel W, Bethge M, et al. Shortcut learning in deep neural networks. Nat Mach Intell 2020;2:665-73. https://doi.org/10.1038/s42256-020-00257-z

[28] DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat Mach Intell 2021;3:610-9. https://doi.org/10.1038/s42256-021-00338-7

[29] Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. Sci Rep 2018;8:6085. https://doi.org/10.1038/s41598-018-24271-9

[30] Lipton ZC, Kale D, Wetzel R. Directly modeling missing data in sequences with RNNs: improved classification of clinical time series. Proc Mach Learn Res 2016;56:253-70.

[31] Beaulieu-Jones BK, Yuan W, Brat GA, Beam AL, Weber G, Ruffin M, et al. Machine learning for patient risk stratification: standing on, or looking over, the shoulders of clinicians? NPJ Digit Med 2021;4:62. https://doi.org/10.1038/s41746-021-00426-3

[32] Kamran F, Tjandra D, Heiler A, Virzi J, Singh K, King JE, et al. Evaluation of sepsis prediction models before onset of treatment. NEJM AI 2024;1:AIoa2300032. https://doi.org/10.1056/AIoa2300032

[33] Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378

[34] [dataset] Reyna M, Josef C, Jeter R, Shashikumar S, Moody B, Westover MB, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). PhysioNet; 2019. https://doi.org/10.13026/v64v-d857

[35] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PC, Mark RG, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 2000;101:e215-20. https://doi.org/10.1161/01.CIR.101.23.e215

[36] Bone RC, Balk RA, Cerra FB, Dellinger RP, Fein AM, Knaus WA, et al. Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis. Chest 1992;101:1644-55. https://doi.org/10.1378/chest.101.6.1644

[37] Yang M, Liu C, Wang X, Li Y, Gao H, Liu X, et al. An explainable artificial intelligence predictor for early detection of sepsis. Crit Care Med 2020;48:e1091-6. https://doi.org/10.1097/CCM.0000000000004550

[38] Morrill JH, Kormilitzin A, Nevado-Holgado AJ, Swaminathan S, Howison SD, Lyons TJ. Utilization of the signature method to identify the early onset of sepsis from multivariate physiological time series in critical care monitoring. Crit Care Med 2020;48:e976-81. https://doi.org/10.1097/CCM.0000000000004510

[39] Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, et al. LightGBM: a highly efficient gradient boosting decision tree. Adv Neural Inf Process Syst 2017;30:3146-54.

[40] Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7

[41] Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal, internal-external, and external validation. J Clin Epidemiol 2016;69:245-7. https://doi.org/10.1016/j.jclinepi.2015.04.005

[42] Riley RD, Archer L, Snell KIE, Ensor J, Dhiman P, Martin GP, et al. Evaluation of clinical prediction models (part 2): how to undertake an external validation study. BMJ 2024;384:e074820. https://doi.org/10.1136/bmj-2023-074820

[43] Debray TPA, Vergouwe Y, Koffijberg H, Nieboer D, Steyerberg EW, Moons KGM. A new framework to enhance the interpretation of external validation studies of clinical prediction models. J Clin Epidemiol 2015;68:279-89. https://doi.org/10.1016/j.jclinepi.2014.06.018

[44] Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One 2015;10:e0118432. https://doi.org/10.1371/journal.pone.0118432

[45] Austin PC, Steyerberg EW. The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. Stat Med 2019;38:4051-65. https://doi.org/10.1002/sim.8281

[46] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 2006;26:565-74. https://doi.org/10.1177/0272989X06295361

[47] Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision curve analysis. Diagn Progn Res 2019;3:18. https://doi.org/10.1186/s41512-019-0064-7

[48] Efron B, Tibshirani R. Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy. Stat Sci 1986;1:54-75. https://doi.org/10.1214/ss/1177013815

[49] Heidari S, Babor TF, De Castro P, Tort S, Curno M. Sex and Gender Equity in Research: rationale for the SAGER guidelines and recommended use. Res Integr Peer Rev 2016;1:2. https://doi.org/10.1186/s41073-016-0007-6

[50] Steyerberg EW, Borsboom GJJM, van Houwelingen HC, Eijkemans MJC, Habbema JDF. Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. Stat Med 2004;23:2567-86. https://doi.org/10.1002/sim.1844

[51] Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al. From local explanations to global understanding with explainable AI for trees. Nat Mach Intell 2020;2:56-67. https://doi.org/10.1038/s42256-019-0138-9

[52] Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res 2011;12:2825-30.

[53] D'Amour A, Heller K, Moldovan D, Adlam B, Alipanahi B, Beutel A, et al. Underspecification presents challenges for credibility in modern machine learning. J Mach Learn Res 2022;23:1-61.

[54] Sisk R, Sperrin M, Peek N, van Smeden M, Martin GP. Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: a simulation study. Stat Methods Med Res 2023;32:1461-77. https://doi.org/10.1177/09622802231165001

[55] Zhou H, Balakrishnan S, Lipton Z. Domain adaptation under missingness shift. Proc Mach Learn Res 2023;206:9577-606.

[56] Ginestra JC, Giannini HM, Schweickert WD, Meadows L, Lynch MJ, Pavan K, et al. Clinician perception of a machine learning-based early warning system designed to predict severe sepsis and septic shock. Crit Care Med 2019;47:1477-84. https://doi.org/10.1097/CCM.0000000000003803

[57] Sendak MP, Ratliff W, Sarro D, Alderton E, Futoma J, Gao M, et al. Real-world integration of a sepsis deep learning technology into routine clinical care: implementation study. JMIR Med Inform 2020;8:e15182. https://doi.org/10.2196/15182

[58] van der Vegt AH, Scott IA, Dermawan K, Schnetler RJ, Kalke VR, Lane PJ. Deployment of machine learning algorithms to predict sepsis: systematic review and application of the SALIENT clinical AI implementation framework. J Am Med Inform Assoc 2023;30:1349-61. https://doi.org/10.1093/jamia/ocad075

[59] Davis SE, Greevy RA Jr, Fonnesbeck C, Lasko TA, Walsh CG, Matheny ME. A nonparametric updating method to correct clinical prediction model drift. J Am Med Inform Assoc 2019;26:1448-57. https://doi.org/10.1093/jamia/ocz127

[60] Moons KGM, Damen JAA, Kaul T, Hooft L, Andaur Navarro C, Dhiman P, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
