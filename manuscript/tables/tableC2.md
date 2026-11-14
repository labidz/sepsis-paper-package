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