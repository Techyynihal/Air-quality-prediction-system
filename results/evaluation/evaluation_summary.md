# Final Test Evaluation

- **Model:** efficientnetb0  |  **Test images:** 1815
- **Accuracy:** 0.3256
- **Macro F1:** 0.228  |  **Weighted F1:** 0.25
- **Macro P/R:** 0.1894 / 0.2969
- **Generalization gap (valF1 - testF1):** 0.0181

## Per-class metrics

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| a_Good | 0.2389 | 0.3496 | 0.2838 |
| b_Moderate | 0.3507 | 0.6131 | 0.4462 |
| c_Unhealthy_for_Sensitive_Groups | 0.0193 | 0.0096 | 0.0128 |
| d_Unhealthy | 0.4925 | 0.799 | 0.6094 |
| e_Very_Unhealthy | 0.0349 | 0.0099 | 0.0155 |
| f_Severe | 0.0 | 0.0 | 0.0 |

> Single final evaluation on the untouched test set.
