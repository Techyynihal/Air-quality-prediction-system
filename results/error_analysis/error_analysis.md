# Error Analysis

- Test images: 1815
- Errors: 1224 (67.44%)

## Most confused class pairs

| Actual | Predicted | Count |
| --- | --- | ---: |
| c_Unhealthy_for_Sensitive_Groups | d_Unhealthy | 188 |
| a_Good | b_Moderate | 149 |
| c_Unhealthy_for_Sensitive_Groups | b_Moderate | 148 |
| e_Very_Unhealthy | c_Unhealthy_for_Sensitive_Groups | 143 |
| e_Very_Unhealthy | d_Unhealthy | 134 |
| f_Severe | a_Good | 114 |
| b_Moderate | a_Good | 91 |
| d_Unhealthy | c_Unhealthy_for_Sensitive_Groups | 49 |
| c_Unhealthy_for_Sensitive_Groups | a_Good | 44 |
| f_Severe | e_Very_Unhealthy | 38 |

## Per-class error rate

| Class | Support | Errors | Rate |
| --- | ---: | ---: | ---: |
| a_Good | 246 | 160 | 0.6504 |
| b_Moderate | 274 | 106 | 0.3869 |
| c_Unhealthy_for_Sensitive_Groups | 418 | 414 | 0.9904 |
| d_Unhealthy | 413 | 83 | 0.201 |
| e_Very_Unhealthy | 302 | 299 | 0.9901 |
| f_Severe | 162 | 162 | 1.0 |

> Confusions may reflect possible visual factors (haze, lighting, similar scenes); not asserted as proven causes.
