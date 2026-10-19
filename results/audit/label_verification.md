# Label Verification

- **Label source:** folder
- **Classes (6):** a_Good, b_Moderate, c_Unhealthy_for_Sensitive_Groups, d_Unhealthy, e_Very_Unhealthy, f_Severe

Labels are taken directly from image sub-folder names (each folder = one class). This is unambiguous ground truth; no labels were renamed or merged.

## AQI relationship (observed, not imposed)

### AQI

| Class | min | median | max |
| --- | ---: | ---: | ---: |
| a_Good | 15.0 | 43.0 | 49.0 |
| b_Moderate | 51.0 | 71.0 | 97.0 |
| c_Unhealthy_for_Sensitive_Groups | 101.0 | 124.0 | 145.0 |
| d_Unhealthy | 150.0 | 158.0 | 199.0 |
| e_Very_Unhealthy | 202.0 | 245.0 | 299.0 |
| f_Severe | 320.0 | 363.0 | 450.0 |

### PM2.5

| Class | min | median | max |
| --- | ---: | ---: | ---: |
| a_Good | 4.0 | 21.0 | 37.0 |
| b_Moderate | 26.0 | 30.27 | 68.0 |
| c_Unhealthy_for_Sensitive_Groups | 33.18 | 55.0 | 334.0 |
| d_Unhealthy | 35.0 | 70.08 | 308.0 |
| e_Very_Unhealthy | 158.0 | 279.0 | 401.0 |
| f_Severe | 257.0 | 337.0 | 500.0 |

### PM10

| Class | min | median | max |
| --- | ---: | ---: | ---: |
| a_Good | 7.0 | 37.0 | 59.0 |
| b_Moderate | 44.12 | 59.0 | 78.0 |
| c_Unhealthy_for_Sensitive_Groups | 48.66 | 108.0 | 276.0 |
| d_Unhealthy | 90.8 | 122.0 | 466.0 |
| e_Very_Unhealthy | 132.0 | 199.0 | 420.0 |
| f_Severe | 139.0 | 198.0 | 480.0 |

> Labels were verified against AQI columns for consistency; they were NOT altered to make classification easier.
