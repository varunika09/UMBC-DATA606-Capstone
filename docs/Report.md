# Early Sepsis Risk Prediction Using ICU Clinical Data

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

**Author:** Varunika Bussa

**GitHub Repository:** [https://github.com/varunika09/UMBC-DATA606-Capstone.git](https://github.com/varunika09/UMBC-DATA606-Capstone.git)

**LinkedIn Profile:** [www.linkedin.com/in/varunika-bussa-99a837233](https://www.linkedin.com/in/varunika-bussa-99a837233)

**PowerPoint Presentation:** *(link to be added)*

**YouTube Video:** *(link to be added)*

---

# 1. Background

## What is this project about?

This project focuses on the early prediction of sepsis in ICU patients using machine learning. Sepsis is a life-threatening condition caused by the body's extreme response to infection. If not detected early, it can lead to organ failure and death, early intervention significantly improves survival rates.

The goal is to build a predictive model that estimates whether a patient is at risk of developing sepsis **within the next 6 hours** using routinely collected ICU vital signs and laboratory measurements, and to deploy it as an interpretable early-warning application.

## Why does it matter?

- Sepsis affects approximately **1.7 million Americans annually** and causes **270,000 deaths per year** (CDC)
- It is a leading cause of ICU mortality worldwide
- Existing clinical scoring systems (SOFA, qSOFA, SIRS) are **reactive** - they confirm sepsis after it develops, not before
- Each hour of delayed treatment increases mortality by **4–8%**
- Hospitals collect continuous patient monitoring data, but automated early-warning systems are not always implemented
- A machine learning-based system can detect subtle deterioration patterns **hours before** clinical diagnosis, enabling earlier intervention

## Research Questions

1. Can ICU vital signs and lab values predict sepsis onset **6 hours** before it is clinically diagnosed?
2. Which physiological measurements are most predictive of sepsis?
3. Can we build an interpretable model suitable for clinical decision support?
4. How well does the model generalize to unseen patients?

---

# 2. Data

## Data Source

This project uses the **PhysioNet 2019 Sepsis Challenge Dataset**:
[https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

The dataset contains ICU time-series clinical measurements collected from multiple hospital systems as part of the 2019 PhysioNet Computing in Cardiology Challenge.

## Data Size and Shape

| Property | Value |
|---|---|
| File size | 146.7 MB |
| Total rows | 1,552,210 |
| Total columns | 43 |
| Unique patients | 40,336 |
| Sepsis-positive rows | 27,916 (1.80%) |
| Sepsis-positive patients | 2,932 (7.27%) |

## What does each row represent?

Each row represents **one ICU hour for one patient**. A patient with a 50-hour ICU stay contributes 50 rows. Patient IDs are non-sequential integers ranging from 1 to 120,000 — only 40,336 of a larger hospital database met the study inclusion criteria.

## Data Dictionary

### Vital Signs - recorded approximately every hour

| Column | Type | Definition |
|---|---|---|
| HR | float | Heart rate (bpm) |
| O2Sat | float | Oxygen saturation (%) |
| Temp | float | Body temperature (degrees C) |
| SBP | float | Systolic blood pressure (mmHg) |
| MAP | float | Mean arterial pressure (mmHg) |
| DBP | float | Diastolic blood pressure (mmHg) |
| Resp | float | Respiratory rate (breaths/min) |
| EtCO2 | float | End-tidal CO2 — dropped (96.3% missing) |

### Laboratory Values — ordered as needed, not every hour

| Column | Type | Definition |
|---|---|---|
| Lactate | float | Blood lactate (mmol/L) — key sepsis marker |
| WBC | float | White blood cell count — immune response |
| Creatinine | float | Kidney function marker |
| pH | float | Blood acidity — metabolic status |
| Glucose | float | Blood sugar (mg/dL) |
| BUN | float | Blood urea nitrogen — kidney function |
| Hgb | float | Hemoglobin — oxygen-carrying capacity |
| Hct | float | Hematocrit |
| Platelets | float | Platelet count — clotting function |
| PTT | float | Partial thromboplastin time — clotting |
| TroponinI | float | Cardiac injury marker |
| Bilirubin_total | float | Liver function |
| Bilirubin_direct | float | Liver function — dropped (99.8% missing) |
| Fibrinogen | float | Clotting factor — dropped (99.3% missing) |
| BaseExcess | float | Metabolic acid-base balance |
| HCO3 | float | Bicarbonate |
| FiO2 | float | Fraction of inspired oxygen |
| PaCO2 | float | Arterial CO2 pressure |
| SaO2 | float | Arterial oxygen saturation |
| AST | float | Liver enzyme |
| Alkalinephos | float | Liver enzyme |
| Calcium | float | Electrolyte |
| Chloride | float | Electrolyte |
| Magnesium | float | Electrolyte |
| Phosphate | float | Electrolyte |
| Potassium | float | Electrolyte |

### Demographics and Administrative

| Column | Type | Definition |
|---|---|---|
| Age | float | Patient age (years) |
| Gender | int | 0 = Female, 1 = Male |
| Unit1 | int | Medical ICU flag |
| Unit2 | int | Surgical ICU flag |
| HospAdmTime | float | Hours before ICU transfer from hospital admission |
| ICULOS | int | ICU length of stay counter — excluded from ML features (see Section 4) |
| Hour | int | Hour index within patient's stay |
| Patient_ID | int | Unique patient identifier |

## Target Variable

| Column | Type | Values |
|---|---|---|
| SepsisLabel | int | 1 = Sepsis currently, 0 = No Sepsis |
| SepsisLabel_6h | int | 1 = Sepsis within next 6 hours (model target), 0 = No |

The early-warning label `SepsisLabel_6h` was engineered by flagging the 6 hours immediately before each patient's confirmed sepsis onset. This shifts the prediction horizon forward: instead of confirming sepsis after it occurs, the model predicts imminent onset. The positive rate for this target is 2.71%.

---

# 3. Exploratory Data Analysis (EDA)

## 3.1 Class Imbalance

The dataset is heavily imbalanced at both the row and patient level:

| Level | No Sepsis | Sepsis | Sepsis Rate |
|---|---|---|---|
| Row-level | 1,524,294 | 27,916 | 1.80% |
| Patient-level | 37,404 | 2,932 | 7.27% |

The row-level rate is lower because each patient contributes many hourly rows, most of which occur before any deterioration begins. The patient-level rate is the more clinically meaningful figure.

A model that always predicts "No Sepsis" achieves 92.73% accuracy but catches zero sepsis cases — this is why **Precision-Recall AUC is used as the primary evaluation metric** rather than accuracy.

> <img width="1108" height="450" alt="newplot (23)" src="https://github.com/user-attachments/assets/e816a2bc-f5ba-4720-bdd2-defa4cf3af08" />

## 3.2 Sepsis Onset Timing

For the 2,932 sepsis patients, the ICU hour at which sepsis was first labeled:

| Statistic | Value | Interpretation |
|---|---|---|
| Median onset | Hour 29 | Half of patients develop sepsis within approximately 1.2 days |
| Mean onset | Hour 51 | Right-skewed due to late-onset cases |
| Earliest onset | Hour 1 | Patients admitted already septic |
| Latest onset | Hour 331 | Hospital-acquired sepsis after 14+ days |

The wide onset range (1-331 hours) is clinically realistic. The model must generalize across both early and late deterioration patterns.

> <img width="1108" height="400" alt="newplot (24)" src="https://github.com/user-attachments/assets/f4303c46-366d-4906-930b-390ff9b4e0d4" />

## 3.3 Missingness Analysis

Missing values are a defining characteristic of ICU datasets. Features were categorized by missingness level:

| Category | Features | Missingness | Action Taken |
|---|---|---|---|
| Drop — too sparse | EtCO2, Fibrinogen, Bilirubin_direct | >96% | Removed |
| High missing labs | Lactate, TroponinI, SaO2 | 70-85% | Forward-fill + flags |
| Moderate missing labs | pH, HCO3, FiO2, PTT | 50-70% | Forward-fill + flags |
| Low missing labs | WBC, Creatinine, Glucose | 15-25% | Forward-fill |
| Vital signs | HR, MAP, Resp, O2Sat | 10-15% | Use directly |

**MNAR — Missing Not At Random:** Labs are only ordered when clinicians suspect a problem. A missing Lactate value means the doctor did not order it, which is itself clinical information. This is exploited through missingness flag features (Section 3.8).

> <img width="1108" height="700" alt="newplot (25)" src="https://github.com/user-attachments/assets/35fb2542-36d8-449a-a7e9-020a8ec12939" />

## 3.4 Data Cleaning

**Duplicate check:** Zero exact duplicate rows and zero duplicate Patient_ID + Hour combinations found. Each row is a unique patient-hour observation.

**Outlier capping:** 993 clinically implausible values were identified across 7 vital signs. These are data entry errors. Values were capped at established clinical boundaries rather than removed — removing would discard all other valid measurements in that same row.

| Vital Sign | Lower Cap | Upper Cap | Outliers Corrected |
|---|---|---|---|
| HR (bpm) | 20 | 250 | 4 |
| MAP (mmHg) | 20 | 200 | 444 |
| SBP (mmHg) | 40 | 300 | 114 |
| DBP (mmHg) | 10 | 200 | 103 |
| Resp (br/min) | 1 | 80 | 50 |
| O2Sat (%) | 50 | 100 | 272 |
| Temp (degrees C) | 25 | 45 | 6 |
| Total | | | 993 |

## 3.5 Vital Signs Analysis

Mean vital signs compared between sepsis and non-sepsis rows:

| Vital Sign | No Sepsis | Sepsis | Difference |
|---|---|---|---|
| Heart Rate (bpm) | 83.4 | 89.7 | +7.48% |
| Respiratory Rate (br/min) | 18.5 | 20.2 | +9.45% |
| MAP (mmHg) | 82.1 | 79.8 | -2.76% |
| Temperature (degrees C) | 36.9 | 37.2 | +0.75% |
| O2Sat (%) | 97.2 | 96.8 | -0.41% |

No single vital sign strongly separates sepsis from non-sepsis at the individual reading level. The largest difference is Respiratory Rate at +9.45% — insufficient for reliable classification on its own. This confirms that temporal trends and feature combinations are required, which motivates the feature engineering in Section 3.8.

> <img width="1108" height="600" alt="newplot (26)" src="https://github.com/user-attachments/assets/01c3916b-490c-45ed-aa0a-f6d5d054fe6b" />

> <img width="1108" height="400" alt="newplot (27)" src="https://github.com/user-attachments/assets/7adf2fe8-7f4e-40ec-ae5a-bce5cf248a10" />

## 3.6 Lab Values and Demographics

**Lab values:** Lactate > 2 mmol/L is a clinical sepsis criterion. Despite 70% missingness, when Lactate is measured it shows a clear rightward shift in sepsis patients. Rising Creatinine (kidney dysfunction) and declining pH (acidosis) are also markers of sepsis-induced organ damage.

> <img width="1108" height="600" alt="newplot (30)" src="https://github.com/user-attachments/assets/2d706442-405f-4d1b-86e0-d3e98e762b4b" />

> <img width="1108" height="400" alt="newplot (28)" src="https://github.com/user-attachments/assets/8a197253-666f-423a-bf4d-78f34264e550" />

**ICU stay and demographics:**

| Metric | No Sepsis | Sepsis |
|---|---|---|
| Mean ICU stay (hours) | 36.9 | 58.8 |
| Median age (years) | ~62 | ~65 |
| Male patients (%) | ~55% | ~57% |

Sepsis patients stay approximately 60% longer in the ICU (58.8h vs 36.9h), reflecting increased clinical complexity.

> <img width="1108" height="400" alt="newplot (31)" src="https://github.com/user-attachments/assets/cc9a9f05-5d02-460a-a0f6-a003112dc208" />

> <img width="1108" height="400" alt="newplot (32)" src="https://github.com/user-attachments/assets/caa34d30-6a0d-4e8c-bb5c-8caead91d7b2" />

> <img width="1108" height="400" alt="newplot (33)" src="https://github.com/user-attachments/assets/91486ce3-b7c4-4d13-a7de-4a6a2861b50a" />

## 3.7 Temporal Patterns and Correlation Analysis

**Temporal patterns:** The most important EDA finding is that trends over time are more predictive than any single hourly reading. A visualization of Patient (sepsis onset at hour 250) shows HR rising gradually across 150+ hours before onset — a signal invisible in any individual reading but clear in the rolling mean.

> <img width="1108" height="550" alt="newplot (34)" src="https://github.com/user-attachments/assets/8941f711-31ca-417d-93e3-78904236c982" />

**Correlation analysis:** No feature exceeds 0.13 correlation with SepsisLabel. SBP, MAP, and DBP are highly intercorrelated (0.54-0.85). These weak linear relationships confirm that a machine learning model with engineered temporal features is needed.

> <img width="1108" height="550" alt="newplot (36)" src="https://github.com/user-attachments/assets/f33937a6-09a1-4208-aab9-e3cba7b60cd3" />

<img width="1108" height="400" alt="newplot (35)" src="https://github.com/user-attachments/assets/fc4703da-604c-462e-b645-567069a6a476" />

---

## 3.8 Feature Engineering

Based on EDA findings, the following pipeline transforms raw ICU time-series into ML-ready features. All operations were performed **after** the train/val/test split to prevent data leakage.

### Train / Validation / Test Split

Split by **Patient_ID** (not by rows) to ensure no patient appears in more than one split. Stratified by sepsis status to preserve class balance:

| Split | Patients | Rows | Sepsis Rate |
|---|---|---|---|
| Train (70%) | 28,235 | 1,086,436 | 7.27% |
| Validation (15%) | 6,050 | 233,750 | 7.27% |
| Test (15%) | 6,051 | 232,024 | 7.27% |

Zero patient overlap between any two splits confirmed. The SepsisLabel_6h positive rate across all splits is 2.71%.

> <img width="1108" height="460" alt="newplot (37)" src="https://github.com/user-attachments/assets/5040a42d-5b60-44de-a2ba-7f18f77a171d" />

### Why Split Before Engineering?

If the split were performed after feature engineering, the 6-hour rolling windows and forward-fill operations would inadvertently incorporate information from test-set patients into training calculations. This constitutes **data leakage** — the model would appear to perform better than it actually generalizes. Splitting first ensures the test set is truly unseen.

### Excluding ICULOS

ICULOS (ICU length of stay counter) was deliberately **excluded** from the feature set. Initial experiments with ICULOS included showed it as the single most important predictor (mean |SHAP| = 0.64), far exceeding all physiological features. This reflects a superficial pattern — longer stays accumulate more labeled hours — rather than genuine physiological learning. Removing ICULOS forces the model to learn from clinical signals, which is both more scientifically honest and more clinically meaningful.

### Forward Fill Lab Values

Lab values carried forward within each patient's timeline — per patient only, never across patients. Backward-fill was avoided as it would use future measurements to fill past hours (data leakage).

### Missingness Flags

Two features created for each of 10 key labs — **20 new features**:

| Feature | Description |
|---|---|
| Lactate_measured (0/1) | Was this lab actually drawn at this hour? |
| Lactate_hrs_since | Hours since last measurement, capped at 999 |

### 6-Hour Rolling Window Features

Five statistics over the previous 6 hours for all 7 vital signs — **35 new features**:

| Statistic | What it captures |
|---|---|
| Mean (e.g. HR_mean_6h) | Average level — smooths noise to reveal underlying trend |
| Min (e.g. HR_min_6h) | Dangerous dips within the window |
| Max (e.g. HR_max_6h) | Dangerous spikes within the window |
| Std (e.g. HR_std_6h) | Variability — high std signals physiological instability |
| Range (e.g. HR_range_6h) | Overall fluctuation magnitude |

> <img width="1108" height="445" alt="newplot (38)" src="https://github.com/user-attachments/assets/596a3ea1-3722-4212-a580-2589b5e2aaf3" />

### Slope / Trend Features

Linear slope over the previous 6 hours for 9 key features — **9 new features**. Two patients with the same current HR of 105 are clinically very different if one has been rising (+5/hour) versus falling (-5/hour).

| Feature | Rising means | Falling means |
|---|---|---|
| HR_slope_6h | Cardiac stress increasing | Stabilizing |
| MAP_slope_6h | BP recovering | BP dropping — dangerous |
| Resp_slope_6h | Breathing harder | Breathing easier |
| Lactate_slope_6h | Metabolic failure worsening | Recovering |
| pH_slope_6h | Recovering from acidosis | Acidosis worsening |

> <img width="1108" height="530" alt="newplot (39)" src="https://github.com/user-attachments/assets/68d8da77-2e65-42fa-a032-280e291b0e74" />

### SOFA-Inspired Derived Features

Five composite clinical indicators — **5 new features**:

| Feature | Formula | Sepsis Mean | Non-Sepsis Mean |
|---|---|---|---|
| shock_index | HR / SBP | 0.779 | 0.707 |
| pulse_pressure | SBP minus DBP | 60.3 | 60.6 |
| sirs_score | Count of SIRS criteria met (0-3) | 0.928 | 0.609 |
| map_low_flag | 1 if MAP < 65 mmHg | 0.141 | 0.098 |
| fever_flag | 1 if Temp outside 36-38 degrees C | 0.104 | 0.048 |

### Final Feature Set

| Feature Type | Count |
|---|---|
| Raw vital signs (outlier capped) | 7 |
| Raw labs (forward-filled) | 24 |
| Demographics | 5 |
| Rolling window features (6h) | 35 |
| Slope / trend features (6h) | 9 |
| Lab measured flags | 10 |
| Lab hours-since features | 10 |
| SOFA-inspired derived | 5 |
| TOTAL FEATURES | 105 |

ICULOS was excluded. Final dataset: 1,552,210 rows x 110 columns saved as train.parquet (98.9 MB), val.parquet (22.1 MB), test.parquet (21.9 MB).

---

# 4. Model Training

## 4.1 Predictive Task

Binary classification — predict for each patient-hour whether the patient will develop sepsis within the next 6 hours (SepsisLabel_6h = 1) or not. The positive rate is 2.71% — a class imbalance ratio of approximately 36:1.

## 4.2 Models Evaluated

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Baseline | Linear model; requires imputation and scaling |
| LightGBM (initial) | Primary candidate | Handles NaN natively; fast on 1.5M rows |
| LightGBM (Optuna-tuned) | Primary candidate | 50-trial Bayesian hyperparameter search |
| XGBoost (initial) | Primary candidate | Histogram method; scale_pos_weight = 35.9 |
| XGBoost (Optuna-tuned) | SELECTED BEST | 50-trial Bayesian search; best validation PR-AUC |

**Why not deep learning?** Tree-based gradient boosting methods are the state-of-the-art for tabular clinical data. Deep learning (LSTM, Transformer) would require more careful temporal alignment and would sacrifice interpretability — a critical requirement for clinical decision support.

## 4.3 Evaluation Metrics

| Metric | Why it matters |
|---|---|
| PR-AUC (primary) | Best metric for imbalanced data — directly measures minority class performance. Random baseline = 0.027 |
| Recall (Sensitivity) | Proportion of actual sepsis cases caught — missing a case is clinically dangerous |
| Precision | Proportion of sepsis alerts that are genuine — controls alert fatigue |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Secondary metric for overall discrimination quality |

**Why not accuracy?** A model that always predicts "No Sepsis" achieves 97.3% accuracy on the target variable. This is useless clinically — accuracy is completely misleading for imbalanced binary targets.

## 4.4 Hyperparameter Tuning

Optuna was used for Bayesian hyperparameter optimization — 50 trials per model, optimizing validation PR-AUC. Bayesian optimization is significantly more efficient than grid search for high-dimensional parameter spaces.

Best XGBoost parameters found:
- n_estimators: 916
- learning_rate: 0.0304
- max_depth: 5
- min_child_weight: 2
- subsample: 0.689
- colsample_bytree: 0.997
- reg_alpha: 0.207
- reg_lambda: 0.003

## 4.5 Model Comparison — Validation Set

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| XGBoost Tuned | 0.1274 | 0.8080 | 0.2968 | 0.1502 | 0.1994 |
| LightGBM Tuned | 0.1226 | 0.8074 | 0.3248 | 0.1377 | 0.1935 |
| LightGBM Initial | 0.1194 | 0.8036 | 0.3073 | 0.1421 | 0.1943 |
| Logistic Regression | 0.0847 | 0.7562 | 0.2922 | 0.1106 | 0.1605 |

XGBoost Tuned was selected as the best model by PR-AUC on the held-out validation set.

> <img width="1093" height="500" alt="newplot (40)" src="https://github.com/user-attachments/assets/b06164e1-ef51-43db-9d70-c5b6401aac1a" />

## 4.6 Final Test Set Results — XGBoost Tuned

The test set was evaluated once only, after model selection. These numbers were not used for any further tuning.

| Metric | Value | Interpretation |
|---|---|---|
| PR-AUC | 0.1261 | 4.7x better than random baseline (0.027) |
| ROC-AUC | 0.8055 | Correctly ranks sepsis patient above non-sepsis 80.6% of the time |
| Recall at threshold 0.737 | 28.4% | Catches 28 of 100 pre-sepsis cases at academic threshold |
| Precision at threshold 0.737 | 16.4% | 1 in 6 alerts is real at academic threshold |
| F1-Score | 0.2079 | |
| Recall at threshold 0.20 | 95.5% | Catches 95 of 100 pre-sepsis cases — clinical app setting |

### Confusion Matrix (threshold = 0.737)

| | Predicted No Sepsis | Predicted Sepsis |
|---|---|---|
| Actual No Sepsis | TN = 216,674 | FP = 9,071 |
| Actual Sepsis | FN = 4,498 | TP = 1,781 |

> <img width="1093" height="400" alt="newplot (41)" src="https://github.com/user-attachments/assets/37a326a5-2884-40c2-8788-978d33658c57" />

### Threshold Tradeoff Analysis

The model output is a continuous risk probability. The decision threshold converts this to a binary alert. Different thresholds serve different clinical contexts:

| Threshold | Recall | Precision | TP | FP | FN | Use Case |
|---|---|---|---|---|---|---|
| 0.20 | 95.5% | 3.8% | 5,994 | 150,167 | 285 | ICU triage / clinical app |
| 0.40 | 78.8% | 6.1% | ~4,948 | ~76,000 | ~1,331 | Moderate sensitivity |
| 0.60 | 53.4% | 10.3% | ~3,353 | ~29,000 | ~2,926 | Reduced false alarms |
| 0.733 | 28.4% | 16.4% | 1,781 | 9,071 | 4,498 | Academic F1-optimal |

> <img width="1936" height="1172" alt="image" src="https://github.com/user-attachments/assets/75670bfe-22d4-45a4-9b8e-f0e08894c926" />

> <img width="2256" height="730" alt="image" src="https://github.com/user-attachments/assets/4269fa9c-cc9a-4808-8eca-a7c04fc2ec58" />

> <img width="1510" height="1178" alt="image" src="https://github.com/user-attachments/assets/4dc8cb7d-b62a-44bb-a24c-23f2fb94c5d7" />

## 4.7 Model Interpretability — SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to explain both global feature importance and individual patient predictions.

### Global Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean SHAP | Clinical Meaning |
|---|---|---|---|
| 1 | Lactate | 0.276 | Classic sepsis biomarker |
| 2 | FiO2 | 0.263 | Fraction of inspired oxygen — respiratory support level |
| 3 | HospAdmTime | 0.190 | Time from hospital admission to ICU transfer |
| 4 | Unit2 | 0.174 | Surgical vs medical ICU — different risk profiles |
| 5 | Temp_max_6h | 0.154 | Peak temperature in last 6h — rolling temporal feature |
| 6 | Lactate_hrs_since | 0.120 | Missingness flag — time since last Lactate draw |
| 7 | WBC | 0.113 | White blood cell count — immune response |
| 8 | BUN | 0.109 | Blood urea nitrogen — kidney function deterioration |
| 9 | Resp_mean_6h | 0.080 | Average respiratory rate over 6h — temporal trend |
| 10 | HR_max_6h | 0.058 | Peak heart rate over 6h — rolling cardiac feature |

After removing ICULOS, the top features are clinically meaningful physiological signals. Lactate (#1) is a literal criterion for clinical sepsis diagnosis. Rolling features (Temp_max_6h, Resp_mean_6h, HR_max_6h) validate the temporal engineering strategy. Missingness flags (Lactate_hrs_since) confirm the MNAR hypothesis.

> <img width="1510" height="1184" alt="image" src="https://github.com/user-attachments/assets/74486335-0a08-4f26-90bb-c298e7cd696d" />

> <img width="1144" height="1242" alt="image" src="https://github.com/user-attachments/assets/290d12d5-11e6-4864-ad8b-4c706775ae57" />

### Individual Patient Explanation

For a true positive patient predicted at 95.4% risk (correctly flagged), the SHAP waterfall plot shows exactly which features drove the prediction up or down from the baseline.

> <img width="1144" height="1186" alt="image" src="https://github.com/user-attachments/assets/60b3672b-a873-4534-90dd-06df4830ddbd" />

### Feature Type Contribution Summary

| Feature Type | Avg SHAP | Interpretation |
|---|---|---|
| Raw vitals and labs | 0.0505 | Highest — Lactate and FiO2 dominate |
| Rolling window (6h) | 0.0226 | Second — temporal patterns add value |
| Missingness flags | 0.0139 | Third — MNAR signal is real |
| Slope / trend (6h) | 0.0066 | Lowest globally — more impactful for individual patients |

---

# 5. Application of the Trained Models

## 5.1 Streamlit Early Warning Application

A Streamlit web application was built to demonstrate the model's clinical utility. The app loads the held-out test set (6,051 patients, 232,024 observations) as a simulated ICU patient database and allows interactive exploration of model predictions.

**Run instructions:**
```bash
cd ~/Desktop/sepsis_project
source venv/bin/activate
streamlit run app.py
```

## 5.2 Application Features

### Sidebar Controls

- **Alert Mode:** Three clinical operating modes
  - High Sensitivity (threshold 0.20) — catches 95.5% of pre-sepsis cases; designed for ICU triage
  - Balanced (threshold 0.737) — F1-optimal; recommended for standard monitoring
  - High Precision (threshold 0.60) — fewer false alarms; for resource-limited settings
  - Custom — manual threshold adjustment via slider
- **Patient Selection:** Filter by sepsis-positive patients, or browse all 6,051 test patients
- **ICU Hour Slider:** Navigate through a patient's full ICU stay hour by hour

### Tab 1 — Overview

Risk Gauge (semicircular needle with alert threshold line), Vital Signs Panel (last 12 hours of HR, MAP, Resp, O2Sat with clinical alert thresholds), and Key Clinical Indicators (metric cards for current values).

### Tab 2 — Risk Timeline

Full-stay risk trajectory plot showing the model's predicted risk score at every ICU hour. Alert zones above threshold are filled in red. Confirmed sepsis onset is marked with an amber line. Summary statistics include peak risk, hours above threshold, and total ICU stay length.

### Tab 3 — Clinical Values

All clinical measurements at the selected hour: vital signs with normal range indicators, laboratory values, clinical risk scores (SIRS, Shock Index), and engineered temporal features (rolling means, slopes, missingness flags).

### Tab 4 — SHAP Explanation

Per-patient SHAP waterfall plot with a plain-language reading guide. Red bars show features that increased risk; blue bars show features that decreased risk.

> <img width="2940" height="1594" alt="image" src="https://github.com/user-attachments/assets/4a40849a-9c7b-45c4-913d-2dbdba927e20" />

> <img width="2940" height="1594" alt="image" src="https://github.com/user-attachments/assets/805dda3c-3a0b-434e-96f2-10942a8b28e8" />

> <img width="2940" height="1594" alt="image" src="https://github.com/user-attachments/assets/726568a6-3344-4a6e-aab4-d2d8b1197078" />

> <img width="2940" height="1594" alt="image" src="https://github.com/user-attachments/assets/ed60a53f-76e9-4b0a-ba26-2cd27e11d0a3" />

---

# 6. Conclusion

## Summary

This project developed a complete machine learning pipeline for early sepsis prediction from ICU clinical data. Starting from 43 raw clinical measurements collected hourly from 40,336 ICU patients, the pipeline engineered 105 temporal features, trained and compared four models, selected XGBoost Tuned (PR-AUC 0.1261, ROC-AUC 0.8055), demonstrated 95.5% case detection at threshold 0.20, validated SHAP feature importance against clinical knowledge, and deployed an interactive early warning application.

### Benchmarking

| System | ROC-AUC | Precision | Notes |
|---|---|---|---|
| Epic Sepsis Model (deployed) | — | ~12% | Commercial clinical AI, widely deployed in US hospitals |
| RNN best result (PhysioNet 2019) | 0.82 | 21% | Best published result on this exact dataset |
| XGBoost Tuned (this project) | 0.806 | 16.4% | Interpretable tabular ML, no deep learning |

Our precision of 16.4% exceeds the Epic Sepsis Model (~12%), one of the most widely deployed clinical AI tools in US hospitals. The model matches the best published result on this dataset using interpretable tabular ML.

## Limitations

- **PR-AUC of 0.126 is modest.** Sepsis prediction from tabular ICU data is genuinely difficult — the signal-to-noise ratio is low, and many physiological processes that precede sepsis overlap with other conditions.
- **Administrative features rank highly.** HospAdmTime and Unit2 appear in the top 5 SHAP features, indicating the model uses patient admission context in addition to purely physiological signals.
- **Single hospital system.** The PhysioNet 2019 dataset originates from one hospital system. The model has not been validated externally.
- **Probability calibration.** The Brier score of 0.153 indicates the model's predicted probabilities are not well-calibrated as literal probabilities. The model is better used for relative risk ranking than absolute probability estimation.
- **Research prototype.** This system has not undergone clinical validation, regulatory review, or FDA approval. It must not be used for real clinical decisions.

## Lessons Learned

- The choice of primary metric is as important as the model. Using accuracy on an imbalanced target produces a misleading picture of model quality.
- Feature engineering contributed as much as model selection. The improvement from Logistic Regression (PR-AUC 0.085) to XGBoost Tuned (0.127) reflects both temporal feature value and model capacity.
- Splitting before feature engineering is non-negotiable. Rolling windows and forward-fill across the full dataset before splitting would constitute data leakage.
- Removing ICULOS improved scientific integrity. ICULOS was the single most powerful predictor when included but reflected administrative patterns rather than physiological learning.
- Missingness is a feature, not a problem. The MNAR pattern in lab ordering contributed meaningfully via the measured and hrs_since flags.
- Threshold selection is a clinical decision, not a statistical one. Providing multiple operating modes in the app reflects the reality that different clinical contexts require different tradeoffs.

## Future Research Directions

- LSTM / Transformer modeling for longer-range temporal dependency capture
- Multi-hospital external validation on datasets such as MIMIC-IV
- Consecutive alert filtering (requiring 2-3 consecutive hours above threshold) to reduce false alarm rates
- Probability calibration using Platt scaling or isotonic regression
- Real-time EHR integration via HL7 FHIR APIs
- Demographic bias analysis across age, gender, race, and insurance status

---

# 7. References

1. Reyna, M. A., et al. (2019). "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine, 48(2), 210-217.

2. Singer, M., et al. (2016). "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." JAMA, 315(8), 801-810.

3. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." Advances in Neural Information Processing Systems, 30.

4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD, 785-794.

5. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." Advances in Neural Information Processing Systems, 30.

6. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." Proceedings of the 25th ACM SIGKDD, 2623-2631.

7. Fleuren, L. M., et al. (2020). "Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy." Intensive Care Medicine, 46(3), 383-400.

8. Wong, A., et al. (2021). "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients." JAMA Internal Medicine, 181(8), 1065-1070.

9. PhysioNet Challenge 2019 Dataset: https://physionet.org/content/challenge-2019/1.0.0/
