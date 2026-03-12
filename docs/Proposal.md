# Early Sepsis Risk Prediction Using ICU Clinical Data

Prepared for UMBC Data Science Master Degree Capstone  
by Dr. Chaojie (Jay) Wang

**Author:** Varunika Bussa  
**GitHub Repository:** [https://github.com/varunika09/UMBC-DATA606-Capstone.git](https://github.com/varunika09/UMBC-DATA606-Capstone.git)  
**LinkedIn Profile:** [www.linkedin.com/in/varunika-bussa-99a837233](https://www.linkedin.com/in/varunika-bussa-99a837233)  
**PowerPoint Presentation:** *(link to be added)*  
**YouTube Video:** *(link to be added)*

---

# 1. Background

## What is this project about?

This project focuses on the early prediction of sepsis in ICU patients using machine learning. Sepsis is a life-threatening condition caused by the body's extreme response to infection. If not detected early, it can lead to organ failure and death — early intervention significantly improves survival rates.

The goal is to build a predictive model that estimates whether a patient is at risk of developing sepsis using routinely collected ICU vital signs and laboratory measurements, and to deploy it as an interpretable early-warning application.

## Why does it matter?

- Sepsis affects approximately **1.7 million Americans annually** and causes **270,000 deaths per year** (CDC)
- It is a leading cause of ICU mortality worldwide
- Existing clinical scoring systems (SOFA, qSOFA, SIRS) are **reactive** — they confirm sepsis after it develops, not before
- Hospitals collect continuous patient monitoring data, but automated early-warning systems are not always implemented
- A machine learning–based system can detect subtle deterioration patterns **hours before** clinical diagnosis, enabling earlier intervention

## Research Questions

1. Can ICU vital signs and lab values predict sepsis onset before it is clinically diagnosed?
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

### Vital Signs — recorded approximately every hour

| Column | Type | Definition |
|---|---|---|
| HR | float | Heart rate (bpm) |
| O2Sat | float | Oxygen saturation (%) |
| Temp | float | Body temperature (°C) |
| SBP | float | Systolic blood pressure (mmHg) |
| MAP | float | Mean arterial pressure (mmHg) |
| DBP | float | Diastolic blood pressure (mmHg) |
| Resp | float | Respiratory rate (breaths/min) |
| EtCO2 | float | End-tidal CO2 — **dropped** (96.3% missing) |

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
| Bilirubin_direct | float | Liver function — **dropped** (99.8% missing) |
| Fibrinogen | float | Clotting factor — **dropped** (99.3% missing) |
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
| HospAdmTime | float | Hours before ICU transfer from hospital admission (negative = admitted to hospital first) |
| ICULOS | int | ICU length of stay counter (hours since ICU admission) |
| Hour | int | Hour index within patient's stay |
| Patient_ID | int | Unique patient identifier |

## Target Variable

| Column | Type | Values |
|---|---|---|
| SepsisLabel | int | 1 = Sepsis, 0 = No Sepsis |

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

<img width="1108" height="450" alt="image" src="https://github.com/user-attachments/assets/8c8363aa-dc91-41e7-b78b-b44b66daa4a5" />


## 3.2 Sepsis Onset Timing

For the 2,932 sepsis patients, the ICU hour at which sepsis was first labeled:

| Statistic | Value | Interpretation |
|---|---|---|
| Median onset | Hour 29 | Half of patients develop sepsis within ~1.2 days |
| Mean onset | Hour 51 | Right-skewed due to late-onset cases |
| Earliest onset | Hour 1 | Patients admitted already septic |
| Latest onset | Hour 331 | Hospital-acquired sepsis after 14+ days |

The wide onset range (1–331 hours) is clinically realistic. The model must generalize across both early and late deterioration patterns.

<img width="1108" height="400" alt="image" src="https://github.com/user-attachments/assets/7fd08ce8-2777-41ee-8783-cad65ea16d78" />


## 3.3 Missingness Analysis

Missing values are a defining characteristic of ICU datasets. Features were categorized by missingness level:

| Category | Features | Missingness | Action Taken |
|---|---|---|---|
| Drop — too sparse | EtCO2, Fibrinogen, Bilirubin_direct | >96% | Removed |
| High missing labs | Lactate, TroponinI, SaO2 | 70–85% | Forward-fill + flags |
| Moderate missing labs | pH, HCO3, FiO2, PTT | 50–70% | Forward-fill + flags |
| Low missing labs | WBC, Creatinine, Glucose | 15–25% | Forward-fill |
| Vital signs | HR, MAP, Resp, O2Sat | 10–15% | Use directly |

**MNAR — Missing Not At Random:** Labs are only ordered when clinicians suspect a problem. A missing Lactate value means the doctor did not order it, which is itself clinical information. This is exploited through missingness flag features (Section 3.7).

<img width="1108" height="700" alt="image" src="https://github.com/user-attachments/assets/25feb18c-c8cb-4134-a54b-f4f494210360" />


## 3.4 Data Cleaning

**Duplicate check:** Zero exact duplicate rows and zero duplicate Patient_ID + Hour combinations found. Each row is a unique patient-hour observation. ✅

**Outlier capping:** 993 clinically implausible values were identified across 7 vital signs (e.g., HR = 280 bpm, MAP = 300 mmHg). These are data entry errors. Values were **capped** at established clinical boundaries rather than removed — removing would discard all other valid measurements in that same row.

| Vital Sign | Lower Cap | Upper Cap | Outliers Corrected |
|---|---|---|---|
| HR (bpm) | 20 | 250 | 4 |
| MAP (mmHg) | 20 | 200 | 444 |
| SBP (mmHg) | 40 | 300 | 114 |
| DBP (mmHg) | 10 | 200 | 103 |
| Resp (br/min) | 1 | 80 | 50 |
| O2Sat (%) | 50 | 100 | 272 |
| Temp (°C) | 25 | 45 | 6 |
| **Total** | | | **993** |

## 3.5 Vital Signs Analysis

Mean vital signs compared between sepsis and non-sepsis rows:

| Vital Sign | No Sepsis | Sepsis | Difference |
|---|---|---|---|
| Heart Rate (bpm) | 83.4 | 89.7 | +7.48% |
| Respiratory Rate (br/min) | 18.5 | 20.2 | +9.45% |
| MAP (mmHg) | 82.1 | 79.8 | −2.76% |
| Temperature (°C) | 36.9 | 37.2 | +0.75% |
| O2Sat (%) | 97.2 | 96.8 | −0.41% |

No single vital sign strongly separates sepsis from non-sepsis at the individual reading level. The largest difference is Respiratory Rate at +9.45% — insufficient for reliable classification on its own. This confirms that **temporal trends and feature combinations** are required, which motivates the feature engineering in Section 3.7.

<img width="1108" height="600" alt="image" src="https://github.com/user-attachments/assets/2ce16a43-7a45-40d3-8ea5-c033da838b8a" />


## 3.6 Lab Values and Demographics

**Lab values:** Lactate > 2 mmol/L is a clinical sepsis criterion. Despite 70% missingness, when Lactate is measured it shows a clear rightward shift in sepsis patients. Rising Creatinine (kidney dysfunction) and declining pH (acidosis) are also markers of sepsis-induced organ damage.

<img width="1108" height="400" alt="image" src="https://github.com/user-attachments/assets/388be072-ef08-40e7-a9ff-e510230e17cd" />

<img width="1108" height="600" alt="image" src="https://github.com/user-attachments/assets/fa8acaf5-1be7-46c6-8374-527fcc342b3f" />


**ICU stay and demographics:**

| Metric | No Sepsis | Sepsis |
|---|---|---|
| Mean ICU stay (hours) | 36.9 | 58.8 |
| Median age (years) | ~62 | ~65 |
| Male patients (%) | ~55% | ~57% |

Sepsis patients stay ~60% longer in the ICU (58.8h vs 36.9h), reflecting increased clinical complexity.

<img width="1108" height="400" alt="image" src="https://github.com/user-attachments/assets/d114b2ed-a4bb-4655-9382-691febe44cd1" />


## 3.7 Temporal Patterns and Correlation Analysis

**Temporal patterns:** The most important EDA finding is that trends over time are more predictive than any single hourly reading. A visualization of Patient 9 (sepsis onset at hour 250) shows HR rising gradually across 150+ hours before onset — a signal invisible in any individual reading but clear in the rolling mean.

<img width="1108" height="500" alt="image" src="https://github.com/user-attachments/assets/c3b91260-5af5-43da-b107-598da2fe2602" />


**Correlation analysis:** No feature exceeds 0.13 correlation with SepsisLabel. SBP, MAP, and DBP are highly intercorrelated (0.54–0.85). These weak linear relationships confirm that a machine learning model with engineered temporal features is needed — no simple rule-based threshold will work.

<img width="1108" height="550" alt="image" src="https://github.com/user-attachments/assets/349d71b9-a94e-43c9-a20a-b342da5063c9" />


---

## 3.8 Feature Engineering

Based on EDA findings, the following pipeline transforms raw ICU time-series into ML-ready features. All operations were performed **after** the train/val/test split.

### Train / Validation / Test Split

Split by **Patient_ID** (not by rows) to ensure no patient appears in more than one split. Stratified to preserve the 7.27% sepsis rate:

| Split | Patients | Rows | Sepsis Rate |
|---|---|---|---|
| Train (70%) | 28,235 | 1,086,436 | 7.27% |
| Validation (15%) | 6,050 | 233,750 | 7.27% |
| Test (15%) | 6,051 | 232,024 | 7.27% |

Zero patient overlap between any two splits confirmed. ✅

### Forward Fill Lab Values

Lab values carried forward within each patient's timeline — per patient only, never across patients. Backward-fill was avoided as it would use future measurements to fill past hours (data leakage).

### Missingness Flags

Two features created for each of 10 key labs (20 new features):

| Feature | Description |
|---|---|
| `Lactate_measured` (0/1) | Was this lab actually drawn at this hour? |
| `Lactate_hrs_since` | Hours since last measurement, capped at 999 |

### 6-Hour Rolling Window Features

Five statistics over the previous 6 hours for all 7 vital signs — **35 new features**:

| Statistic | What it captures |
|---|---|
| Mean (e.g. HR_mean_6h) | Average level — smooths noise to reveal underlying trend |
| Min (e.g. HR_min_6h) | Dangerous dips within the window |
| Max (e.g. HR_max_6h) | Dangerous spikes within the window |
| Std (e.g. HR_std_6h) | Variability — high std signals physiological instability |
| Range (e.g. HR_range_6h) | Overall fluctuation magnitude |

── HR Rolling Features Example (Patient sample, first 8 hours) ──
 Hour      HR  HR_mean_6h  HR_min_6h  HR_max_6h  HR_std_6h  HR_range_6h
    0     NaN         NaN        NaN        NaN      0.000          NaN
    1  97.000      97.000     97.000     97.000      0.000        0.000
    2  89.000      93.000     89.000     97.000      5.657        8.000
    3  90.000      92.000     89.000     97.000      4.359        8.000
    4 103.000      94.750     89.000    103.000      6.551       14.000
    5 110.000      97.800     89.000    110.000      8.871       21.000
    6 108.000      99.500     89.000    110.000      8.961       21.000
    7 106.000     101.000     89.000    110.000      9.209       21.000

### Slope / Trend Features

Linear slope over the previous 6 hours for 9 key features — **9 new features**. Two patients with the same current HR of 105 are clinically very different if one has been rising (+5/hour) versus falling (−5/hour). Slope captures this direction.

| Feature | Rising = | Falling = |
|---|---|---|
| HR_slope_6h | Cardiac stress increasing | Stabilizing |
| MAP_slope_6h | BP recovering | BP dropping — dangerous |
| Resp_slope_6h | Breathing harder | Breathing easier |
| Lactate_slope_6h | Metabolic failure worsening | Recovering |
| pH_slope_6h | Recovering from acidosis | Acidosis worsening |

> 📊 *Insert visualization: Resp slope 2-panel chart (raw+mean / slope over time with sepsis onset marker)*

### SOFA-Inspired Derived Features

Five composite clinical indicators — **5 new features**:

| Feature | Formula | Sepsis Mean | Non-Sepsis Mean |
|---|---|---|---|
| shock_index | HR / SBP | 0.779 | 0.707 ↑ |
| pulse_pressure | SBP − DBP | 60.3 | 60.6 (minimal) |
| sirs_score | Count of SIRS criteria met (0–3) | 0.928 | 0.609 ↑ |
| map_low_flag | 1 if MAP < 65 mmHg | 0.141 | 0.098 ↑ |
| fever_flag | 1 if Temp outside 36–38°C | 0.104 | 0.048 ↑ |

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
| **TOTAL FEATURES** | **106** |

Final dataset: **1,552,210 rows × 110 columns** saved as `train.parquet` (98.9 MB), `val.parquet` (22.1 MB), `test.parquet` (21.9 MB).

---

# 4. Modeling

## Predictive Task

Binary classification — predict for each patient-hour whether **SepsisLabel = 1** (sepsis risk) or **SepsisLabel = 0** (no sepsis).

## Models

| Model | Role |
|---|---|
| Logistic Regression | Baseline — establishes minimum performance |
| XGBoost | Primary candidate |
| LightGBM | Primary candidate — handles NaN natively, efficient on large data |

Hyperparameter tuning will use **Optuna** (Bayesian optimization).

## Evaluation Metrics

| Metric | Why it matters |
|---|---|
| **PR-AUC** (primary) | Best metric for imbalanced data — directly measures minority class performance |
| Recall | Proportion of actual sepsis cases caught — missing a case is clinically dangerous |
| Precision | Proportion of sepsis alerts that are real — controls alert fatigue |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Secondary metric for overall discrimination |

**Clinical priority: high recall** — missing a sepsis case is more dangerous than a false alarm.

## Interpretability

Model predictions will be explained using **SHAP (SHapley Additive exPlanations)** to identify which features drove each individual prediction — making the model explainable to clinicians.

## Final Product — Streamlit Application

The final deliverable is a Streamlit web application that:

1. Loads the held-out test set as a simulated ICU patient database
2. Allows the user to select a Patient_ID and ICU hour
3. Retrieves that patient's engineered features automatically
4. Outputs predicted sepsis risk probability and binary classification
5. Displays the top SHAP features driving that specific prediction
