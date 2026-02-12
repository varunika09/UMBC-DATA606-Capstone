# Early Sepsis Detection from ICU Time-Series (PhysioNet 2019) using Aggregated Clinical Features

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
Author: Varunika Bussa  
GitHub Repo: [https://github.com/varunika09/UMBC-DATA606-Capstone.git]  
LinkedIn: [www.linkedin.com/in/varunika-bussa-99a837233]  
Slides: [To be added]  
YouTube: [To be added]  

---

## 1. Background

### What is this project about?
Sepsis is a life-threatening condition where the body’s response to infection can cause organ dysfunction. In ICU settings, detecting sepsis early is critical because early treatment can significantly improve outcomes. This capstone project builds a machine learning system that predicts **sepsis onset** using **hourly ICU measurements** (vital signs, lab tests, and demographics).

This project uses the PhysioNet 2019 Sepsis Challenge dataset where each patient has a time series of clinical measurements over their ICU stay. Each hour includes measurements (often with missing values), and a label indicates whether the patient is in the sepsis onset window according to a Sepsis-3–based definition.

### Why does it matter?
- **Patient impact:** Earlier identification can support faster clinical intervention.
- **Hospital impact:** False alarms consume limited ICU resources, while late prediction can be dangerous.
- **Industry relevance:** Predictive modeling for clinical deterioration is a major real-world application of ML in healthcare, with strong emphasis on interpretability and evaluation under class imbalance.

### Research questions
1. Can we accurately predict `SepsisLabel` using ICU vitals, labs, and demographics at the hourly level?
2. Which clinical measurements and trends (e.g., rising heart rate, low blood pressure, elevated lactate) are most predictive of sepsis onset?
3. How do different ML models perform under class imbalance (baseline vs tree-based models)?
4. Can we provide interpretable explanations (feature importance / SHAP) to support clinical understanding of predictions?

---

## 2. Data

### Data source
**PhysioNet / Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data**  
Dataset landing page: https://physionet.org/content/challenge-2019/1.0.0/  
The challenge repository provides **one file per patient** (e.g., `p00101.psv`).

### Data size
According to the dataset description, the complete training database is approximately **42 MB** (two parts).  
In the Kaggle mirror/download, the packaged size may appear larger due to packaging/versioning, but the underlying patient files and variables are the same.

### Data shape
The training data consists of:
- **Training set A:** 20,336 subjects  
- **Training set B:** 20,000 subjects  
- **Total:** 40,336 subjects (one file per subject)

Each patient file contains:
- A variable number of rows (hours in ICU).
- A fixed set of columns (clinical variables + label).

Because each patient has a different ICU length of stay, the overall dataset size is best described as:
- **~40k patient files**
- **Hundreds of thousands to millions of hourly records** (to be reported after loading and aggregation)

### Time period
The dataset is **de-identified**. Time is represented through:
- `ICULOS` = ICU length of stay in hours since ICU admission  
This supports time-aware modeling without real calendar dates.

### What does each row represent?
Each row represents:
> **One hour of ICU monitoring for one patient** (a “patient-hour”).

Example format:
`HR | O2Sat | Temp | ... | HospAdmTime | ICULOS | SepsisLabel`

Missing values (`NaN`) indicate that a measurement was not recorded in that hour (common in ICU data, especially labs).

---

## 3. Variables, Target, and Features

### Data dictionary (column groups)
Each patient-hour file contains **41 total columns**:

#### Vital signs (8)
| Column | Type | Definition | Typical values |
|---|---|---|---|
| HR | float | Heart rate (beats/min) | continuous |
| O2Sat | float | Pulse oximetry (%) | 0–100 |
| Temp | float | Temperature (°C) | continuous |
| SBP | float | Systolic blood pressure (mm Hg) | continuous |
| MAP | float | Mean arterial pressure (mm Hg) | continuous |
| DBP | float | Diastolic blood pressure (mm Hg) | continuous |
| Resp | float | Respiratory rate (breaths/min) | continuous |
| EtCO2 | float | End tidal CO2 (mm Hg) | continuous |

#### Laboratory values (26)
Examples include:
- Lactate, Creatinine, BUN, Glucose, WBC, Platelets, pH, PaCO2, etc.
All are numeric continuous lab measurements recorded intermittently.

#### Demographics / administrative / time variables (6)
| Column | Type | Definition | Values |
|---|---|---|---|
| Age | float | Age in years (100 for age ≥ 90) | continuous |
| Gender | int | 0 female, 1 male | {0,1} |
| Unit1 | int | ICU unit identifier | {0,1} |
| Unit2 | int | ICU unit identifier | {0,1} |
| HospAdmTime | float | Hours between hospital admit and ICU admit | continuous |
| ICULOS | int | ICU length-of-stay in hours | positive integer |

#### Outcome label (1)
| Column | Type | Definition | Values |
|---|---|---|---|
| SepsisLabel | int | Sepsis onset indicator (Sepsis-3–based) | {0,1} |

### Target/label for ML model
**Target variable:** `SepsisLabel`  
- `1` indicates the patient is in the sepsis onset window  
- `0` indicates no sepsis label at that hour  

This project will predict `SepsisLabel` at the hourly level (patient-hour prediction).

### Clinically meaningful predictors (medical intuition)
Sepsis and clinical deterioration often correlate with patterns such as:
- **Hemodynamic instability:** low blood pressure (SBP/MAP/DBP), rising heart rate (HR)
- **Respiratory stress:** increased Resp, lower oxygen saturation (O2Sat)
- **Perfusion/organ stress:** elevated **Lactate**, kidney markers (Creatinine/BUN), acid-base markers (pH, BaseExcess, HCO3)
- **Inflammation/infection response:** abnormal **WBC**, platelet changes, fever/hypothermia via Temp

The model will not “diagnose” sepsis directly; it will learn statistical relationships between these measurements and the provided `SepsisLabel`.

### Features/predictors used for modeling (engineered tabular features)
To make the time-series usable for standard ML models (LogReg / RandomForest / XGBoost), hourly values will be converted into robust features, including:

**1) Current snapshot features**
- Latest available values (e.g., HR at hour t, MAP at hour t)

**2) Rolling window statistics (time-aware features)**
For the last 3–6 hours (configurable), compute:
- Mean / min / max / standard deviation
- Trends (difference vs 3 hours ago; slope proxy)

**3) Missingness indicators**
Because missing values are informative in ICU workflows (labs not measured every hour), add:
- “was_measured” flags per variable
- counts of missing values in last window

**4) Patient context features**
- Age, Gender, Unit1/Unit2, HospAdmTime, ICULOS

### Planned ML models (for proposal)
- **Baseline:** Logistic Regression (interpretable baseline)
- **Nonlinear models:** Random Forest, XGBoost (strong for tabular + missingness patterns)
- **Evaluation:** ROC-AUC + Precision-Recall AUC, Recall/F1 (important under class imbalance)
- **Interpretability:** feature importance and SHAP to explain top drivers of risk predictions

---

## Final product (Streamlit)
A Streamlit prototype that allows a user to:
- Select or input a patient snapshot (current hour vitals/labs/demographics)
- Get predicted **sepsis risk (probability)** and classification (0/1)
- View explanation of which factors contributed most to the prediction (e.g., lactate, MAP trend, HR trend)

This demonstrates an interpretable early-warning ML workflow suitable for healthcare analytics contexts.

---
