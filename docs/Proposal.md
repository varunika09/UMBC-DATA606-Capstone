# Early Sepsis Detection from ICU Time-Series Using Aggregated Clinical Features

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
Author: Varunika Bussa  
GitHub Repo: https://github.com/varunika09/UMBC-DATA606-Capstone.git 
LinkedIn: www.linkedin.com/in/varunika-bussa-99a837233  
Slides: [To be added]  
YouTube: [To be added]  

---

## 1. Background

### What is this project about?
Sepsis is a life-threatening condition caused by an extreme response to infection that can lead to organ dysfunction and death if not identified early. In ICU settings, early detection is critical because timely intervention can significantly improve outcomes.

This project builds a machine learning model that predicts **SepsisLabel** (0/1) using hourly ICU measurements (vitals, labs, demographics, and ICU timing variables). The dataset is based on the PhysioNet 2019 Sepsis Prediction Challenge, where the label corresponds to the onset window of sepsis according to a Sepsis-3–based definition.

Rather than using deep sequence models, the project converts patient time-series into **aggregated time-window features** (e.g., rolling averages, trends, and variability over the last 3–6 hours), enabling strong and interpretable tabular ML models (e.g., Logistic Regression, Random Forest, XGBoost).

### Why does it matter?
- **Clinical impact:** Sepsis can deteriorate quickly; delays increase risk of mortality.
- **Operational impact:** False alarms waste limited hospital resources, while late predictions can be harmful.
- **Industry relevance:** Early warning systems are a major real-world application of ML in healthcare, where interpretability and imbalance-aware evaluation matter.

### Research questions
1. Can we accurately predict **SepsisLabel** using ICU vitals, labs, and patient context at the hourly level?
2. Which measurements and short-term trends are most predictive of sepsis onset (e.g., rising HR, falling MAP, elevated lactate)?
3. Which ML model performs best on this task under severe class imbalance?
4. Can model explanations (feature importance / SHAP) highlight clinically meaningful predictors?

---

## 2. Data

### Data source
PhysioNet / Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data  
https://physionet.org/content/challenge-2019/1.0.0/

Original challenge data is provided as **one file per patient** (PSV). For this project, a trusted flattened version (`Dataset.csv`) is used, containing all patient-hour records from Training Set A and B with a patient identifier.

### Data size
- Total patient-hour records: **1,552,210 rows**
- Total columns: **44 columns** (including identifiers and outcome label)
- Unique patients: **40,336**
- Positive label prevalence (`SepsisLabel=1`): **27,916 rows (~1.80%)**
- Negative label prevalence (`SepsisLabel=0`): **1,524,294 rows (~98.20%)**

This indicates a **highly imbalanced classification problem**, consistent with real sepsis prediction tasks.

### Time representation
The dataset is de-identified. Time is represented as:
- `ICULOS`: ICU length-of-stay in hours since ICU admission
- A patient-hour row corresponds to measurements captured at a given ICU hour.

### What does each row represent?
Each row represents:
> One hour of ICU monitoring for one patient (“patient-hour”).

Measurements may be missing (`NaN`) when a variable was not recorded during that hour, which is common in ICU settings (especially for lab tests).

---

## 3. Variables, Target, and Features

### Target variable (label)
**Target column:** `SepsisLabel`  
- `1` indicates the patient is in the sepsis onset window (Sepsis-3–based definition per challenge)  
- `0` indicates no sepsis label at that hour

This project predicts `SepsisLabel` directly using information available up to that hour.

### Key clinical predictors (medical intuition)
Sepsis onset often correlates with patterns such as:
- **Hemodynamic instability:** decreasing blood pressure (SBP/MAP/DBP) and compensatory higher HR
- **Respiratory stress:** increased Resp and reduced oxygenation (O2Sat)
- **Perfusion and metabolic stress:** elevated **Lactate**, acid-base imbalance (pH, HCO3, BaseExcess)
- **Inflammation/infection response:** abnormal **WBC**, platelet changes, and temperature abnormalities
- **Organ dysfunction signals:** elevated Creatinine/BUN (kidney stress), bilirubin (liver involvement)

The model learns statistical relationships between these measurements and `SepsisLabel` rather than explicitly applying clinical rules.

### Feature selection and aggregation (what will feed the ML models)
The raw dataset is hourly time-series. To support strong and interpretable ML, features will be engineered using short time windows (e.g., last 3–6 hours), including:

1) **Current snapshot features**
- Latest recorded values at hour t (e.g., HR(t), MAP(t), Lactate(t))

2) **Aggregated window statistics (trend + stability)**
For each selected variable over the last N hours:
- mean, min, max, standard deviation
- trend (value at t minus value at t-N)

3) **Missingness features**
Because missingness is informative in clinical workflows:
- measurement present flag per variable
- count of missing values within the window

4) **Patient context features**
- Age, Gender, ICU unit indicators (Unit1/Unit2), HospAdmTime, ICULOS

### Models (straightforward ML, professor-friendly)
- **Baseline:** Logistic Regression
- **Tree-based:** Random Forest
- **Boosted trees:** XGBoost

### Evaluation metrics (imbalance-aware)
- ROC-AUC
- PR-AUC (important with rare positives)
- Recall, Precision, F1 (clinical relevance emphasizes catching positives while controlling false alarms)

### Interpretability
- Feature importance and SHAP analysis to identify the top contributing physiological signals.

---

## Final Product (Streamlit)
A Streamlit web app that allows a user to:
- Choose a patient and ICU hour (or input features for a snapshot)
- Receive a predicted sepsis risk (probability) and classification (0/1)
- View an explanation of key contributing factors (top features)

This functions as a lightweight, interpretable early-warning prototype.
