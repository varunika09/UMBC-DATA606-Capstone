# Early Sepsis Risk Prediction Using ICU Clinical Data

Prepared for UMBC Data Science Master Degree Capstone  
by Dr. Chaojie (Jay) Wang  

**Author:** Your Name  
**GitHub Repository:** [(https://github.com/varunika09/UMBC-DATA606-Capstone.git)]  
**LinkedIn Profile:** [www.linkedin.com/in/varunika-bussa-99a837233]  
**PowerPoint Presentation:** [Add Link]  
**YouTube Demo:** [Add Link]  

---

# 1. Background

## What is this project about?

This project focuses on the early prediction of sepsis in ICU patients using machine learning.

Sepsis is a life-threatening condition caused by the body’s extreme response to infection. If not detected early, it can lead to organ failure and death. Early intervention significantly improves survival rates.

The goal of this project is to build a predictive model that estimates whether a patient is at risk of developing sepsis using routinely collected ICU vital signs and laboratory measurements.

---

## Why does it matter?

- Sepsis is a leading cause of ICU mortality worldwide.
- Early detection reduces mortality and hospital costs.
- Hospitals collect large volumes of patient monitoring data, but automated risk scoring systems are not always implemented.
- A machine learning–based early warning system can support clinical decision-making.

This project demonstrates how data science can directly contribute to healthcare impact.

---

## Research Questions

1. Can ICU vital signs and lab values predict sepsis onset?
2. Which physiological measurements are most predictive of sepsis?
3. Can we build an interpretable model suitable for clinical decision support?
4. How well does the model generalize to unseen patients?

---

# 2. Data

## Data Source

This project uses the **PhysioNet 2019 Sepsis Challenge Dataset**:

https://physionet.org/content/challenge-2019/1.0.0/

This dataset contains ICU time-series clinical measurements collected from hospital systems.

---

## Data Size

- Total rows: **1,552,210**
- Total columns: **44**
- Unique patients: **40,336**
- Sepsis positive rows: **27,916**
- Sepsis rate (rows): ~**1.8%**

The dataset reflects real-world clinical imbalance.

---

## What does each row represent?

Each row represents:

- A single ICU hour
- For a specific Patient_ID
- With measurements recorded during that hour

Each patient has multiple hourly records.

---

## Data Structure

The dataset includes:

### 1. Vital Signs

- HR (Heart rate)
- O2Sat (Oxygen saturation)
- Temp (Temperature)
- SBP (Systolic blood pressure)
- MAP (Mean arterial pressure)
- DBP (Diastolic blood pressure)
- Resp (Respiratory rate)
- EtCO2 (End tidal CO2)

---

### 2. Laboratory Values

Examples include:

- Lactate
- WBC (White blood cell count)
- Creatinine
- Bilirubin
- Platelets
- Glucose
- BUN
- Hemoglobin
- Hematocrit
- pH
- PaCO2
- TroponinI

These biomarkers are clinically relevant for infection and organ dysfunction.

---

### 3. Demographics

- Age
- Gender (0 = Female, 1 = Male)
- Unit1
- Unit2
- HospAdmTime
- ICULOS (ICU length of stay in hours)

---

## Target Variable

- **SepsisLabel**
  - 1 = Sepsis
  - 0 = No sepsis

This is the classification label for the ML model.

---

# Feature Engineering

The original data is time-series (hourly measurements).  
To simplify modeling, the data will be converted into aggregated tabular features.

For each patient-hour, summary statistics will be computed over recent windows (e.g., last 6 hours):

- Mean HR
- Maximum temperature
- Minimum MAP
- Lactate trend
- Change in WBC
- Variability measures
- Missingness indicators

This allows:

- Capturing physiological trends
- Reducing time-series complexity
- Using interpretable ML models
- Maintaining clinical relevance

---

# Data Splitting Strategy

Data will be split by **Patient_ID** into:

- Training set
- Validation set
- Test set

This prevents leakage across time steps from the same patient.

The held-out test set will simulate unseen ICU patients.

---

# 3. Modeling

## Predictive Task

Binary classification:

Predict whether:

- SepsisLabel = 1 (Sepsis risk)
- SepsisLabel = 0 (No sepsis)

---

## Models

Primary models:

- Logistic Regression (baseline)
- Random Forest
- XGBoost (primary model)

Optional comparison:

- LightGBM

Deep learning models are not the primary focus to maintain interpretability and computational efficiency.

---

## Evaluation Metrics

Because the dataset is imbalanced, evaluation will include:

- ROC-AUC
- Precision-Recall AUC
- Recall (Sensitivity)
- F1-score
- Confusion Matrix

Clinical emphasis:

- High recall (avoid missing sepsis cases)
- Controlled false positive rate

---

# Clinical Interpretation

Sepsis is clinically associated with:

- Elevated heart rate
- Low blood pressure
- Elevated lactate
- Abnormal white blood cell count
- Fever or hypothermia
- Organ dysfunction markers

Feature importance analysis will identify which variables most influence predictions.

---

# Final Product (Streamlit Application)

The final deliverable will be a Streamlit web application that:

1. Uses the held-out test dataset as a simulated ICU patient database.
2. Allows the user to select:
   - Patient_ID
   - ICU hour
3. Automatically retrieves that patient’s recent clinical data.
4. Outputs:
   - Predicted sepsis risk probability
   - Binary classification (0 or 1)
5. Displays top contributing features for interpretability.

This application functions as a lightweight early-warning prototype for ICU monitoring.

---

# Expected Contribution

This project demonstrates:

- Handling large-scale healthcare time-series data
- Transforming time-series into structured ML-ready features
- Building interpretable clinical risk models
- Deploying predictive healthcare systems using Streamlit

The system bridges machine learning and clinical decision support in a meaningful and impactful way.
