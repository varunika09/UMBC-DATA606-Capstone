# 1. Title and Author

## Project Title
**Goal-Based Portfolio Recommendation System with Probabilistic Simulation and Risk-Aware Allocation (Streamlit App)**

Prepared for **UMBC Data Science Master Degree Capstone** by **Dr Chaojie (Jay) Wang**

## Author Name
**Varunika Bussa**

## Links
- GitHub Repo: *https://github.com/varunika09/UMBC-DATA606-Capstone.git*  
- LinkedIn: *www.linkedin.com/in/varunika-bussa-99a837233*  
- PowerPoint Presentation:  
- YouTube Video:

---

# 2. Background

## What is it about?
This project builds a **goal-based portfolio recommendation system** that helps users plan investments for real-life goals such as:
- **Down payment** (short-to-medium term)
- **Emergency fund** (short term, low risk)
- **Retirement** (long term, growth oriented)

Instead of recommending “stocks to buy,” the system recommends a **portfolio allocation** across a small set of **ETFs** (Exchange-Traded Funds) and shows:

- **Probability of reaching the user’s goal** by the chosen date  
- **Risk metrics** such as worst-case outcomes and maximum drawdowns  
- **Actionable suggestions** to improve success probability (e.g., increase monthly contribution, extend timeline, choose a safer allocation)

A Streamlit web app is the final product. It supports **semi-dynamic data updates** (using a “Refresh Market Data” button) so recommendations can update over time.

---

## What are ETFs?
An **ETF (Exchange-Traded Fund)** is a single investment that contains a **basket of many assets**.  
For example, a broad U.S. stock ETF may hold hundreds/thousands of companies. ETFs are widely used because they:
- provide **diversification**
- reduce single-company risk compared to picking individual stocks
- are liquid and publicly traded like a stock

In this project, ETFs represent different asset classes (e.g., stocks, bonds, gold) to build diversified portfolios.

---

## Why does it matter?
Most people struggle with these real planning questions:
- “Is my goal achievable with my current savings and monthly contributions?”
- “How risky is my plan? What is the worst-case outcome?”
- “What should I change to increase my chance of success?”

Typical portfolio tools focus on “best returns,” but real users need **goal success probability** and **risk-under-uncertainty**.  
This capstone demonstrates practical data science skills used in decision-support systems:
- probabilistic simulation
- optimization under constraints
- scenario analysis
- explainability and user-centric design

---

## Research Questions
1. **Goal Success Prediction:** Given a user’s goal amount, timeline, starting amount, and monthly contributions, what is the **probability of achieving the goal** under different ETF allocations?
2. **Risk-Aware Recommendation:** Which portfolio allocation maximizes goal success probability while meeting user risk constraints (e.g., drawdown tolerance)?
3. **Sensitivity & What-if Analysis:** How do probability of success and risk change under scenario changes such as:
   - reduced contributions
   - shorter/longer horizon
   - market downturn stress tests
   - inflation adjustments (optional)?
4. **Explainability:** Can the system provide clear, plain-English explanations for:
   - why a portfolio is recommended
   - what trade-offs exist between risk and probability
   - what actions improve success probability?

---

# 3. Data

## Data Sources
This project uses **public, free financial time-series data**:

### A) Historical ETF Market Data (Primary)
- Source: **Yahoo Finance** (retrieved programmatically via Python; example: `yfinance`)
- Data pulled: **Adjusted Close** prices (and Open/High/Low/Close/Volume)

### B) Macro-Economic Data (for scenarios)
- Source: **FRED (Federal Reserve Economic Data)**  
- Example series:
  - CPI (inflation) for converting goal into real dollars
  - interest rates / unemployment for scenario narratives

---

## Data Size
Approximate size depends on tickers and time range, but is small and manageable:

- ETFs used: **4–6 tickers** (diversified asset universe)
- History: typically **10–20 years** of daily prices per ETF
- Size on disk: usually **< 50 MB** total as CSV

---

## Data Shape
Two main datasets will be created:

### 1) Raw Price Table (per ETF)
- Rows: ~2,500–5,000 trading days per ETF (10–20 years)
- Columns: typically 6–7 (Date + OHLCV + Adjusted Close)

### 2) Combined Portfolio Modeling Table (derived)
After merging ETFs by Date and computing returns:
- Rows: number of trading days (or months if resampled monthly)
- Columns: one return column per ETF + engineered risk features

---

## Time Period
Time-bound dataset based on selected window, for example:
- **2010-01-01 to present** (depending on ETF availability)
A “semi-dynamic refresh” option will allow updates to the most recent date.

---

## What does each row represent?
Depending on processing stage:

- In the **price dataset**: one row represents **one trading day** for an ETF.
- In the **returns dataset**: one row represents **one time step** (daily or monthly) of returns used for simulation.

---

## Data Dictionary

### A) Yahoo Finance Price Data (typical columns)
| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| Date | date | Trading date | YYYY-MM-DD |
| Open | float | Opening price for the day | positive real |
| High | float | Highest price for the day | positive real |
| Low | float | Lowest price for the day | positive real |
| Close | float | Closing price for the day | positive real |
| Adj Close | float | Closing price adjusted for splits/dividends | positive real |
| Volume | int | Number of shares traded | non-negative integer |

### B) Derived Returns / Features (computed in notebook)
| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| <TICKER>_ret | float | Period return for ETF (daily or monthly) | real (can be negative) |
| rolling_vol | float | Rolling volatility estimate (e.g., 30-day or 12-month) | non-negative real |
| rolling_corr_* | float | Rolling correlations between assets | [-1, 1] |
| portfolio_value | float | Simulated portfolio value over time | non-negative real |
| drawdown | float | % drop from peak portfolio value | [0, 1] |
| max_drawdown | float | Maximum drawdown over simulation path | [0, 1] |

### C) User Input Variables (captured in Streamlit)
| Variable | Data Type | Definition | Potential Values |
|---|---|---|---|
| goal_type | categorical | Type of user goal | {Emergency, DownPayment, Retirement} |
| target_amount | float | Goal amount in dollars | positive real |
| horizon_months | int | Investment duration in months | positive integer |
| start_amount | float | Starting principal | non-negative real |
| monthly_contribution | float | Monthly deposit amount | non-negative real |
| risk_tolerance | categorical | Risk preference | {Conservative, Balanced, Aggressive} |
| max_drawdown_pct | float| Maximum loss the user is willing to tolerate during the investment period | 0–100 (%)|


---

## What is the target/label in the ML model?
This project is primarily a **simulation + optimization** decision-support system, but we define clear targets for evaluation and modeling:

### Primary Target (Binary)
- **goal_success**: whether the final simulated portfolio value meets or exceeds the target amount at the end of the horizon  
  - Values: {0 = not achieved, 1 = achieved}

### Secondary Targets (Continuous / Risk outcomes)
- **ending_wealth**: final portfolio value at horizon
- **max_drawdown**: worst peak-to-trough loss during the horizon
- **success_probability**: proportion of simulations where goal_success = 1

---

## Which variables may be selected as features/predictors?
Features come from both market data and user inputs:

### Market-derived features
- historical returns per ETF
- rolling volatility per ETF
- rolling correlations among ETFs
- scenario modifiers (optional: inflation-adjusted returns using CPI)

### User-input features
- goal type
- target amount
- horizon
- starting amount
- monthly contribution
- risk tolerance / drawdown limit

These features influence:
- portfolio constraints (e.g., limit equity allocation for conservative short-term goals)
- optimization objective (maximize success probability subject to risk constraints)

---

## Notes on Dataset Collection and Storage (per project requirements)
- Datasets will be fetched and stored in: `data/`
  - Example outputs: `data/prices.csv`, `data/returns.csv`, `data/meta.json`
- Exploratory data analysis and feature engineering will be performed in: `notebooks/`
  - Example notebook: `notebooks/01_data_exploration.ipynb`
- The Streamlit application will use the stored snapshot by default and allow optional updates via a refresh button.
