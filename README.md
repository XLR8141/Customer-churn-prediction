# Customer Churn Prediction

A machine learning project predicting customer churn on the Telco Customer Churn dataset (7,043 customers), built as a mid-term project for the KPITB EDSIY-KP AI/ML cohort.

## Dataset
Telco Customer Churn dataset — 33 original columns, ~26.5% churn rate (moderately imbalanced binary classification).

## Approach

**1. Exploratory Data Analysis**
- Checked data structure, duplicates (0 found), and missing values
- Fixed `Total Charges` (stored as text with 11 blanks — corrected to numeric, filled with 0 for brand-new customers with 0 tenure)
- Analyzed class balance, numerical feature distributions, outliers (IQR method — none found), and churn rate by key categorical features

**2. Preprocessing**
- Dropped identifier, location, and churn-derived leakage columns (e.g. `CustomerID`, `Churn Score`, `CLTV`, `Churn Reason`) to prevent data leakage
- Standardized categorical values (collapsed "No internet/phone service" into "No")
- Binary-encoded Yes/No and gender columns; one-hot encoded multi-category columns
- Stratified 80/20 train-test split; scaled numerical features with `StandardScaler`

**3. Modeling**
Trained and compared two algorithms, each with two imbalance-handling strategies:
- **Logistic Regression** — class-weighted, and SMOTE-resampled (with `GridSearchCV` hyperparameter tuning on the SMOTE variant)
- **XGBoost** — plain, and SMOTE-resampled
- **Dummy baseline** (always predicts "No Churn") included to confirm models are learning real signal, not exploiting class imbalance

## Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression (balanced) | 0.51 | 0.78 | 0.62 |
| **Logistic Regression (SMOTE)** | **0.54** | **0.75** | **0.63** |
| Logistic Regression (SMOTE, tuned) | 0.54 | 0.75 | 0.62 |
| XGBoost (plain) | 0.61 | 0.54 | 0.57 |
| XGBoost + SMOTE | 0.56 | 0.66 | 0.61 |
| Dummy baseline | — | 0.00 | — (73% accuracy, uninformative) |

**Final model: Logistic Regression + SMOTE** — selected for the best F1-score (0.63). F1 was prioritized over accuracy/recall alone because both false negatives (losing a paying customer) and false positives (wasted retention spend) carry real cost.

**Key finding:** more complex models (XGBoost) did not outperform a properly tuned, imbalance-corrected Logistic Regression — suggesting churn's relationship with key features here is largely linear.

**Top churn drivers:** month-to-month contracts, short tenure, fiber-optic internet, and higher monthly charges.

## Limitations
- XGBoost was evaluated with lighter hyperparameter tuning than Logistic Regression, so this comparison may understate XGBoost's ceiling.

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib, seaborn

## How to Run
Open `notebook.ipynb` in Jupyter and run cells in order. Requires the Telco Customer Churn dataset (`Telco_customer_churn.xlsx`) in the same directory.
