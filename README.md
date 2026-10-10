
# Customer Churn Prediction & Retention Intelligence Platform

An end-to-end machine learning project that predicts customer churn, segments customers by risk, explains predictions, and supports retention decisions through an interactive Streamlit application.

## Business Problem

Customer churn reduces recurring revenue and increases the cost of acquiring replacement customers. Businesses need to identify customers who may leave, understand their risk factors, and prioritize retention efforts.

This project transforms customer data into actionable retention intelligence using machine learning, probability calibration, risk segmentation, SHAP explainability, and business-value analysis.

## Project Objectives

- Analyze customer churn patterns and data quality.
- Engineer features for machine learning.
- Compare Logistic Regression, Random Forest, and Gradient Boosting.
- Tune model hyperparameters using cross-validation.
- Calibrate churn probabilities.
- Evaluate business-oriented classification thresholds.
- Segment customers into Low, Medium, High, and Critical risk.
- Explain model predictions with SHAP.
- Generate customer-level retention recommendations.
- Present results in an interactive Streamlit dashboard.

## Dataset

The project uses the Telco Customer Churn dataset.

- **Customers:** 7,043
- **Original columns:** 21
- **Target:** `Churn`
- **Prediction task:** Binary classification

The dataset includes customer tenure, contract type, services, billing information, and churn outcomes.

Place the original CSV at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SHAP
- Streamlit
- Joblib
- Pytest
- Jupyter Notebook

## Machine Learning Workflow

1. Data understanding and quality checks
2. Exploratory data analysis
3. Feature engineering and one-hot encoding
4. Baseline Logistic Regression
5. Model comparison
6. Hyperparameter tuning
7. Probability calibration
8. Business threshold analysis
9. Final evaluation
10. Customer risk segmentation
11. SHAP explainability
12. Business intelligence and retention recommendations

## Model Performance

The previously recorded evaluation results for the calibrated Gradient Boosting workflow were approximately:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.846 |
| PR-AUC | 0.674 |
| Recall | 0.813 |
| F1 Score | 0.626 |
| Brier Score | 0.134 |
| Log Loss | 0.415 |

These values should be verified against the saved outputs before being treated as the final published results. The evaluation split was reused during model development and threshold selection, so an untouched temporal or external test set would provide stronger evidence of generalization.

## Customer Risk Segmentation

Customers are assigned to four risk categories using predicted churn probability:

| Segment | Probability |
|---|---|
| Low | Below 10% |
| Medium | 10% to below 30% |
| High | 30% to below 50% |
| Critical | 50% or higher |

Risk segments help prioritize customer outreach. These probability bands are project-defined decision rules, not universal industry standards.

## Explainability

SHAP is used to identify influential model features and explain individual predictions.

The project includes global feature importance and customer-level risk drivers. These explanations describe model behavior and associations; they do not establish causation.

## Business Value

The project evaluates retention scenarios using explicitly stated assumptions about intervention cost, retention success, and customer value.

These are illustrative assumptions, not measured company savings. Real business deployment would require actual financial data and controlled experiments to measure incremental retention.

## Streamlit Application

The application provides:

- Executive dashboard
- Individual customer churn-risk scoring
- Risk segmentation summary
- High-risk customer review
- Retention action recommendations

The application loads the saved model rather than retraining it on each launch.

## Project Structure

```text
customer-churn-intelligence/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── outputs/
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── model_utils.py
│   └── risk_scoring.py
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_model_utils.py
│   └── test_risk_scoring.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and open its project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the raw dataset and trained model artifacts are available at their expected paths.

## Run the Application

From the project root:

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Run Tests

From the project root:

```bash
python -m pytest tests -v
```

## Limitations and Future Work

- Evaluate on an untouched temporal or external test set.
- Validate calibration on new customer data.
- Replace illustrative retention assumptions with real business economics.
- Investigate uplift modeling and controlled retention experiments.
- Add monitoring for data drift and model performance.
- Explore automated scoring and deployment workflows.

## Author

**Sneh Desai**

Graduate student in Informatics with interests in data science, machine learning, analytics, and business intelligence.
