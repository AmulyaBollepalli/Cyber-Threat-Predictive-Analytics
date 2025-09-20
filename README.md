# Cyber Threat Predictive Analytics for Improving Network Security

## Problem Statement
Cyber threats are a growing concern for organizations. This Django-based web application predicts potential cyber threats using machine learning models, helping organizations proactively manage network security risks.

## Datasets
- `data/train.csv` — Training dataset
- `data/test.csv` — Test dataset

## Tools & Technologies
- **Django 4.2**
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)

## Approach
1. Data preprocessing and cleaning
2. Feature engineering and selection
3. Train models: Logistic Regression, Decision Tree, Random Forest, XGBoost, and optionally SVM
4. Evaluate models (accuracy chart and table in `outputs/`)
5. Deploy Django web app to detect cyber threat risk

**Note:** SVM training is computationally expensive and may take several hours. Sample metrics for SVM are provided in `outputs/`.

## How to Run
1. Create and activate a virtual environment (recommended)
2.Install dependencies
3.Run the Django server
4.Open your browser at http://127.0.0.1:8000/ to access the app

##Key Findings

XGBoost provides the highest accuracy among the algorithms used.

The predictive model can help organizations identify potential cyber threats proactively.
