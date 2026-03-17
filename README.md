# Telco Customer Churn Prediction System

## Project Structure
```
churn_project/
├── data/
│   └── cleaned_telco_churn.csv      ← Place your dataset here
├── models/
│   └── best_model.pkl               ← Auto-generated after training
├── outputs/
│   ├── eda_overview.png
│   ├── model_evaluation.png
│   ├── learning_curve.png
│   ├── feature_importance.csv
│   ├── high_risk_customers.csv
│   ├── all_customers_scored.csv
│   └── dashboard_data.json
├── churn_pipeline.py                ← Main ML pipeline
├── predict_new_customers.py         ← Inference on new data
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Run Pipeline
```bash
python churn_pipeline.py
```

## Predict New Customers
```bash
python predict_new_customers.py --input data/new_customers.csv
```

## Data Requirements
Your CSV should have these columns (already preprocessed/one-hot encoded):
- SeniorCitizen, tenure, MonthlyCharges, TotalCharges
- gender_Male, Partner_Yes, Dependents_Yes
- PhoneService_Yes, MultipleLines_*, InternetService_*
- OnlineSecurity_*, OnlineBackup_*, DeviceProtection_*
- TechSupport_*, StreamingTV_*, StreamingMovies_*
- Contract_One year, Contract_Two year
- PaperlessBilling_Yes, PaymentMethod_*
- Churn (target: 0/1)

## Model Performance (on provided dataset)
| Model               | ROC-AUC | CV-AUC ± std      | F1 (Churn) |
|---------------------|---------|-------------------|------------|
| Logistic Regression | 0.8473  | 0.8484 ± 0.0053   | 0.619      |
| Gradient Boosting   | 0.8396  | 0.8412 ± 0.0036   | 0.568      |
| Random Forest       | 0.8231  | 0.8299 ± 0.0079   | 0.541      |

## Risk Categories
- **Low Risk**    : Churn probability < 30%
- **Medium Risk** : Churn probability 30–60%
- **High Risk**   : Churn probability > 60%
