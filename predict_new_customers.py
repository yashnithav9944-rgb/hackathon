"""
predict_new_customers.py
────────────────────────
Load the saved model and score new customers from a CSV.
Usage:  python predict_new_customers.py --input new_customers.csv
"""
import argparse
import pickle
import pandas as pd
import numpy as np


def load_model(model_path: str = "models/best_model.pkl"):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle['features'], bundle['model_name']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must mirror the features used during training."""
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df['CLV']             = df['tenure'] * df['MonthlyCharges']
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer']   = (df['tenure'] <= 3).astype(int)
    df['IsMatureCustomer']   = (df['tenure'] >= 24).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)

    df['HasSecurityBundle'] = (
        (df.get('OnlineSecurity_Yes', pd.Series(0, index=df.index)) == 1) &
        (df.get('OnlineBackup_Yes',   pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    service_cols = ['PhoneService_Yes','OnlineSecurity_Yes','OnlineBackup_Yes',
                    'DeviceProtection_Yes','TechSupport_Yes','StreamingTV_Yes','StreamingMovies_Yes']
    existing = [c for c in service_cols if c in df.columns]
    df['ServiceCount']     = df[existing].sum(axis=1)
    df['ChargePerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
    df['SpendTrend']       = df['MonthlyCharges'] / (df['AvgMonthlySpend'] + 0.01)
    return df


def predict(input_csv: str, output_csv: str = "outputs/predictions.csv"):
    model, scaler, features, model_name = load_model()
    print(f"Loaded: {model_name}")

    df = pd.read_csv(input_csv)
    df = engineer_features(df)

    # Align columns
    for col in features:
        if col not in df.columns:
            df[col] = 0
    X = df[features]

    # Some models need scaled input
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = model.predict_proba(scaler.transform(X))[:, 1]

    df['churn_probability'] = proba
    df['risk_category']     = pd.cut(proba, bins=[0,0.30,0.60,1.01],
                                     labels=['Low Risk','Medium Risk','High Risk'])
    df['churn_prediction']  = (proba >= 0.5).astype(int)

    df.to_csv(output_csv, index=False)
    print(f"Saved predictions → {output_csv}")
    print(df['risk_category'].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='data/new_customers.csv')
    parser.add_argument('--output', default='outputs/predictions.csv')
    args = parser.parse_args()
    predict(args.input, args.output)
