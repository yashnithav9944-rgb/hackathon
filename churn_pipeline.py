"""
=============================================================
  TELCO CUSTOMER CHURN PREDICTION SYSTEM
  Full ML Pipeline: EDA → Preprocessing → Feature Engineering
  → Model Training → Evaluation → Risk Segmentation → CLV
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import pickle

# ─── CONFIG ───────────────────────────────────────────────
DATA_PATH   = "data/cleaned_telco_churn.csv"   # <-- put your CSV here
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

RISK_THRESHOLDS = {"Low": 0.30, "Medium": 0.60, "High": 1.01}
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# ─── 1. DATA LOADING & VALIDATION ─────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    """Load dataset and perform sanity checks."""
    df = pd.read_csv(path)
    print(f"✅  Loaded  {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Convert booleans to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Report missing values
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print("⚠️  Missing values found:")
        print(nulls[nulls > 0])
        df.fillna(df.median(numeric_only=True), inplace=True)
        print("   → Imputed with column median.")
    else:
        print("✅  No missing values.")

    # Class balance
    churn_rate = df['Churn'].mean()
    print(f"📊  Churn rate: {churn_rate:.2%}  "
          f"(Churned={df['Churn'].sum():,}, "
          f"Active={(~df['Churn'].astype(bool)).sum():,})")
    return df


# ─── 2. FEATURE ENGINEERING ───────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful derived features."""
    df = df.copy()

    # Customer Lifetime Value proxy
    df['CLV'] = df['tenure'] * df['MonthlyCharges']

    # Average monthly spend (handles tenure=0 edge case)
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Customer lifecycle flags
    df['IsNewCustomer']      = (df['tenure'] <= 3).astype(int)
    df['IsMatureCustomer']   = (df['tenure'] >= 24).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)

    # Service bundle flags
    df['HasSecurityBundle'] = (
        (df.get('OnlineSecurity_Yes', 0) == 1) &
        (df.get('OnlineBackup_Yes',   0) == 1)
    ).astype(int)

    service_cols = [
        'PhoneService_Yes', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
        'DeviceProtection_Yes', 'TechSupport_Yes',
        'StreamingTV_Yes', 'StreamingMovies_Yes'
    ]
    existing = [c for c in service_cols if c in df.columns]
    df['ServiceCount'] = df[existing].sum(axis=1)

    # Charge ratios
    df['ChargePerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
    df['SpendTrend'] = df['MonthlyCharges'] / (df['AvgMonthlySpend'] + 0.01)

    print(f"✅  Feature engineering: {df.shape[1]} total features")
    return df


# ─── 3. EXPLORATORY DATA ANALYSIS ─────────────────────────
def run_eda(df: pd.DataFrame):
    """Generate EDA plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Telco Churn — EDA Overview", fontsize=16, fontweight='bold')

    # Churn distribution
    ax = axes[0, 0]
    counts = df['Churn'].value_counts()
    ax.bar(['Active', 'Churned'], counts, color=['#2ecc71','#e74c3c'], alpha=0.85)
    ax.set_title("Churn Distribution")
    for i, v in enumerate(counts):
        ax.text(i, v + 50, f'{v:,}\n({v/len(df):.1%})', ha='center')

    # Tenure distribution
    ax = axes[0, 1]
    df[df['Churn']==0]['tenure'].hist(ax=ax, bins=30, alpha=0.6, label='Active', color='#2ecc71')
    df[df['Churn']==1]['tenure'].hist(ax=ax, bins=30, alpha=0.6, label='Churned', color='#e74c3c')
    ax.set_title("Tenure Distribution by Churn")
    ax.legend()

    # Monthly charges
    ax = axes[0, 2]
    df[df['Churn']==0]['MonthlyCharges'].hist(ax=ax, bins=30, alpha=0.6, label='Active', color='#2ecc71')
    df[df['Churn']==1]['MonthlyCharges'].hist(ax=ax, bins=30, alpha=0.6, label='Churned', color='#e74c3c')
    ax.set_title("Monthly Charges by Churn")
    ax.legend()

    # CLV boxplot
    ax = axes[1, 0]
    df.boxplot(column='CLV', by='Churn', ax=ax, notch=True)
    ax.set_title("CLV by Churn Status")
    plt.sca(ax); plt.title("CLV by Churn Status")

    # Correlation heatmap
    ax = axes[1, 1]
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['Churn'].drop('Churn').sort_values()
    top_corr = pd.concat([corr.head(6), corr.tail(6)])
    top_corr.plot(kind='barh', ax=ax, color=['#e74c3c' if x>0 else '#2ecc71' for x in top_corr])
    ax.set_title("Top Correlations with Churn")
    ax.axvline(0, color='black', lw=0.8)

    # Service count
    ax = axes[1, 2]
    service_churn = df.groupby('ServiceCount')['Churn'].mean()
    service_churn.plot(kind='bar', ax=ax, color='#3498db', alpha=0.8)
    ax.set_title("Churn Rate by Service Count")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Number of Services")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  EDA plot saved → outputs/eda_overview.png")


# ─── 4. MODEL TRAINING & EVALUATION ───────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple models; return best + all results."""

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=RANDOM_STATE),
            True   # needs scaled input
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
            ),
            False
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=4,
                subsample=0.8, random_state=RANDOM_STATE
            ),
            False
        ),
    }

    results = {}
    trained = {}

    print("\n" + "="*58)
    print("  MODEL TRAINING & EVALUATION")
    print("="*58)

    for name, (model, scaled) in models.items():
        Xtr = X_tr_s if scaled else X_train.values
        Xte = X_te_s if scaled else X_test.values

        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred  = model.predict(Xte)

        # Cross-validation
        cv = cross_val_score(model, Xtr, y_train,
                             cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                             scoring='roc_auc')

        rpt = classification_report(y_test, pred, output_dict=True)
        results[name] = {
            'roc_auc':      round(roc_auc_score(y_test, proba), 4),
            'avg_precision':round(average_precision_score(y_test, proba), 4),
            'cv_auc_mean':  round(cv.mean(), 4),
            'cv_auc_std':   round(cv.std(), 4),
            'precision':    round(rpt['1']['precision'], 4),
            'recall':       round(rpt['1']['recall'], 4),
            'f1':           round(rpt['1']['f1-score'], 4),
            'accuracy':     round(rpt['accuracy'], 4),
        }
        trained[name] = (model, scaled)

        print(f"\n  {name}")
        print(f"    ROC-AUC  : {results[name]['roc_auc']:.4f}")
        print(f"    CV AUC   : {results[name]['cv_auc_mean']:.4f} ± {results[name]['cv_auc_std']:.4f}")
        print(f"    F1(churn): {results[name]['f1']:.4f}")
        print(f"    Precision: {results[name]['precision']:.4f}  Recall: {results[name]['recall']:.4f}")

    # Best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n🏆  Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    return results, trained, scaler, best_name


# ─── 5. EVALUATION PLOTS ──────────────────────────────────
def plot_evaluation(X_test, y_test, trained, scaler, feature_names):
    """ROC curves, confusion matrix, feature importance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight='bold')

    colors = {'Logistic Regression': '#e74c3c',
              'Random Forest':       '#2ecc71',
              'Gradient Boosting':   '#3498db'}

    # ROC curves
    ax = axes[0]
    for name, (model, scaled) in trained.items():
        Xte = scaler.transform(X_test) if scaled else X_test.values
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], lw=2)
    ax.plot([0,1],[0,1],'--', color='gray', lw=1)
    ax.set_title("ROC Curves"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Confusion matrix (best model = GB)
    ax = axes[1]
    gb_model, _ = trained['Gradient Boosting']
    pred = gb_model.predict(X_test.values)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Active','Churned'], yticklabels=['Active','Churned'])
    ax.set_title("Confusion Matrix (GB)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # Feature importance
    ax = axes[2]
    rf_model, _ = trained['Random Forest']
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fi.plot(kind='barh', ax=ax, color='#3498db', alpha=0.85)
    ax.invert_yaxis()
    ax.set_title("Top 15 Features (Random Forest)")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Evaluation plot saved → outputs/model_evaluation.png")

    # Save feature importances
    fi_full = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi_full.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", header=['importance'])


# ─── 6. RISK SEGMENTATION ─────────────────────────────────
def segment_risk(df_full, X_test, y_test, best_model, scaled, scaler):
    """Assign Low / Medium / High risk; compute CLV priority."""
    Xte = scaler.transform(X_test) if scaled else X_test.values
    proba = best_model.predict_proba(Xte)[:, 1]

    seg = X_test.copy()
    seg['churn_probability'] = proba
    seg['actual_churn']      = y_test.values
    seg['risk_category'] = pd.cut(
        proba, bins=[0, 0.30, 0.60, 1.01],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    seg['CLV_priority'] = seg['CLV']
    seg['retention_urgency'] = seg['churn_probability'] * seg['CLV']

    # Retention suggestions
    def suggest(row):
        suggestions = []
        if row.get('IsNewCustomer', 0): suggestions.append("Onboarding incentive")
        if row.get('MonthlyCharges', 0) > 80: suggestions.append("Loyalty discount")
        if row.get('InternetService_Fiber optic', 0): suggestions.append("Fiber upgrade offer")
        if row.get('Contract_Two year', 0) == 0 and row.get('Contract_One year', 0) == 0:
            suggestions.append("Switch to annual plan offer")
        if row.get('TechSupport_Yes', 0) == 0: suggestions.append("Free TechSupport trial")
        if not suggestions: suggestions.append("Loyalty rewards program")
        return " | ".join(suggestions[:2])

    seg['retention_suggestion'] = seg.apply(suggest, axis=1)

    # Summary stats
    print("\n📊  Risk Segmentation Summary:")
    print(seg['risk_category'].value_counts().to_string())
    print(f"\n💰  Avg CLV by Risk:")
    print(seg.groupby('risk_category')['CLV'].mean().round(2).to_string())

    # Save high-risk customers
    high_risk = (seg[seg['risk_category'] == 'High Risk']
                 .sort_values('retention_urgency', ascending=False)
                 [['churn_probability', 'risk_category', 'CLV',
                   'tenure', 'MonthlyCharges', 'retention_suggestion']])
    high_risk.to_csv(f"{OUTPUT_DIR}/high_risk_customers.csv")
    print(f"\n✅  High-risk customers saved → outputs/high_risk_customers.csv")

    seg.to_csv(f"{OUTPUT_DIR}/all_customers_scored.csv", index=False)
    return seg


# ─── 7. LEARNING CURVE ────────────────────────────────────
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(1), 'o-', label='Train AUC', color='#2ecc71')
    plt.fill_between(train_sizes,
                     train_scores.mean(1) - train_scores.std(1),
                     train_scores.mean(1) + train_scores.std(1), alpha=0.15, color='#2ecc71')
    plt.plot(train_sizes, val_scores.mean(1), 'o-', label='Val AUC', color='#e74c3c')
    plt.fill_between(train_sizes,
                     val_scores.mean(1) - val_scores.std(1),
                     val_scores.mean(1) + val_scores.std(1), alpha=0.15, color='#e74c3c')
    plt.title("Learning Curve — Gradient Boosting")
    plt.xlabel("Training Size"); plt.ylabel("ROC-AUC")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Learning curve saved → outputs/learning_curve.png")


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("\n" + "="*58)
    print("   TELCO CHURN PREDICTION — FULL PIPELINE")
    print("="*58)

    # 1. Load
    df = load_and_validate(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. EDA
    run_eda(df)

    # 4. Split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n✅  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 5. Train & evaluate
    results, trained, scaler, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, X.columns.tolist()
    )

    # 6. Evaluation plots
    plot_evaluation(X_test, y_test, trained, scaler, X.columns.tolist())

    # 7. Learning curve (generalization check)
    gb, _ = trained['Gradient Boosting']
    plot_learning_curve(gb, X.values, y.values)

    # 8. Risk segmentation
    best_model, best_scaled = trained[best_name]
    seg = segment_risk(df, X_test, y_test, best_model, best_scaled, scaler)

    # 9. Save model
    with open(f"{MODEL_DIR}/best_model.pkl", 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler,
                     'features': X.columns.tolist(), 'model_name': best_name}, f)
    print(f"✅  Model saved → models/best_model.pkl")

    # 10. Save dashboard JSON
    fi = pd.Series(trained['Random Forest'][0].feature_importances_,
                   index=X.columns).sort_values(ascending=False).head(15)
    dashboard_data = {
        'model_results':      results,
        'feature_importance': fi.to_dict(),
        'risk_distribution':  seg['risk_category'].value_counts().to_dict(),
        'clv_by_risk':        seg.groupby('risk_category')['CLV'].mean().round(2).to_dict(),
        'churn_rate':         float(y.mean()),
        'total_customers':    int(len(df)),
        'high_risk_count':    int((seg['risk_category'] == 'High Risk').sum()),
    }
    with open(f"{OUTPUT_DIR}/dashboard_data.json", 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print("✅  Dashboard data saved → outputs/dashboard_data.json")

    print("\n" + "="*58)
    print("  ✅  PIPELINE COMPLETE")
    print("="*58)


if __name__ == "__main__":
    main()
