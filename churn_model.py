import pandas as pd

# Load dataset
path = r"C:\Users\india\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(path)
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#encode categorial data
df = pd.get_dummies(df)
# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
df.drop_duplicates(inplace=True)#check for duplicates
# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
y_prob = model.predict_proba(X_test)[:,1]
def risk_category(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

risk = [risk_category(p) for p in y_prob]
import pandas as pd

importance = pd.Series(model.coef_[0], index=X.columns)
print(importance.sort_values(ascending=False))
new_df = pd.read_csv("cleaned_telco_churn.csv")
print(new_df.head())