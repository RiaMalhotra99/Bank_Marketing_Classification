import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef

df = pd.read_csv("data/bank.csv", sep=';')
df.head()
y=df['y']
X=df.drop('y', axis=1)
y=y.map({'no': 0, 'yes': 1})

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensemble Model - Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100,random_state=42)
random_forest_model.fit(X_train_scaled, y_train)
y_pred_random_forest = random_forest_model.predict(X_test_scaled)
y_prob_random_forest = random_forest_model.predict_proba(X_test_scaled)[:, 1]

accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
precision_random_forest = precision_score(y_test, y_pred_random_forest)
recall_random_forest = recall_score(y_test, y_pred_random_forest)
f1_random_forest = f1_score(y_test, y_pred_random_forest)
auc_random_forest = roc_auc_score(y_test, y_prob_random_forest)
mcc_random_forest = matthews_corrcoef(y_test, y_pred_random_forest)

print("Random Forest Classifier Performance:")
print(f"Accuracy  : {accuracy_random_forest:.4f}")
print(f"Precision : {precision_random_forest:.4f}")
print(f"Recall    : {recall_random_forest:.4f}")
print(f"F1 Score  : {f1_random_forest:.4f}")
print(f"AUC Score : {auc_random_forest:.4f}")
print(f"MCC Score : {mcc_random_forest:.4f}")
