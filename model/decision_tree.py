import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef

dataset = pd.read_csv("data/bank.csv", sep=';')
dataset.head()
y=dataset['y']
X=dataset.drop('y', axis=1)
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

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dctr_model = DecisionTreeClassifier(random_state=42)
dctr_model.fit(X_train_scaled, y_train)
y_pred_dctr = dctr_model.predict(X_test_scaled)
y_prob_dctr = dctr_model.predict_proba(X_test_scaled)[:, 1]

accuracy_dctr = accuracy_score(y_test, y_pred_dctr)
precision_dctr = precision_score(y_test, y_pred_dctr)
recall_dctr = recall_score(y_test, y_pred_dctr)
f1_dctr = f1_score(y_test, y_pred_dctr)
auc_dctr = roc_auc_score(y_test, y_prob_dctr)
mcc_dctr = matthews_corrcoef(y_test, y_pred_dt)

print("Decision Tree Classifier Performance:")
print(f"Accuracy  : {accuracy_dctr:.4f}")
print(f"Precision : {precision_dctr:.4f}")
print(f"Recall    : {recall_dctr:.4f}")
print(f"F1 Score  : {f1_dctr:.4f}")
print(f"AUC Score : {auc_dctr:.4f}")
print(f"MCC Score : {mcc_dctr:.4f}")
