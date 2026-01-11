import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef

bankdata = pd.read_csv("data/bank.csv", sep=';')
bankdata.head()
y=bankdata['y']
X=bankdata.drop('y', axis=1)
y=y.map({'no': 0, 'yes': 1})

# Identify the categorical columns and numerical columns
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensemble Model - XGBoost
from xgboost import XGBClassifier
xgboost_model = XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=42)
xgboost_model.fit(X_train_scaled, y_train)

y_pred_xgboost = xgboost_model.predict(X_test_scaled)
y_prob_xgboost = xgboost_model.predict_proba(X_test_scaled)[:, 1]

accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
precision_xgboost = precision_score(y_test, y_pred_xgboost)
recall_xgboost = recall_score(y_test, y_pred_xgboost)
f1_xgboost = f1_score(y_test, y_pred_xgboost)
auc_xgboost = roc_auc_score(y_test, y_prob_xgboost)
mcc_xgboost = matthews_corrcoef(y_test, y_pred_xgboost)

print("XGBoost Classifier Performace:")
print(f"Accuracy  : {accuracy_xgboost:.4f}")
print(f"Precision : {precision_xgboost:.4f}")
print(f"Recall    : {recall_xgboost:.4f}")
print(f"F1 Score  : {f1_xgboost:.4f}")
print(f"AUC Score : {auc_xgboost:.4f}")
print(f"MCC Score : {mcc_xgboost:.4f}")
