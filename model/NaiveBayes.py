import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef

data = pd.read_csv("data/bank.csv", sep=';')
data.head()
y=dataset['y']
X=data.drop('y', axis=1)
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

# Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB
naiveb_model = GaussianNB()
naiveb_model.fit(X_train_scaled, y_train)
y_pred_naiveb = naiveb_model.predict(X_test_scaled)
y_prob_naiveb = naiveb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluating the performance
accuracy_naiveb= accuracy_score(y_test, y_pred_naiveb)
precision_naiveb = precision_score(y_test, y_pred_naiveb)
recall_naiveb = recall_score(y_test, y_pred_naiveb)
f1_naiveb = f1_score(y_test, y_pred_naiveb)
auc_naiveb = roc_auc_score(y_test, y_prob_naiveb)
mcc_naiveb = matthews_corrcoef(y_test, y_pred_naiveb)

# Display the performance
print("Naive Bayes (Gaussian) Classifier Performace:")
print(f"Accuracy  : {accuracy_naiveb:.4f}")
print(f"Precision : {precision_naiveb:.4f}")
print(f"Recall    : {recall_naiveb:.4f}")
print(f"F1 Score  : {f1_naiveb:.4f}")
print(f"AUC Score : {auc_naiveb:.4f}")
print(f"MCC Score : {mcc_naiveb:.4f}")
