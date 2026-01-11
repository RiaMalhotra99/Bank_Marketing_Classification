# Bank_Marketing_Classification

**Introduction**: 

This project focuses on building and comparing multiple machine learning classification models using the Bank Marketing dataset. The objective is to predict whether a bank customer will subscribe to a term deposit based on various personal and campaign-related features.

**Dataset Description**: 

The dataset used in this project is the Bank Marketing dataset from the UCI Machine Learning Repository. It contains information collected during direct marketing campaigns of a Portuguese banking institution.

**Number of records**: 4521
**Number of features**: 16
**Target variable**: Subscription to term deposit (yes / no)

**Dataset source**: 

Link : https://archive.ics.uci.edu/dataset/222/bank+marketing

**Models Implemented**: 

The following classification models were implemented and evaluated:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors
Gaussian Naive Bayes
Random Forest Classifier
XGBoost Classifier

**Evaluation Metrics**: 

Each model was evaluated using the following metrics:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

**Model Performance Comparison** :

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8917 | 0.8905 | 0.5536 | 0.2981 | 0.3875 | 0.3532 |
| Decision Tree | 0.8497 | 0.6766 | 0.3730 | 0.4519 | 0.4087 | 0.3255 |
| KNN | 0.8840  | 0.7415 | 0.4878 | 0.1923| 0.2759 | 0.2547 |
| Naive Bayes | 0.8298 | 0.7899 | 0.3214 | 0.4327 | 0.3689 | 0.2770 |
| Random Forest | 0.8884 | 0.8907 | 0.5306 | 0.2500| 0.3399 | 0.3119 |
| XGBoost | 0.8928 | 0.9021 | 0.5467 |  0.3942 | 0.4581 | 0.4069 |
