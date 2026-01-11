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

**Observations on the Performance of each Model** :

 Model | Observation |
|------|-------------|
| Logistic Regression | This model showed strong overall accuracy and AUC, but the recall value was low, meaning many actual positive cases were not correctly identified. |
| Decision Tree | This model provided fairly balanced precision and recall, but its lower AUC suggests that it does not generalize as well as ensemble-based methods. |
| KNN | It achieved good accuracy, but it struggled to correctly detect positive instances, resulting in low recall and F1 score despite using scaled features. |
| Naive Bayes | This model was able to capture more positive cases compared to some models, but its precision was lower, indicating a higher number of false positives. |
| Random Forest | It performed better than a single decision tree and achieved high accuracy and AUC, although recall for the minority class was still limited. |
| XGBoost | It produced the most consistent results across metrics, with the highest AUC, F1 score, and MCC, making it the best-performing model in this comparison. |
