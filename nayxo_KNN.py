import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report)
import os

# ============================
# STEP 1: LOAD THE DATA
# ============================
data = pd.read_csv("C:\\Users\\noaar\\OneDrive\\Bureau\\FISE A3\\AI Bloc\\HumanForYou\\HumanForYou-maxime\\processed_data_final.csv")

# ============================
# STEP 2: PREPARE THE DATA
# ============================
if 'Attrition_1' not in data.columns:
    raise ValueError("The target column 'Attrition_1' is missing from the dataset!")

# Features and target
X = data.drop(columns=['Attrition_1'])  # Drop target column
y = data['Attrition_1']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================
# STEP 3: HYPERPARAMETER TUNING FOR KNN
# ============================
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2],  # For Minkowski distance
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50]
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters from GridSearch:", grid_search.best_params_)

# Use the best parameters
knn_model = grid_search.best_estimator_

# ============================
# STEP 4: TRAIN THE K-NEAREST NEIGHBORS CLASSIFIER
# ============================
knn_model.fit(X_train, y_train)

# Predictions
y_pred = knn_model.predict(X_test)
y_pred_proba = knn_model.predict_proba(X_test)[:, 1]

# ============================
# STEP 5: EVALUATE THE MODEL
# ============================
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_roc_curve.png")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ============================
# COMPLETION MESSAGE
# ============================
print("K-Nearest Neighbors model training and evaluation complete. The ROC Curve and Confusion Matrix have been saved.")
