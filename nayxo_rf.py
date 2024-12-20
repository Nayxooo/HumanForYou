import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
# STEP 3: HYPERPARAMETER TUNING FOR RANDOM FOREST
# ============================
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10], # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],   # Minimum samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
    'bootstrap': [True, False]       # Whether bootstrap samples are used when building trees
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters from GridSearch:", grid_search.best_params_)

# Use the best parameters
rf_model = grid_search.best_estimator_

# ============================
# STEP 4: TRAIN THE RANDOM FOREST
# ============================
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

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
plt.savefig("rf_roc_curve.png")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ============================
# COMPLETION MESSAGE
# ============================
print("Random Forest model training and evaluation complete. The ROC Curve and Confusion Matrix have been saved.")
