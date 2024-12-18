# Required Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
)
from imblearn.over_sampling import SMOTE

# Seaborn Style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ============================
# STEP 1: SETUP OUTPUT FOLDER
# ============================
output_folder = "logistic_regression_stats"  # New subfolder for Logistic Regression outputs
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================
# STEP 2: LOAD THE DATA
# ============================
print("Loading dataset...")
data = pd.read_csv("processed_data_final.csv")

# ============================
# STEP 3: PREPARE THE DATA
# ============================
print("Preparing the dataset...")
X = data.drop(columns=['Attrition_1'])  # Drop target column
y = data['Attrition_1']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# STEP 4: BALANCE THE DATASET
# ============================
print("Balancing the dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ============================
# STEP 5: GRID SEARCH FOR BEST PARAMETERS
# ============================
print("Performing hyperparameter tuning using GridSearchCV...")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # Regularization penalty
    'solver': ['liblinear', 'lbfgs']
}

logistic_model = LogisticRegression(max_iter=500, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_balanced, y_train_balanced)

print("Best Parameters Found:", grid_search.best_params_)
best_logistic_model = grid_search.best_estimator_

# ============================
# STEP 6: TRAIN FINAL MODEL
# ============================
print("Training Logistic Regression model with optimized hyperparameters...")
best_logistic_model.fit(X_train_balanced, y_train_balanced)
y_pred = best_logistic_model.predict(X_test)
y_pred_proba = best_logistic_model.predict_proba(X_test)[:, 1]

# ============================
# STEP 7: EVALUATE THE MODEL
# ============================
# ROC Curve
print("Generating ROC Curve...")
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
plt.savefig(os.path.join(output_folder, "roc_curve.png"))
plt.show()

# Confusion Matrix
print("Generating Confusion Matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test Set Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set ROC-AUC Score: {roc_auc:.4f}")

# ============================
# STEP 8: LOSS AND ACCURACY GRAPHS
# ============================
print("Generating Loss and Accuracy Graphs...")
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for train_idx, val_idx in cv.split(X_train_balanced, y_train_balanced):
    X_cv_train, X_cv_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
    y_cv_train, y_cv_val = y_train_balanced[train_idx], y_train_balanced[val_idx]

    best_logistic_model.fit(X_cv_train, y_cv_train)
    
    # Loss (1 - Accuracy)
    train_losses.append(1 - best_logistic_model.score(X_cv_train, y_cv_train))
    val_losses.append(1 - best_logistic_model.score(X_cv_val, y_cv_val))
    
    # Accuracy
    train_accuracies.append(best_logistic_model.score(X_cv_train, y_cv_train))
    val_accuracies.append(best_logistic_model.score(X_cv_val, y_cv_val))

# Plot Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Validation Loss')
plt.title("Model Loss Progression (Logistic Regression)")
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Loss (1 - Accuracy)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "loss_progression.png"))
plt.show()

# Plot Accuracy Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o', label='Training Accuracy')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', label='Validation Accuracy')
plt.title("Model Accuracy Progression (Logistic Regression)")
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accuracy_progression.png"))
plt.show()

# ============================
# STEP 9: SAVE RESULTS
# ============================
print("Saving predictions...")
data['Predicted_Attrition'] = best_logistic_model.predict(X)
data['Attrition_Probability'] = best_logistic_model.predict_proba(X)[:, 1]
output_file = os.path.join(output_folder, "logistic_regression_results.csv")
data.to_csv(output_file, index=False)
print(f"Results saved to '{output_file}'.")

# ============================
# COMPLETION MESSAGE
# ============================
print("Logistic Regression model training and evaluation complete. Results and graphs are saved in 'logistic_regression_stats' folder.")
