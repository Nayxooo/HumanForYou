import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
)
import os

sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ============================
# STEP 1: SETUP OUTPUT FOLDER
# ============================
output_folder = "svm_stats" # Subfolder for storing stats
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

if 'Attrition_1' not in data.columns:
    raise KeyError("Error: Target column 'Attrition_1' not found in the dataset.")

X = data.drop(columns=['Attrition_1'])  
y = data['Attrition_1']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# STEP 4: GRID SEARCH FOR BEST PARAMETERS
# ============================
print("Performing hyperparameter tuning using GridSearchCV...")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(probability=True, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters Found:", grid_search.best_params_)
best_svm_model = grid_search.best_estimator_

# ============================
# STEP 5: CROSS-VALIDATION EVALUATION
# ============================
print("Evaluating model using cross-validation...")
cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean ROC-AUC Score: {np.mean(cv_scores):.4f}")

# ============================
# STEP 6: TRAIN FINAL MODEL
# ============================
print("Training SVM with optimized hyperparameters...")
best_svm_model.fit(X_train, y_train)

y_pred = best_svm_model.predict(X_test)
y_pred_proba = best_svm_model.predict_proba(X_test)[:, 1]

# ============================
# STEP 7: EVALUATE THE MODEL
# ============================
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

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set ROC-AUC Score: {roc_auc:.4f}")

# ============================
# STEP 8: ACCURACY AND LOSS GRAPHS
# ============================
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    best_svm_model.fit(X_cv_train, y_cv_train)
    
    train_acc = best_svm_model.score(X_cv_train, y_cv_train)
    val_acc = best_svm_model.score(X_cv_val, y_cv_val)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    train_loss = 1 - train_acc
    val_loss = 1 - val_acc
    train_losses.append(train_loss)
    val_losses.append(val_loss)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o', label='Training Accuracy')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', label='Validation Accuracy')
plt.title("Model Accuracy Progression")
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Accuracy (0-1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accuracy_progression.png"))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Validation Loss')
plt.title("Model Loss Progression")
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Loss (1 - Accuracy)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "loss_progression.png"))
plt.show()

# ============================
# STEP 9: FEATURE IMPORTANCE GRAPH
# ============================
print("Plotting Top 10 Most Important Features...")

if best_svm_model.kernel == 'linear':
    feature_importances = np.abs(best_svm_model.coef_[0])
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title("Top 10 Most Important Features (SVM - Linear Kernel)")
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "top_10_features.png"))
    plt.show()
else:
    print("Feature importance is not available for non-linear kernels.")

# ============================
# COMPLETION MESSAGE
# ============================
print("Model training, evaluation, and feature analysis complete.")
print(f"Results (graphs, stats, and top features) saved in the folder: {output_folder}")
