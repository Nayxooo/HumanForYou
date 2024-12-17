# Required Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn Style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ============================
# STEP 1: SETUP OUTPUT FOLDER
# ============================
output_folder = "stats"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================
# STEP 2: LOAD PROCESSED DATA
# ============================
print("Loading processed dataset...")

data = pd.read_csv("processed_data.csv")

# ============================
# STEP 3: TRAIN-TEST SPLIT
# ============================
print("Splitting data into train and test sets...")

if 'Attrition' in data.columns:
    X = data.drop(columns=['Attrition'])  # Drop target variable
    y = data['Attrition']  # Target column (already encoded)
else:
    raise ValueError("The 'Attrition' column is missing in the dataset!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# STEP 4: HYPERPARAMETER TUNING (Grid Search)
# ============================
print("Performing hyperparameter tuning using GridSearchCV...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Use the best parameters
best_rf_model = grid_search.best_estimator_

# ============================
# STEP 5: TRAIN MODEL WITH BEST PARAMETERS
# ============================
print("Training Random Forest model with optimized hyperparameters...")
best_rf_model.fit(X_train, y_train)

# Predictions
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# ============================
# STEP 6: CUSTOM LOSS FUNCTION
# ============================
print("Calculating custom loss function...")

def custom_loss_function(y_true, y_pred_proba):
    return np.mean((y_true - y_pred_proba) ** 2)  # Mean Squared Error as custom loss

custom_loss = custom_loss_function(y_test, y_pred_proba)
print(f"Custom Loss (MSE): {custom_loss:.4f}")

# ============================
# STEP 6b: PLOT CUSTOM LOSS FUNCTION
# ============================
print("Plotting custom loss function across thresholds...")

thresholds = np.linspace(0, 1, 100)  # Generate thresholds from 0 to 1
custom_losses = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)  # Apply threshold to probabilities
    loss = custom_loss_function(y_test, y_pred_threshold)       # Calculate custom loss
    custom_losses.append(loss)

# Plot the Custom Loss Function
plt.figure(figsize=(10, 6))
plt.plot(thresholds, custom_losses, color='blue', linewidth=2)
plt.title("Custom Loss Function vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Custom Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "custom_loss_plot.png"))
plt.show()

# ============================
# STEP 6c: ACCURACY LEARNING GRAPH
# ============================
print("Plotting accuracy over training and validation phases...")

train_accuracies = []
test_accuracies = []

# Loop over different numbers of trees (n_estimators) to simulate learning
n_estimators_range = range(10, 301, 20)  # From 10 to 300 trees in steps of 20
for n in n_estimators_range:
    temp_model = RandomForestClassifier(
        n_estimators=n,
        max_depth=grid_search.best_params_.get('max_depth', None),
        min_samples_split=grid_search.best_params_.get('min_samples_split', 2),
        min_samples_leaf=grid_search.best_params_.get('min_samples_leaf', 1),
        random_state=42
    )
    temp_model.fit(X_train, y_train)
    
    # Accuracy on training and test sets
    train_acc = accuracy_score(y_train, temp_model.predict(X_train))
    test_acc = accuracy_score(y_test, temp_model.predict(X_test))
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Plot the Accuracy Graph
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accuracies, label="Training Accuracy", color='blue', linewidth=2)
plt.plot(n_estimators_range, test_accuracies, label="Validation Accuracy", color='green', linewidth=2)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accuracy_learning_graph.png"))
plt.show()

# ============================
# STEP 7: EVALUATE THE MODEL
# ============================
print("Evaluating the model...")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test Set ROC-AUC Score: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.show()

# ============================
# STEP 8: FEATURE IMPORTANCE
# ============================
print("Analyzing feature importance...")

feature_importances = best_rf_model.feature_importances_
features = X.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[sorted_idx], y=features[sorted_idx], palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "feature_importance.png"))
plt.show()

# ============================
# COMPLETION MESSAGE
# ============================
print("\nModel training, evaluation, and interpretability completed successfully!")
