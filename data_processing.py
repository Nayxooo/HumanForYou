# Required Libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ============================
# STEP 1: SETUP OUTPUT FOLDER
# ============================
output_folder = "stats"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================
# STEP 2: LOAD DATA
# ============================
print("Loading datasets...")
general_data = pd.read_csv("general_data.csv")
manager_data = pd.read_csv("manager_survey_data.csv")
employee_data = pd.read_csv("employee_survey_data.csv")
in_time = pd.read_csv("in_time.csv")
out_time = pd.read_csv("out_time.csv")

# ============================
# STEP 3: MERGE DATA
# ============================
print("Merging datasets...")

# Ensure all employees in general_data are retained
data = general_data.merge(manager_data, on="EmployeeID", how="left")
data = data.merge(employee_data, on="EmployeeID", how="left")

# ============================
# STEP 4: HANDLE MISSING VALUES
# ============================
print("Handling missing values...")

# Fill missing values for numerical columns with median
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        print(f"Filling missing values in {col} with median...")
        data[col].fillna(data[col].median(), inplace=True)

# Fill missing values for categorical columns with mode
for col in data.select_dtypes(include=[object]).columns:
    if data[col].isnull().sum() > 0:
        print(f"Filling missing values in {col} with mode...")
        data[col].fillna(data[col].mode()[0], inplace=True)

# ============================
# STEP 5: HANDLE OUTLIERS
# ============================
print("Handling outliers...")

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Removing outliers in {col}...")
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Exclude unwanted columns from outlier removal
numerical_cols = data.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col not in ['EmployeeCount', 'EmployeeID', 'StandardHours']]

for col in numerical_cols:
    data = remove_outliers(data, col)

# ============================
# STEP 6: AGGREGATE IN/OUT TIME DATA
# ============================
print("Aggregating in/out time data...")

# Clean up in_time and out_time datasets
in_time_clean = in_time.dropna(axis=1, how='all')
out_time_clean = out_time.dropna(axis=1, how='all')

# Parse datetime and calculate average working hours
for col in in_time_clean.columns[1:]:
    in_time_clean[col] = pd.to_datetime(in_time_clean[col], errors='coerce')
    out_time_clean[col] = pd.to_datetime(out_time_clean[col], errors='coerce')

working_hours = (out_time_clean.iloc[:, 1:] - in_time_clean.iloc[:, 1:]).mean(axis=1).dt.total_seconds() / 3600
working_hours = working_hours.fillna(0)  # Fill missing working hours with 0
data['AvgWorkHours'] = working_hours

# ============================
# STEP 7: ENCODE CATEGORICAL DATA
# ============================
print("Encoding categorical variables...")

label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=[object]).columns
# Exclude unwanted columns from encoding
categorical_cols = [col for col in categorical_cols if col not in ['Gender']]

for col in categorical_cols:
    if data[col].nunique() == 2:
        print(f"Encoding binary column: {col}")
        data[col] = label_encoder.fit_transform(data[col])

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# ============================
# STEP 8: NORMALIZE NUMERICAL DATA
# ============================
print("Normalizing numerical data...")

scaler = MinMaxScaler()
# Exclude unwanted columns from scaling
numerical_cols = data.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col not in ['EmployeeCount', 'EmployeeID', 'StandardHours']]

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Log transform skewed columns
for col in numerical_cols:
    if data[col].skew() > 1:
        print(f"Log-transforming skewed column: {col}")
        data[f'Log_{col}'] = np.log1p(data[col])

# ============================
# STEP 9: DROP EXCLUDED COLUMNS
# ============================
print("Dropping excluded columns...")
excluded_cols = ['EmployeeCount', 'EmployeeID', 'Gender', 'StandardHours']
processed_data = data.drop(columns=excluded_cols, errors='ignore')

# ============================
# STEP 10: EXPORT PROCESSED DATA
# ============================
print("Exporting processed data...")

output_file = "processed_data_final.csv"
processed_data.to_csv(output_file, index=False)

print(f"\nProcessed data saved to '{output_file}'.")
print("Program completed successfully!")
