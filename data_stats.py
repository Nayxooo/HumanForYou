import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import os

# =========================
# STEP 1: LOAD THE DATASET
# =========================

data = pd.read_csv("C:\\Users\\noaar\\OneDrive\\Bureau\\FISE A3\\AI Bloc\\HumanForYou\\HumanForYou-maxime\\processed_data_final.csv")


# =========================
# STEP 2: SELECT RELEVANT COLUMNS
# =========================
# Select only the columns of interest for the analysis
# selected_columns = [
#     'Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
#     'EnvironmentSatisfaction', 'WorkLifeBalance', 'DistanceFromHome'
# ]
# data = data[selected_columns]

# =========================
# STEP 3: CREATE OUTPUT FOLDER
# =========================
output_folder = "analysis_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# =========================
# STEP 5: CORRELATION HEATMAP
# =========================
# plt.figure(figsize=(10, 8))
# corr_matrix = data.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=True, yticklabels=True)
# plt.title("Correlation Heatmap", fontsize=16)
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
# plt.close()
# print("Correlation heatmap saved to 'correlation_heatmap.png'.")

# =========================
# STEP 6: SCATTER MATRIX
# =========================
# numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
# if len(numeric_columns) > 0:
#     scatter_matrix_fig = scatter_matrix(
#         data[numeric_columns],
#         alpha=0.5,
#         figsize=(12, 12),
#         diagonal="hist"
#     )
#     plt.suptitle("Scatter Matrix", fontsize=16)
#     plt.savefig(os.path.join(output_folder, "scatter_matrix.png"))
#     plt.close()
#     print("Scatter matrix saved to 'scatter_matrix.png'.")
# else:
#     print("No numeric columns available for scatter matrix.")

# print("Analysis complete. Check the 'analysis_plots' folder for outputs.")

# ============================
# STEP 3: CREATE PIE CHART
# ============================
# Calculate the percentages
attrition_counts = data['Attrition_1'].value_counts()
attrition_percentages = attrition_counts / len(data) * 100

labels = ['Stayed', 'Left']  # Modify labels if needed
colors = ['#66b3ff', '#ff6666']  # Custom colors for better visualization
explode = (0, 0.1)  # Slightly "explode" the slice for 'Left' (attrition)

plt.figure(figsize=(8, 8))
plt.pie(attrition_percentages, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, explode=explode, shadow=True)
plt.title("Attrition Distribution (%)", fontsize=16)
plt.tight_layout()
plt.savefig("attrition_pie_chart.png")
plt.show()

print("Pie chart saved as 'attrition_pie_chart.png'.")
