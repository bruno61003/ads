import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor

# Load dataset
data = pd.read_csv('wine.csv', skipinitialspace=True)

# Drop unnecessary columns (Unnamed and non-numeric)
data = data.dropna(axis=1, how='all')

# Select numeric columns (exclude 'Wine' as it's categorical)
numeric_data = data.select_dtypes(include=[np.number]).drop(columns=['Wine'])

# ----- Z-Score Outlier Detection -----
z_scores = np.abs(zscore(numeric_data))
threshold = 3  # Consider values above this as outliers
outlier_z = (z_scores > threshold).any(axis=1)

# ----- Local Outlier Factor (LOF) Detection -----
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(numeric_data)
outlier_lof = lof_labels == -1  # Outliers are labeled as -1

# ----- Plot Results -----
plt.figure(figsize=(14, 6))

# 📌 Z-Score Outlier Plot
plt.subplot(1, 2, 1)
plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], c='black', s=10, label="Data points")
plt.scatter(numeric_data.loc[outlier_z, numeric_data.columns[0]], 
            numeric_data.loc[outlier_z, numeric_data.columns[1]], 
            facecolors='none', edgecolors='red', s=200, label="Outlier scores")
plt.title("Outlier Detection using Z-Score")
plt.xlabel(numeric_data.columns[0])
plt.ylabel(numeric_data.columns[1])
plt.legend()

# 📌 Local Outlier Factor (LOF) Plot
plt.subplot(1, 2, 2)
plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], c='black', s=10, label="Data points")
plt.scatter(numeric_data.loc[outlier_lof, numeric_data.columns[0]], 
            numeric_data.loc[outlier_lof, numeric_data.columns[1]], 
            facecolors='none', edgecolors='red', s=200, label="Outlier scores")
plt.title("Outlier Detection using Local Outlier Factor (LOF)")
plt.xlabel(numeric_data.columns[0])
plt.ylabel(numeric_data.columns[1])
plt.legend()

plt.tight_layout()
plt.show()