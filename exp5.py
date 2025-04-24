import pandas as pd
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = "sample_dataset_100.xlsx"  # Ensure this file is in the working directory
df = pd.read_excel(file_path)

# ---- Correlation Test ----
corr_coeff, corr_p_value = stats.pearsonr(df['Feature_X'], df['Feature_Y'])
print(f"Correlation Coefficient: {corr_coeff:.4f}, P-value: {corr_p_value:.4f}")

if corr_p_value < 0.05:
    print("Conclusion: There is a statistically significant correlation between Feature_X and Feature_Y.")
else:
    print("Conclusion: No significant correlation was found between Feature_X and Feature_Y.")

print("\n" + "-"*50 + "\n")

# ---- Z-Test ---- (Using Welch's t-test as a substitute since population variance is unknown)
z_stat, z_p_value = stats.ttest_ind(df['Group1'], df['Group2'], equal_var=False)
print(f"Z-Test Statistic: {z_stat:.4f}, P-value: {z_p_value:.4f}")

if z_p_value < 0.05:
    print("Conclusion: There is a statistically significant difference between Group1 and Group2.")
else:
    print("Conclusion: No significant difference was found between Group1 and Group2.")

print("\n" + "-"*50 + "\n")

# ---- Chi-Square Test ----
category_counts = df['Category'].value_counts()
chi2_stat, chi2_p_value = stats.chisquare(category_counts)
print(f"Chi-Square Statistic: {chi2_stat:.4f}, P-value: {chi2_p_value:.4f}")

if chi2_p_value < 0.05:
    print("Conclusion: The categorical distribution significantly differs from an expected uniform distribution.")
else:
    print("Conclusion: The categorical distribution does not significantly differ from an expected uniform distribution.")