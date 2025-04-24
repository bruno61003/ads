import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from CSV file
df = pd.read_csv("wine.csv")

# Bar Plot
plt.figure(figsize=(10, 5))
df.groupby("Wine")["Alcohol"].mean().plot(kind='bar', color='skyblue')
plt.title("Average Alcohol Content per Wine Type")
plt.xlabel("Wine Type")
plt.ylabel("Alcohol Content")
plt.show()

# Pie Chart
plt.figure(figsize=(6, 6))
df["Wine"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99"])
plt.title("Distribution of Wine Types")
plt.ylabel("")
plt.show()

# Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Alcohol"])
plt.title("Boxplot of Alcohol Content")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
df["Alcohol"].plot.hist(bins=10, alpha=0.7, color='blue')
plt.title("Histogram of Alcohol Content")
plt.xlabel("Alcohol")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Alcohol"], y=df["Proline"], hue=df["Wine"], palette='coolwarm')
plt.title("Scatter Plot of Alcohol vs Proline")
plt.xlabel("Alcohol")
plt.ylabel("Proline")
plt.show()

# Heat Map
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap of Correlation Matrix")
plt.show()

# Line Chart
plt.figure(figsize=(8, 5))
df["Alcohol"].plot(kind='line', marker='o', color='red')
plt.title("Line Chart of Alcohol Content")
plt.xlabel("Index")
plt.ylabel("Alcohol")
plt.show()