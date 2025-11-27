# ----------------------------------------------
# Importing Required Libraries
# ----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----------------------------------------------
# 1. Load Dataset
# ----------------------------------------------
# Reading CSV file into a pandas DataFrame
df = pd.read_csv('c:\\Users\\Ayushi\\OneDrive\\Documents\\Expanded_data_with_more_features.csv')

# Displaying first 5 rows
print(df.head())


# ----------------------------------------------
# 2. Basic Dataset Summary
# ----------------------------------------------

# Statistical summary of numerical columns
print(df.describe())

# Checking data types and memory usage
print(df.info())

# Checking missing values in each column
print(df.isnull().sum())


# ----------------------------------------------
# 3. Data Cleaning
# ----------------------------------------------

# Dropping an unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# Checking dataset after cleaning
print(df.head())


# ----------------------------------------------
# 4. Data Visualization
# ----------------------------------------------

# ---- 4.1 Gender Distribution ----
plt.figure(figsize=(5,5))
plt.title("Gender Distribution")
kd = sns.countplot(data=df, x="Gender")
kd.bar_label(kd.containers[0])  # Adding bar count labels
plt.show()


# ---- 4.2 Average Scores by Parent Education ----
# Calculating average scores for each education level
gb = df.groupby("ParentEduc").agg({
    "MathScore": "mean",
    "ReadingScore": "mean",
    "WritingScore": "mean"
})
print(gb)

plt.figure(figsize=(5,5))
plt.title("Average Scores by Parent Education Level")
sns.heatmap(gb, annot=True)  # Annotated heatmap
plt.show()


# ---- 4.3 Boxplot for Math Score ----
plt.figure(figsize=(5,5))
plt.title("Outlier of Math Score")
sns.boxplot(data=df, x="MathScore")
plt.show()
