import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ====================================================
# LOAD DATASET
# ====================================================

df = pd.read_csv("patient_data.csv")

print("\n============================")
print("FIRST 5 RECORDS")
print("============================")
print(df.head())

print("\n============================")
print("DATASET INFORMATION")
print("============================")
print(df.info())

print("\n============================")
print("STATISTICAL SUMMARY")
print("============================")
print(df.describe())

# ====================================================
# MISSING VALUES
# ====================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ====================================================
# LABEL ENCODING
# ====================================================

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

# ====================================================
# VISUALIZATION 1
# AGE DISTRIBUTION
# ====================================================

plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ====================================================
# VISUALIZATION 2
# GENDER DISTRIBUTION
# ====================================================

plt.figure(figsize=(6,5))
df["Gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ====================================================
# VISUALIZATION 3
# DIABETES DISTRIBUTION
# ====================================================

plt.figure(figsize=(7,7))
df["Diabetes"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Diabetes Distribution")
plt.ylabel("")
plt.show()

# ====================================================
# VISUALIZATION 4
# BLOOD PRESSURE
# ====================================================

plt.figure(figsize=(8,5))
plt.hist(df["Blood_Pressure"], bins=10)
plt.title("Blood Pressure Distribution")
plt.xlabel("Blood Pressure")
plt.ylabel("Count")
plt.show()

# ====================================================
# VISUALIZATION 5
# CHOLESTEROL DISTRIBUTION
# ====================================================

plt.figure(figsize=(8,5))
plt.hist(df["Cholesterol"], bins=10)
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.show()

# ====================================================
# VISUALIZATION 6
# READMISSION COUNT
# ====================================================

plt.figure(figsize=(6,5))
sns.countplot(x="Readmitted", data=df)
plt.title("Readmission Count")
plt.show()

# ====================================================
# VISUALIZATION 7
# PREVIOUS ADMISSIONS VS READMISSION
# ====================================================

plt.figure(figsize=(8,5))
plt.scatter(
    df["Previous_Admissions"],
    df["Readmitted"]
)
plt.title("Previous Admissions vs Readmission")
plt.xlabel("Previous Admissions")
plt.ylabel("Readmitted")
plt.show()

# ====================================================
# VISUALIZATION 8
# CORRELATION HEATMAP
# ====================================================

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ====================================================
# FEATURES & TARGET
# ====================================================

X = df.drop(
    columns=["Patient_ID", "Readmitted"]
)

y = df["Readmitted"]

# ====================================================
# TRAIN TEST SPLIT
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================================
# LOGISTIC REGRESSION
# ====================================================

print("\n============================")
print("LOGISTIC REGRESSION")
print("============================")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

# ====================================================
# DECISION TREE
# ====================================================

print("\n============================")
print("DECISION TREE")
print("============================")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, dt_pred))

# ====================================================
# RANDOM FOREST
# ====================================================

print("\n============================")
print("RANDOM FOREST")
print("============================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, rf_pred)

precision = precision_score(y_test, rf_pred)

recall = recall_score(y_test, rf_pred)

f1 = f1_score(y_test, rf_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# ====================================================
# CONFUSION MATRIX
# ====================================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ====================================================
# FEATURE IMPORTANCE
# ====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")
plt.show()

# ====================================================
# SAMPLE PREDICTION
# ====================================================

sample_patient = [[
    60,     # Age
    1,      # Gender
    145,    # Blood Pressure
    220,    # Cholesterol
    1,      # Diabetes
    8,      # Stay Days
    3       # Previous Admissions
]]

prediction = rf.predict(sample_patient)

print("\n============================")
print("PATIENT PREDICTION")
print("============================")

if prediction[0] == 1:
    print("Patient likely to be Readmitted")
else:
    print("Patient unlikely to be Readmitted")

print("\nProject Completed Successfully!")