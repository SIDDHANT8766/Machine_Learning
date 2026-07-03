import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*50

###############################################################################
#   Step 1 : Load the Dataset
###############################################################################

print(Border)
print("----------- Step 1 : Load the Dataset ------------")
print(Border)

print()

DatasetPath = "Iris.csv"

df = pd.read_csv(DatasetPath)      # read_csv ()

print("Dataset gets loaded successfully")
print("Initial entries from dataset are :")
print(df.head())

print()

###############################################################################
#   Step 2 : Data Analysis
###############################################################################

print()

print(Border)
print("------------ Step 2 : Data Analysis -------------")
print(Border)

print()

print("Shape of Dataset : ",df.shape)
print("Column names : ",list(df.columns))

print()

print("Missing values (Pre column) :")
print(df.isnull().sum())

print()

print("Class Destribution (Species Count) :")
print(df["species"].value_counts())

print()

print("Statistical Report of Dataset :")
print(df.describe())