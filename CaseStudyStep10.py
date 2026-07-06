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

##############################################################################################################
#                            Step 1 : Load the Dataset
##############################################################################################################

print(Border)
print("----------- Step 1 : Load the Dataset ------------")
print(Border)

print()

DatasetPath = "Iris.csv"

df = pd.read_csv(DatasetPath)      # read_csv("cvsname")   read all the data from csv 

print("Dataset gets loaded successfully")
print("Initial entries from dataset are :")
print(df.head())

print()

##############################################################################################################
#                               Step 2 : Data Analysis
##############################################################################################################

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

print()

##############################################################################################################
#                               Step 3 : Deside Independent & Dependent Variable
##############################################################################################################

print()

print(Border)
print(" Step 3 : Deside Independent & Dependent Variable ")
print(Border)

print()

# X : Independen Variable / Feature
# Y : Dependen Variable  / Lable

feature_cols = [
    "SepalLength (Cm)",
    "SepalWidth (Cm)",
    "PetalLength (Cm)",
    "PetalWidth (Cm)"
]

X = df[feature_cols]
Y = df["species"]

print("X shape : ",X.shape)
print("Y shape : ",Y.shape)

print()

##############################################################################################################
#                               Step 4 : Visualisation of Dataset
##############################################################################################################

print()

print(Border)
print("------- Step 4 : Visualisation of Dataset --------")
print(Border)

# Scatter Plot

plt.figure(figsize = (7,5))  # Inbuilt keyword for size of frame 

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["PetalLength (Cm)"], temp["PetalWidth (Cm)"], label = sp)

plt.title("Iris : Petal length vs Petal width")
plt.xlabel("PetalLength (Cm)")
plt.ylabel("PetalWidth (Cm)")

plt.legend()
plt.grid(True)
plt.show()

print()

##############################################################################################################
#                               Step 5 : Split the Dataset For Tarining & Testing
##############################################################################################################

print()

print(Border)
print(" Step 5 : Split the Dataset For Tarining & Testing ")
print(Border)

print()

# Test size = 20%
# Train size = 80%

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.5,
    random_state = 42
)

print("Data spliting activity done : ")

print()

print("X - Independent : ",X.shape) # (150, 4)
print("Y - Dependent : ",Y.shape)   # (150, )

print("X_train : ",X_train.shape)  # (120,4)
print("X_test : ",X_test.shape)    # (30,4)

print("Y_train : ",Y_train.shape)  # (120, )
print("Y_test : ",Y_test.shape)    # (30, )

print()

##############################################################################################################
#                               Step 6 : Build the Model
##############################################################################################################

print()

print(Border)
print("---------- Step 6 : Build the Model --------------")
print(Border)

print()

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion = "gini",
    max_depth = 5,
    random_state = 42

)

print("Model successfully created : ",model)

print()

##############################################################################################################
#                               Step 7 : Train the Model
##############################################################################################################

print()

print(Border)
print("---------- Step 7 : Train the Model --------------")
print(Border)

print()

model.fit(X_train, Y_train)

print("Model training completed")

print()

##############################################################################################################
#                               Step 8 : Test the Model (Evaluate)
##############################################################################################################

print()

print(Border)
print("--------- Step 8 : Test the Model (Evaluate) ---------")
print(Border)

print()

Y_pred = model.predict(X_test)

print("Model evaluation (testing) complete")

print(Y_pred.shape)

print("Expected answer : ")
print(Y_test)

print("Predicted answer : ")
print(Y_pred)

print()

##############################################################################################################
#                               Step 9 : Evaluate the Model Performance
##############################################################################################################

print()

print(Border)
print("----- Step 9 : Evaluate the Model Performance ----")
print(Border)

print()

Accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy of model is : ",Accuracy*100)

cm = confusion_matrix(Y_test,Y_pred)
print("Confusion matrix : ")
print(cm)

print()

print("Classification Report : ")
print(classification_report(Y_test,Y_pred))

print()

##############################################################################################################
#                               Step 10 : Plot Confusion Matrix
##############################################################################################################

print()

print(Border)
print("-------- Step 10 : Plot Confusion Matrix -------")
print(Border)

print()

data = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
data.plot()
plt.title("Confusion Matrix of Iris Dataset")
plt.show()