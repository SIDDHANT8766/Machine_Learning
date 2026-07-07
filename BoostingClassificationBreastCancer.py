import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#########################################
# Step 1 : Load the Dataset
#########################################

df = pd.read_csv("breast_cancer.csv")

print("Shape of Dataset : \n",df.shape)
print("First 5 elements of Dataset : \n",df.head())


#########################################
# Step 2 : Saperate Feature and Lable of Dataset
#########################################

X = df.drop("target",axis = 1) 
Y = df["target"]

#########################################
# Step 3 : Split Dataset for training and testing
# ########################################

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.2, random_state=42)


#########################################
# Step 4 : Create Boosting model (Adaboost)
#########################################

boost_model = AdaBoostClassifier(
                                    n_estimators = 50,
                                    learning_rate = 1.0,
                                    random_state = 42
                                )

#########################################
# Step 5 : Train the boosting model
#########################################

boost_model.fit(X_train,Y_train)


#########################################
# Step 6 : Test the Boosting model
#########################################

Y_pred = boost_model.predict(X_test)

#########################################
# Step 8 : Evaluate Boosting model
#########################################

print("\nBagging Accuracy : \n",accuracy_score(Y_test,Y_pred))
print("\nComfusion matrix : \n",confusion_matrix(Y_test,Y_pred))