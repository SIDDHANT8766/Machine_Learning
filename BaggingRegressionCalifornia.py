import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

#########################################
# Step 1 : Load the Dataset
#########################################

df = pd.read_csv("california_housing.csv")

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
# Step 4 : Create Base model
#########################################

base_model = DecisionTreeRegressor(random_state = 42)

#########################################
# Step 5 : Create Bagging model
#########################################

bagging_model = BaggingRegressor(
                                    estimator = base_model,
                                    n_estimators = 10,
                                    random_state = 42,
                                )


#########################################
# Step 6 : Train the Bagging model
#########################################

bagging_model.fit(X_train,Y_train)

#########################################
# Step 7 : Test the Bagging model
#########################################

Y_pred = bagging_model.predict(X_test)

#########################################
# Step 8 : Evaluate Bagging model
#########################################

print("\n Mean Squared error : \n",mean_squared_error(Y_test,Y_pred))
print("\nR Square : \n",r2_score(Y_test,Y_pred))