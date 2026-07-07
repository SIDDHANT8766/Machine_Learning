from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier #####
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

###################################################
# Step 1 : Load Dataset
###################################################

data = load_breast_cancer()

X = data.data
Y = data.target

print("\nShape of X : ",X.shape)
print("\nShape of Y : ",Y.shape)

###################################################
# Step 2 : Split Dataset
###################################################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

###################################################
# Step 3 : Create Base Models
###################################################

model_lr = LogisticRegression(max_iter = 5000)
model_dt = DecisionTreeClassifier(random_state = 42)
model_knn = KNeighborsClassifier(n_neighbors = 5)

###################################################
# Step 4 : Train Base Models
###################################################

model_lr.fit(X_train, Y_train)
model_dt.fit(X_train, Y_train)
model_knn.fit(X_train, Y_train)

###################################################
# Step 5 : Calculate Indivisual accuracy
###################################################

pred_lr = model_lr.predict(X_test)
pred_dt = model_dt.predict(X_test)
pred_knn = model_knn.predict(X_test)

acc_lr = accuracy_score(pred_lr, Y_test)
acc_dt = accuracy_score(pred_dt, Y_test)
acc_knn = accuracy_score(pred_knn, Y_test)

print("\nIndivisual model accuracy : ")

print("Accuracy of LR : ",acc_lr)
print("Accuracy of DT : ",acc_dt)
print("Accuracy of KNN : ",acc_knn)

###################################################
# Step 6 : Calculate Invisual accuracy
###################################################