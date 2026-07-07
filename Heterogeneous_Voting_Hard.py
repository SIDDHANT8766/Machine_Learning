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
# Step 5 : Hard voting classification
###################################################

hard_model = VotingClassifier(
                                estimators = [
                                                ('lr',model_lr),
                                                ('dt',model_dt),
                                                ('knn',model_knn)
                                            ],
                                voting = 'hard'
                            )

hard_model.fit(X_train, Y_train)

pred_hard = hard_model.predict(X_test)

acc_hard = accuracy_score(pred_hard, Y_test)

print("\nHard voting accuracy : ",acc_hard)