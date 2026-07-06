import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

def MarvelousClassifier(Datapath):

    border = "-"*50

    #####################################################################

    print(border)
    print("-------- Step 1 : Load the dataset from csv -------")
    print(border)

    df = pd.read_csv(Datapath)

    print("Some entries from dataset : ")
    print(df.head())

    print(border)

    ##################################################################

    print(border)
    print("- Step 2 : Clean the dataset by removing empty row -")
    print(border)

    df.dropna(inplace = True)

    print("Total records :",df.shape[0])
    print("Total columns :",df.shape[1])

    print(border)

    ##################################################################

    print(border)
    print("----- Step 3 : Dependent and independent variable -----")
    print(border)

    X = df.drop(columns = ['Class'])
    Y = df['Class']

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)

    print(border)

    print("Input columns :",X.columns.tolist())
    print("Output column : Class")

    print(border)

    ##################################################################

    print(border)
    print(" Step 4 : Split the data for training and testing ")
    print(border)
                                                                                            #### IMP ####
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42, stratify = Y)

    print(border)

    print("Information of training and testing the data : ")

    print("X_train shape :",X_train.shape)
    print("X_test shape :",X_test.shape)
    print("Y_train shape :",Y_train.shape)
    print("Y_test shape :",Y_test.shape)
    print(border)

    ##################################################################

    print(border)
    print("------------- Step 5 : Feature Scaling ------------ ")
    print(border)

    scalar = StandardScaler()

    #Independet variable scaling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done")

    print(border)

    ##################################################################

                    # hyper parameter tuning (K)
    print(border)
    print("--- Step 6 : Explore the multiple values of K ----")
    print(border)

    accuracy_scores = []
    K_values = range(1,21)

    for k in K_values:      ##### VVIMP #### model object in the loop
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)
    
    print(border)
    print("Accuracy reprt of all K value 1 to 20 : ")

    for value in accuracy_scores:
        print(value)
    
    print(border)

    ##################################################################

    print(border)
    print("--- Step 7 : Plot graph of K vs Accuracy ----")
    print(border)

    plt.figure(figsize=(8,5))
    plt.plot(K_values, accuracy_scores, marker = 'o')
    plt.title("K value vs Accuracy")
    plt.xlabel("Values of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()

    print(border)

    ##################################################################

    print(border)
    print("------- Step 8 : Find best value of K -------")
    print(border)

    best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]

    print("Best values of K is : ",best_k)

    ##################################################################

    print(border)
    print("------- Step 9 : Build final model using best values of K -------")
    print(border)

    final_model = KNeighborsClassifier(n_neighbors = best_k)

    final_model.fit(X_train_scaled,Y_train)
    Y_pred = final_model.predict(X_test_scaled)

    ##################################################################

    print(border)
    print("------- Step 10 : Calculate final accuracy -------")
    print(border)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Final accuracy of model is : ",accuracy*100)

    print(border)

    ##################################################################

    print(border)
    print("------- Step 11 : Display Confusion matrix -------")
    print(border)

    cm = confusion_matrix(Y_test,Y_pred)
    print("Confusion matrix is : \n",cm)

    print(border)

    ##################################################################

    print(border)
    print("------- Step 12 : Display Classification Report -------")
    print(border)

    print("Classification report : ")
    print(classification_report(Y_test,Y_pred))

    print(border)
    

def main():

    border = "-"*50

    print(border)
    print("------------ Wine Classifier using KNN -----------")
    print(border)

    MarvelousClassifier("WinePredictor.csv")


if __name__ == "__main__":
    main()