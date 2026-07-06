import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

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
    print("----- Step 4 : Split the data for training and testing -----")
    print(border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42, stratify = Y)

    print("Training data :",X_train.shape)
    print("Testing data :",X_test.shape)



def main():

    border = "-"*50

    print(border)
    print("------------ Wine Classifier using KNN -----------")
    print(border)

    MarvelousClassifier("WinePredictor.csv")


if __name__ == "__main__":
    main()