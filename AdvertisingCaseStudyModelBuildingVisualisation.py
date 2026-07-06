import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def MarvellousAdvertise(Datapath):

    Border = "-"*50

    #--------------------------------------
    # Step 1 : Load dataset
    #---------------------------------------

    print(Border)
    print("-------------- Step 1 : Load Dataset -------------")
    print(Border)

    df = pd.read_csv(Datapath)

    print("Few records from the dataset : ")
    print(df.head())

    print()

    #--------------------------------------
    # Step 2 : Remove unvanted columns
    #---------------------------------------

    print(Border)
    print("-------- Step 2 : Remove unvanted columns --------")
    print(Border)

    print("Shape of data before removal : ",df.shape)

    if('Unnamed: 0' in df.columns):
        df.drop(columns = ['Unnamed: 0'],inplace = True)

    print("Shape of data After removal : ",df.shape)

    print(Border)
    print("Clean dataset is : ")
    print(Border)

    print(df.head)

    print()

    #--------------------------------------
    # Step 3 : Check missing values
    #---------------------------------------

    print(Border)
    print("--------- Step 3 : Check missing values ----------")
    print(Border)

    print("Missing value count : \n",df.isnull().sum())

    print()

    #--------------------------------------
    # Step 4 : Display Stastical summary
    #---------------------------------------

    print(Border)
    print("------- Step 4 : Display Stastical summary -------")
    print(Border)

    print(df.describe())

    #--------------------------------------
    # Step 5 : Corelation between columns
    #---------------------------------------

    print(Border)
    print("------ Step 5 : Corelation between columns -------")
    print(Border)

    print("Corelation matrix : ")
    print(df.corr())

    print()

    #--------------------------------------
    # Step 6 : Split the dataset into independent and dependent variable
    #---------------------------------------

    print(Border)
    print("Step 6 : Split dataset into independent and dependent variable")
    print(Border)

    X = df[["TV","radio","newspaper"]]
    Y = df["sales"]

    print("Shape of Independent variable : ",X.shape)
    print("Shape of Dependent variable : ",Y.shape)

    print()

    #--------------------------------------
    # Step 7 : Split the dataset for training and testing
    #---------------------------------------

    print(Border)
    print("Step 7 : Split the dataset for training and testing")
    print(Border)

    X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

    print("X_train Shape : ",X_train.shape)
    print("X_test Shape : ",X_test.shape)
    print("Y_train Shape : ",Y_train.shape)
    print("Y_test Shape : ",Y_test.shape)

    #--------------------------------------
    # Step 8 : Create and Train the model
    #---------------------------------------

    print(Border)
    print("------- Step 8 : Create and Train the model ------")
    print(Border)

    model = LinearRegression()          ##########

    model.fit(X_train,Y_train)

    #--------------------------------------
    # Step 9 : Test the model
    #---------------------------------------

    print(Border)
    print("------------- Step 9 : Test the model ------------")
    print(Border)

    Y_pred = model.predict(X_test)

    #--------------------------------------
    # Step 10 : Evaluate the model
    #---------------------------------------

    print(Border)
    print("----------- Step 10 : Evaluate the model ---------")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Y_pred)

    print("Mean Squared Error : ",MSE)
    print("Root Mean Squared Error : ",RMSE)
    print("R Squared Value : ",R2)

    #--------------------------------------
    # Step 11 : Calculate the model coefficient
    #---------------------------------------

    print(Border)
    print("------ Step 11 : Calculate model coefficient -----")
    print(Border)

    for column, value in zip(X.columns, model.coef_):
        print(f"{column} : {value}")

    print("Intercept : ",model.intercept_)


    #--------------------------------------
    # Step 12 : Compare the actual and the predicted values
    #---------------------------------------

    print(Border)
    print("Step 12 : Compare the actual and the predicted values")
    print(Border)

    Result = pd.DataFrame({
        'Actual sale' : Y_test.values, 
        'Predicted Sale ' : Y_pred
         })
    
    print(Result.head())

    #--------------------------------------
    # Step 13 : Plot actual vs predicted
    #---------------------------------------

    print(Border)
    print("---------- Step 13 : Plot actual vs predicted -------")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Actual sale vs Predicted sale")
    plt.grid(True)
    plt.show()


def main():
    
    MarvellousAdvertise("Advertising.csv")
   
if __name__ == "__main__":
    main()