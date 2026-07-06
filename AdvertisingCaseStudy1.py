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



def main():
    
    MarvellousAdvertise("Advertising.csv")

   
if __name__ == "__main__":
    main()