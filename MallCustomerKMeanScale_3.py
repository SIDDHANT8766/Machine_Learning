import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    #-------------------------------------------
    #--Step 1 : Load the dataset
    #-------------------------------------------

    print("Step 1 : Load the dataset")
    df = pd.read_csv("Mall_Customers.csv")

    print("\nFirst few records : \n",df.head())
    print("\nShape of dataset : \n",df.shape)
    print("\nMissing of dataset : \n",df.isnull().sum())

    #-------------------------------------------
    #--Step 2 : Select features (Independent)
    #-------------------------------------------

    print("Step 2 : Select features (Independent)\n")

    X = df[["AnnualIncome","SpendingScore"]]  # Imp

    print("Selected feature : ",X.head())
    print("\nShape of selected feature : \n",X.shape)

    #-------------------------------------------
    #--Step 3 : Scale the data
    #-------------------------------------------

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X) # Imp

    print("\nData after scaling : \n",X_scaled[:5])

if "__main__" == __name__:
    main()