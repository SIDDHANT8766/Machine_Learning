import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

###########################################################################

#----------------------------------------------------
#   Function name : DisplayInfo
#   Description : It Display the formated titel
#   Parameters : titel (str)
#   Return :  None
#   Date : 14/3/2026
#   Author : Siddhant Vivek Gadkari
#----------------------------------------------------

def DisplayInfo(titel):

    print("\n"+"="*50)
    print(titel)
    print("="*50)

###########################################################################

#----------------------------------------------------
#   Function name : ShowData
#   Description : It Shows basic information about dataset
#   Parameters : DataSet/DataFrame (df)
#                  df -> pandas dataframe object
#                message
#                  message -> Heading text to display
#   Return :  None
#   Date : 14/3/2026
#   Author : Siddhant Vivek Gadkari
#----------------------------------------------------

def ShowData(df,message):

    DisplayInfo(message)

    print("\nFirst 5 rows of dataset : \n",df.head())

    print("\nShape of dataset : \n",df.shape)

    print("\nColumns names : \n",df.columns.tolist())

    print("\nMissing valus in each columns : \n",df.isnull().sum())


###########################################################################

#----------------------------------------------------
#   Function name : CleanTitanicData
#   Description : It does preprocessing 
#                   It removes unnessasary column
#                   It handales missing value
#                   It Converts text data to numeric formate
#                   It does encoding to categorical columns
#   Parameters : df -> Pandas dataframe
#   Return :  df -> Clean Pandas Dataframe 
#   Date : 14/3/2026
#   Author : Siddhant Vivek Gadkari
#----------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo(" Step 2 : Original Data")

    print(df.head())

    # Remove unnecessary column
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\nColumns tobe dropped : ")
    print(existing_columns)

    # Drop the unwanted column
    df = df.drop(columns = existing_columns)
    DisplayInfo(" Step 2 : Data After removel")
    print(df.head())

    # Handle Age column 
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # coerce -> invalid values gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors = "coerce")

        age_median = df["Age"].median()

        # Repalce missing valuse with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing : \n",df["Age"].head(10))

    # Handle fare column
    if "Fare" in df.columns:
        print("\nFare column before preprocessing : ")
        print(df["Fare"].head(10))

        # coerce -> invalid values gets converted as NaN
        df["Fare"] = pd.to_numeric(df["Fare"], errors = "coerce")

        Fare_median = df["Fare"].median()
        print("\nMedian of Fare column is : ",Fare_median)

        # Repalce missing valuse with median
        df["Fare"] = df["Fare"].fillna(Fare_median)

        print("\nFare column after preprocessing : \n",df["Fare"].head(10))

    # Handle Embarked column
    if "Embarked" in df.columns:
        print("\nEmbarked column before preprocessing : ")
        print(df["Embarked"].head(10))

        # Convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        # Repalce missing valuse with median
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)
        
        # Get moset frequnt values
        Embarked_mode = df["Embarked"].mode()[0]
        print("\nMode of Embarked column : ",Embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(Embarked_mode) 

        print("\nEmbarked column after preprocessing : \n",df["Embarked"].head(10))

    # handle Sex column
    if "Sex" in df.columns:
        print("\nSex column before preprocessing : ")
        print(df["Sex"].head(10))

        # coerce -> invalid values gets converted as NaN
        df["Sex"] = pd.to_numeric(df["Sex"], errors = "coerce")

        print("\nSex column after preprocessing : \n",df["Sex"].head(10))


    DisplayInfo("Data after preprocessing")
    print(df.head())

    print("\nMissing values after preprocessing")
    print(df.isnull().sum())

    # Encode Embarked column
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)

    print("\nData after encoding\n")
    print(df.head())

    print("\nShape od fataste : \n",df.shape)

    # Convert booleans column into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after encoding\n")
    print(df.head())

    return df

###########################################################################

#----------------------------------------------------
#   Function name : MarvellousTitanicLogistic
#   Description : This is main pipeline controller
#                  It loads the dataset, shows raw data
#                   It preproceeess the dataset & train the data
#   Parameters : Datapath of dataset file
#   Return :  None
#   Date : 14/3/2026
#   Author : Siddhant Vivek Gadkari
#----------------------------------------------------

def MarvellousTitanicLogistic(DataPath):
    DisplayInfo(" Step 1 : Loading the Dataset ")
    
    df = pd.read_csv(DataPath)

    ShowData(df,"Initial Datset")

    df = CleanTitanicData(df)
    

###########################################################################

#----------------------------------------------------
#   Function name : main
#   Description : Satring point of the application
#   Parameters : None
#   Return :  None
#   Date : 14/3/2026
#   Author : Siddhant Vivek Gadkari
#----------------------------------------------------

def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if "__main__" == __name__:
    main()