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