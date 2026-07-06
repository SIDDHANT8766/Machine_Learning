import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousPredictor():
    # Load the data

    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    print("Value of independent variable : X - ",X)
    print("Value of dependent variable : Y - ",Y)

    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    print("X_mean is : ",mean_X)   # 3.0
    print("Y_mean is : ",mean_Y)   # 3.6

def main():
    MarvellousPredictor()

if "__main__" == __name__:
    main()