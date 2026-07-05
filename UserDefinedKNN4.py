#   point    X     Y   lable  
#     A      1     2     R
#     B      2     3     R
#     C      3     1     B
#     D      5     6     B

# Predict(3,3)  -> ?

import numpy as np
import math

def EucDistance(P1, P2):
    Ans = math.sqrt((P1['X'] - P2['X']) ** 2 + (P1['Y'] - P2['Y']) ** 2)

    return Ans

def MarvellousKNeighborsClassifier():

    Border = "-"*50

    data = [
                {'point' : 'A', 'X' : 1, 'Y' : 2, 'label' : 'Red'},
                {'point' : 'B', 'X' : 2, 'Y' : 3, 'label' : 'Red'},
                {'point' : 'C', 'X' : 3, 'Y' : 1, 'label' : 'Blue'},
                {'point' : 'D', 'X' : 5, 'Y' : 6, 'label' : 'Blue'}
            ]
    
    print(Border)
    print("Marvellous UserDefined KNN")
    print(Border)

    print()

    print(Border)
    print("Training Dataset")
    print(Border)

    for i in data:
        print(i)

    print(Border)

    new_point = {'X' : 3, 'Y' : 3}

    # Calculate all distaces

    for d in data:
        d['distance'] = EucDistance(d, new_point)

    print(Border)

    print("Calculated distanxce are : ")
    
    for d in data:
        print(d)

    print(Border)

    print()

    sorted_data = sorted(data, key = lambda item : item['distance'])

    print()

    print(Border)
    print("Sorted data is : ")

    for d in sorted_data:
        print(d)

    print(Border)

def main():
    MarvellousKNeighborsClassifier()

if __name__ == "__main__":
    main()