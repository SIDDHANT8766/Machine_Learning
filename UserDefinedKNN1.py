#   point    X     Y   lable  
#     A      1     2     R
#     B      2     3     R
#     C      3     1     B
#     D      5     6     B

# Predict(3,3)  -> ?

def MarvellousKNeighborsClassifier():

    Border = "-"*40

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

def main():
    MarvellousKNeighborsClassifier()

if __name__ == "__main__":
    main()