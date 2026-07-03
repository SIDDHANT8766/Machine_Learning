from sklearn.datasets import load_iris

def main():
    print("Iris Classificatio Case Study")

    Dataset = load_iris()

    # Metadata  of Dataset
    print("Independent variables are : ")
    print(Dataset.feature_names)
    print("Length of Independent variable is : ",len(Dataset.feature_names))

    print("Dependent variable are : ")
    print(Dataset.target_names)
    print("Length of Dependent variable is : ",len(Dataset.target_names))


if __name__ == "__main__":
    main()