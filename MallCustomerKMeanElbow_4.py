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

    #-------------------------------------------
    #--Step 4 : Use Elbow method
    #-------------------------------------------

    WCSS = []

    for i in range(1,11):
        model = KMeans(n_clusters=i, random_state=42, n_init=10)
        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,11), WCSS, marker = 'o')
    plt.xlabel("Number of cluster")
    plt.ylabel("WCSS")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()

    #-------------------------------------------
    #--Step 5 : Train the model And Test also
    #-------------------------------------------

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    print("\nDataset with cluster : \n",df.head(30))


if "__main__" == __name__:
    main()