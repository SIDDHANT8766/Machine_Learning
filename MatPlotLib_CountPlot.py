import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    # Catogarical Data
    sns.countplot(x = ["A", "B", "A", "A", "B", "A", "C"])

    plt.show()

if __name__ == "__main__":
    main()