# Import pandas
import pandas as pd

def read_data():
    # Read in white wine data
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

    # Read in red wine data
    red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

if __name__ == "__main__":
    read_data()
    pass
