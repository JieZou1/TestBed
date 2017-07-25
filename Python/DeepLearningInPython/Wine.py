import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


class Wine:
    # class variables, shared by all instances
    white_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    red_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    def __init__(self):
        # instance variables, unique to each instance
        self.white = pd.DataFrame()
        self.red = pd.DataFrame()
        self.wines = pd.DataFrame()

    def read_data(self):
        # Read in white wine data
        self.white = pd.read_csv(self.white_url, sep=';')

        # Read in red wine data
        self.red = pd.read_csv(self.red_url, sep=';')

    def data_explore(self):
        # Print info on white wine
        print(self.white.info())

        # Print info on red wine
        print(self.red.info())

        # First rows of `red`
        print(self.red.head())

        # Last rows of `white`
        print(self.white.tail())

        # Take a sample of 5 rows of `red`
        print(self.red.sample(5))

        # Describe `white`
        print(self.white.describe())

        # Double check for null values in `red`
        print(pd.isnull(self.red))

    def data_viz(self):
        fig, ax = plt.subplots(1, 2)

        ax[0].hist(self.red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
        ax[1].hist(self.white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, hspace=0.05, wspace=0.5)
        ax[0].set_ylim([0, 1000])
        ax[0].set_xlabel("Alcohol in % Vol")
        ax[0].set_ylabel("Frequency")
        ax[1].set_ylim([0, 1000])
        ax[1].set_xlabel("Alcohol in % Vol")
        ax[1].set_ylabel("Frequency")
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.suptitle("Distribution of Alcohol in % Vol")

        # sulphate vs. quality

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].scatter(self.red['quality'], self.red["sulphates"], color="red")
        ax[1].scatter(self.white['quality'], self.white['sulphates'], color="white", edgecolors="black", lw=0.5)

        ax[0].set_title("Red Wine")
        ax[1].set_title("White Wine")
        ax[0].set_xlabel("Quality")
        ax[1].set_xlabel("Quality")
        ax[0].set_ylabel("Sulphates")
        ax[1].set_ylabel("Sulphates")
        ax[0].set_xlim([0, 10])
        ax[1].set_xlim([0, 10])
        ax[0].set_ylim([0, 2.5])
        ax[1].set_ylim([0, 2.5])
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle("Wine Quality by Amount of Sulphates")

        # Acidity vs. Alcohol
        np.random.seed(570)

        red_labels = np.unique(self.red['quality'])
        white_labels = np.unique(self.white['quality'])

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        red_colors = np.random.rand(6, 4)
        white_colors = np.append(red_colors, np.random.rand(1, 4), axis=0)

        for i in range(len(red_colors)):
            red_y = self.red['alcohol'][self.red.quality == red_labels[i]]
            red_x = self.red['volatile acidity'][self.red.quality == red_labels[i]]
            ax[0].scatter(red_x, red_y, c=red_colors[i])
        for i in range(len(white_colors)):
            white_y = self.white['alcohol'][self.white.quality == white_labels[i]]
            white_x = self.white['volatile acidity'][self.white.quality == white_labels[i]]
            ax[1].scatter(white_x, white_y, c=white_colors[i])

        ax[0].set_title("Red Wine")
        ax[1].set_title("White Wine")
        ax[0].set_xlim([0, 1.7])
        ax[1].set_xlim([0, 1.7])
        ax[0].set_ylim([5, 15.5])
        ax[1].set_ylim([5, 15.5])
        ax[0].set_xlabel("Volatile Acidity")
        ax[0].set_ylabel("Alcohol")
        ax[1].set_xlabel("Volatile Acidity")
        ax[1].set_ylabel("Alcohol")
        ax[0].legend(red_labels, loc='best', bbox_to_anchor=(1.3, 1))
        ax[1].legend(white_labels, loc='best', bbox_to_anchor=(1.3, 1))
        fig.suptitle("Alcohol - Volatile Acidity")
        fig.subplots_adjust(top=0.85, wspace=0.7)

        plt.show()

    def data_preprocess(self):
        # Add `type` column to `red` with value 1
        self.red['type'] = 1

        # Add `type` column to `white` with value 0
        self.white['type'] = 0

        # Append `white` to `red`
        self.wines = self.red.append(self.white, ignore_index=True)

        corr = self.wines.corr()
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        sns.plt.show()

        pass

    def train_test_wine_type(self):
        # Specify the data
        X = self.wines.ix[:, 0:11]

        # Specify the target labels and flatten the array
        y = np.ravel(self.wines.type)

        # Split the data up in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Define the scaler
        scaler = StandardScaler().fit(X_train)

        # Scale the train set
        X_train = scaler.transform(X_train)

        # Scale the test set
        X_test = scaler.transform(X_test)

        # Initialize the constructor
        model = Sequential()

        # Add an input layer
        model.add(Dense(12, activation='relu', input_shape=(11,)))

        # Add one hidden layer
        model.add(Dense(8, activation='relu'))

        # Add an output layer
        model.add(Dense(1, activation='sigmoid'))

        print(model.output_shape)
        model.summary()
        print(model.get_config())
        print(model.get_weights())

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

        y_pred = model.predict(X_test)

        score = model.evaluate(X_test, y_test, verbose=1)

        print(score)

        # Confusion matrix
        y_pred = (y_pred >= 0.5).astype(int)
        confusion_matrix(y_test, y_pred)

        # Precision
        precision_score(y_test, y_pred)

        # Recall
        recall_score(y_test, y_pred)

        # F1 score
        f1_score(y_test, y_pred)

        # Cohen's kappa
        cohen_kappa_score(y_test, y_pred)

        pass

    def train_test_wine_quality(self):
        # Isolate target labels
        y = self.wines.quality

        # Isolate data
        X = self.wines.drop('quality', axis=1)

        # Scale the data with `StandardScaler`
        X = StandardScaler().fit_transform(X)

        seed = 7
        np.random.seed(seed)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train, test in kfold.split(X, y):
            model = Sequential()
            model.add(Dense(64, input_dim=12, activation='relu'))
            model.add(Dense(1))
            model.summary()
            model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
            model.fit(X[train], y[train], epochs=10, verbose=1)

            mse_value, mae_value = model.evaluate(X[test], y[test], verbose=0)
            print(mse_value)
            print(mae_value)

        pass


if __name__ == "__main__":
    wine = Wine()
    wine.read_data()
    wine.data_explore()
    wine.data_viz()
    wine.data_preprocess()
    # wine.train_test_wine_type()
    wine.train_test_wine_quality()
    pass
