import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import pprint as pp


def trainPerceptron(csvFileName):
    """
    Function that takes in a csv file name and trains the perceptron on the data.
    :param: csvFileName -> file name a directory if not in the same directory
    """
    data = pd.read_csv(csvFileName)
    print(data)
    print(len(data.columns))
    scaler = MinMaxScaler().fit(data.iloc[:, :-1])
    #print(scaler.mean_)
    print(scaler.scale_)
    normalizedData = scaler.transform(data.iloc[:, :-1])
    print(normalizedData)
    weights = np.array(np.random.randn(len(data.columns)-1,1))
    print(weights)
    trained = np.dot(normalizedData,weights)
    print(trained)
    output = []
    for val in trained:
        if val > 0:
            output.append(1)
        else:
            output.append(-1)
    print(output)
    target = data.iloc[:, -1]
    error = output - target
    print(error)
 #  shuffled_data = shuffle(data)
 #  print(shuffled_data)
 #  shuffled_data.reset_index().iloc[:, 1:]
 #  print(shuffled_data)


def main():
    fileName = "diabetes.csv"
    trainPerceptron(fileName)


if __name__ == "__main__":
    main()