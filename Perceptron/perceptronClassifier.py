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
    learningRate = 0.01
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
    avrError = np.mean(error)
    print(avrError)

    print(weights)
    for val in weights:
        val = val - learningRate * avrError
 #  shuffled_data = shuffle(data)
 #  print(shuffled_data)
 #  shuffled_data.reset_index().iloc[:, 1:]
 #  print(shuffled_data)

 # need to bound the error. Average all the error rates and normalize
 # number of epochs
 # each epoch needs an accuracy score
 # an epoch is one run through the training data and then test on that
 # average all the accuracies of the folds
 # weights carry over for each epic on each fold
 # 


def main():
    fileName = "diabetes.csv"
    trainPerceptron(fileName)


if __name__ == "__main__":
    main()