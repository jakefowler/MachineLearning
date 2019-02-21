import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pprint as pp
import matplotlib.pylab as plt 


def trainPerceptron(csvFileName):
    """
    Function that takes in a csv file name and trains the perceptron on the data.
    :param: csvFileName -> file name a directory if not in the same directory
    """
    epochAccuracies = []
    totalAccuracies = []
    learningRate = 0.001
    data = pd.read_csv(csvFileName)
    scaler = MinMaxScaler().fit(data.iloc[:, :-1])
    normalizedData = scaler.transform(data.iloc[:, :-1])
    # normalizedData = data.iloc[:, :-1] # unnormalized
    bias = np.ones((768,1), dtype=np.float64)
    normalizedData = np.append(normalizedData, bias, axis=1)
    weights = np.array(np.random.randn(len(data.columns),1))

    print("------WeightsBefore------")
    print(weights)

    target = data.iloc[:, -1].values
    target = target.reshape(768,1)

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    for train, test in skf.split(normalizedData, target):
        for i in range(1000):
            trained = np.dot(normalizedData,weights)
            output = np.where(trained>0,1,-1)
            epochAccuracies.append(calculateAccuracy(target, output))
            error = output - target
            weights -= learningRate*np.dot(np.transpose(normalizedData), error)
            i += 1
        totalAccuracies.append(np.mean(epochAccuracies))
        print("------NewWeights------")
        print(weights)
        print("The average accuracy over the fold is", np.mean(epochAccuracies))

        plt.plot(epochAccuracies, "bo")
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Epochs")
        plt.title("Perceptron at learning rate 0.001")
        plt.show()
        weights = np.array(np.random.randn(len(data.columns),1))
        epochAccuracies = []
    print("Average accuracy for this perceptron is", np.mean(totalAccuracies))
    print("The learning rate was set to", learningRate)

def calculateAccuracy(correct, prediction):
    return accuracy_score(correct, prediction)


def main():
    fileName = "diabetes.csv"
    trainPerceptron(fileName)


if __name__ == "__main__":
    main()