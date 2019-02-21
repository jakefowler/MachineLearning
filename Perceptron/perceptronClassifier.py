import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
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
    output = np.where(trained>0,1,-1)
    #output = np.array(shape=(768,1))
    # output = []
    #print(output)
    #for val in trained:
    #    if val > 0:
    #        output.append(1)
    #    else:
    #        output.append(-1)
    print("------Output------")
    
    print(output)
    target = data.iloc[:, -1].values
    target = target.reshape(768,1)
    #data = data.values
    #target = data.values
    print("------Target------")
    print(target)

    print("Accuracy = ", calculateAccuracy(target, output))

    error = output - target
    print("------Error------")
    print(error)
    avrError = np.mean(error)
    print("------AverageError------")
    print(avrError)

    print("------Weights------")
    print(weights)

    print("------NormalizedDataBefore------")
    #print(normalizedData)
    #normalizedData = np.transpose(normalizedData)
    print("------NormalizedDataAfter------")
    print(normalizedData)
    print("------TransposedError------")
    print(np.transpose(error))
    print("------WeightsBefore------")
    print(weights)
    #print("Shape of weights", weights.shape())
    #print("Shape of normalizedData", normalizedData.shape())
    #print("Shape of error ", error.shape())
    weights -= learningRate*np.dot(np.transpose(normalizedData), error)
    print("------NewWeights------")
    print(weights)

   # for row in data:
     #   val = val - learningRate * avrError
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

def calculateAccuracy(correct, prediction):
    return accuracy_score(correct, prediction)


def main():
    fileName = "diabetes.csv"
    trainPerceptron(fileName)


if __name__ == "__main__":
    main()