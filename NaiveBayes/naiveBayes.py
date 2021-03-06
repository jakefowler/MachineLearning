import pandas as pd 
import math
import time

def probablilityOfClass(data):
    """
    Function that takes in a dataframe and gets the probablility 
    that the class is 1 or -1
    Param: data -> the full data frame
    Returns two floats for the positive and negative ratios of the class
    """
    # positive Class is all the rows from the dataframe that have the value 1 for their class
    positiveClass = data.loc[data["Class"] == 1]
    
    # positiveClassProb is the probablility that the class is 1 instead of -1 in the main dataframe
    positiveClassProb = len(positiveClass)/len(data)

    # The probablility that the class is -1
    negativeClassProb = 1 - positiveClassProb

    return positiveClassProb, negativeClassProb

def ratiosOfFeatures(data):
    """
    This function takes in a dataframe and gets the ratios of the features.
    Param: data -> The dataframe of one class
    Returns: Two dataframes filled with the positive and negative ratios for the features
    """
    # This sums up all the features in the class to use for getting the ratios
    sumOfPositiveCounts = data.sum()

    # These are the ratios for the features being positive in the class
    positiveFeaturesRatios = sumOfPositiveCounts.truediv(len(data))

    # The ratios from the features being negative in the class
    negativeFeaturesRatios = 1 - positiveFeaturesRatios
    
    return positiveFeaturesRatios, negativeFeaturesRatios

# Training Naive Bayes
def trainingNaiveBayes(csvFileName):
    """
    Function that takes in the file name of a csv file in the same directory and 
    trains off of that data.
    Param: csvFileName -> csv file name. can include path if file isn't located in the same directory.
    """
    # data holds the full csv file in a dataframe
    data = pd.read_csv(csvFileName)
    print(data)

    # positive Class is all the rows from the dataframe that have the value 1 for their class
    positiveClass = data.loc[data["Class"] == 1]

    # negative Class is all the rows from the dataframe that have the value -1 for their class
    negativeClass = data.loc[data["Class"] == -1]

    # get the class probablilities by passing data to probablilityOfClass()
    positiveClassProb, negativeClassProb = probablilityOfClass(data)

    # get the features positive and negative ratios of the positive class
    positiveClassPositiveRatios, positiveClassNegativeRatios = ratiosOfFeatures(positiveClass) 
    # get the features positive and negative ratios of the negative class
    negativeClassPositiveRatios, negativeClassNegativeRatios = ratiosOfFeatures(negativeClass) 

    return (positiveClassProb, negativeClassProb, positiveClassPositiveRatios, positiveClassNegativeRatios,
            negativeClassPositiveRatios, negativeClassNegativeRatios)

def testingNaiveBayes(positiveClassProb, negativeClassProb, 
    positiveClassPositiveRatios, positiveClassNegativeRatios,
    negativeClassPositiveRatios, negativeClassNegativeRatios, testDataFileName):
    """
    Function that takes in the ratios of the features to classes and ratios of classes to each other.
    It the makes a prediction for the data set from the filename that is given.
    """
    data = pd.read_csv(testDataFileName)
    pred = []

    dataAsList = data.values.tolist()
    
    for i in range(len(dataAsList)):
        classPositiveChance = math.log(positiveClassProb) 
        classNegativeChance = math.log(negativeClassProb)
        for j in range(len(dataAsList[i])-1):
            if dataAsList[i][j] is 1:
                classPositiveChance += math.log(0.00001 if positiveClassPositiveRatios[j] == 0 else positiveClassPositiveRatios[j])
                classNegativeChance += math.log(0.00001 if negativeClassPositiveRatios[j] == 0 else negativeClassPositiveRatios[j])
            if dataAsList[i][j] is 0:
                classPositiveChance += math.log(0.00001 if positiveClassNegativeRatios[j] == 0 else positiveClassNegativeRatios[j])
                classNegativeChance += math.log(0.00001 if negativeClassNegativeRatios[j] == 0 else negativeClassNegativeRatios[j])
        if classPositiveChance > classNegativeChance:
            pred.append(1)
        else:
            pred.append(-1)
    return data["Class"].values.tolist(), pred

def calculateAccuracy(desiredValues, predictedValues):
    """
    Function that calculates the accuracy of a correct list of values and a predicted list.
    Param: desiredValues -> correct list of values
           predictedValues -> list of predictions
    Returns: accuracy value of 0 to 1
    """
    correctValues = []
    for i in range(len(desiredValues)):
        if desiredValues[i] == predictedValues[i]:
            correctValues.append(1)
    accuracy = len(correctValues)/len(desiredValues)
    return accuracy

def main():
    fileName = "test1_5.csv"
    start_time = time.time()
    # Get all the variables through training on the data
    (positiveClassProb, negativeClassProb, positiveClassPositiveRatios, 
    positiveClassNegativeRatios, negativeClassPositiveRatios, 
    negativeClassNegativeRatios) = trainingNaiveBayes(fileName)
    print("Training on the data took %s seconds" % (time.time() - start_time))

    start_time = time.time()
    # Pass all the trained variables into the testing function 
    # along with the file name of the testing data
    desiredValues, predictedValues = testingNaiveBayes(positiveClassProb, negativeClassProb, 
    positiveClassPositiveRatios, positiveClassNegativeRatios,
    negativeClassPositiveRatios, negativeClassNegativeRatios, fileName)
    print("Testing the data took %s seconds" % (time.time() - start_time))
    print("The Correct values are:\n", desiredValues)
    print("Predicted values are:\n", predictedValues)

    accuracy = calculateAccuracy(desiredValues, predictedValues)
    print("The accuracy on this data set is: ", accuracy)

if __name__ == "__main__":
    main()