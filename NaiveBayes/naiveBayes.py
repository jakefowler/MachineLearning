import pandas as pd 



def ratiosOfFeatures(data):
    



# Training Naive Bayes
def trainingNaiveBayes(csvFileName):
    """
    Function that takes in the file name of a csv file in the same directory and 
    trains off of that data.
    Param: csvFileName -> csv file name. can include path if file isn't located in the same directory.
    """
    # data holds the full csv file in a dataframe
    data = pd.read_csv("test1_1.csv")
    print(data)

    # positive Class is all the rows from the dataframe that have the value 1 for their class
    positiveClass = data.loc[data["Class"] == 1]
    print("Positive counts:\n", positiveClass)

    # positiveClassProb is the probablility that the class is 1 instead of -1 in the main dataframe
    positiveClassProb = len(positiveClass)/len(data)
    print("The probablility that the class is positive is:", positiveClassProb)

    # The probablility that the class is -1
    negativeClassProb = 1 - positiveClassProb
    print("The probablility that the class is negative is:", negativeClassProb)

    # This sums up all the features in the posotive class to use for getting the ratios
    sumOfPositiveCounts = positiveClass.sum()
    print("Sum of positive values in the positive class:\n",sumOfPositiveCounts)

    # These are the ratios for the features being positive in the positive class
    positiveClassPositiveRatios = sumOfPositiveCounts.truediv(len(positiveClass))
    print("Positive class positive features ratios are:\n", positiveClassPositiveRatios)

    # The ratios from the features being negative in the posotive class
    positiveClassNegativeRatios = 1 - positiveClassPositiveRatios
    print("Positive Classes negative feature ratios are:\n", positiveClassNegativeRatios)

    negativeClass = data.loc[data["Class"] == -1] 
    print("Negative class valuses are:\n{} and the type of Negative Count is:\n{}".format(negativeClass, type(negativeClass)))

    negativeClassPositiveRatios = negativeClass.sum().truediv(len(negativeClass))

    print("Negative class positive ratios are:\n",negativeClassPositiveRatios)

    negativeClassNegativeRatios = 1 - negativeClassPositiveRatios

    print("The negative class negative ratios are:\n", negativeClassNegativeRatios)

# Testing Naive Bayes on the same data set

#prediction = pd.DataFrame()


#data.loc[:,"Prediciton"] = prediction[0]