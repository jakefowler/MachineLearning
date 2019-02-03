import pandas as pd 

data = pd.read_csv("test1_1.csv")
print(data)

# Training Naive Bayes

positiveCounts = data.loc[data["Class"] == 1]

positiveClassProb = len(positiveCounts)/len(data)

print("The probablility that the class is positive is:", positiveClassProb)

negativeClassProb = 1 - positiveClassProb

print("The probablility that the class is negative is:", negativeClassProb)

print("Positive counts:\n", positiveCounts)

sumOfPositiveCounts = positiveCounts.sum()

print("Sum of positive counts:\n",sumOfPositiveCounts)

negativeCounts = data.loc[data["Class"] == -1] 

print("Negative counts are:\n{} and the type of Negative Counts is:\n{}".format(negativeCounts, type(negativeCounts)))

negativeRatios = negativeCounts.sum().truediv(len(negativeCounts))

print("Negative Ratios are:\n",negativeRatios)

positiveRatios = sumOfPositiveCounts.truediv(len(positiveCounts))

print("Positive Ratios are:\n", positiveRatios)

# Testing Naive Bayes on the same data set

#prediction = pd.DataFrame()


#data.loc[:,"Prediciton"] = prediction[0]