import pandas as pd 

data = pd.read_csv("test_1.csv")
data

#data.loc(:,"Class")

data.loc[data["Class"] == 1]

data.loc[data["Class"] == -1]
len(data.loc[data["Class"] == -1])/len(data)


positiveCounts = data.loc[data["Class"] == 1].sum()

positiveCounts[1]

