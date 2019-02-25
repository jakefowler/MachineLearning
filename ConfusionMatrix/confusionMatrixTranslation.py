# Confusion Matrix A
#       +   -
#   +   263 25
#   -   8   271

# TP = true positive
# FN = false negative

#       +   -
#   +   TP  FN
#   -   FP  TN

import math

exit = False

# (TP + TN)/(TP + FP + TN + FN)
def accuracy(TP, FP, FN, TN):
    print("Accuracy: ", round((TP + TN)/(TP + FP + TN + FN), 4))

# TP/(TP + FN)
# Only applies for one class
def sensitivityRecall(TP, FN):
    sRVal = (TP / (TP + FN)) 
    print("Sensitivity/Recall: ", round(sRVal, 4))
    return sRVal

# TN/(TN + FP)
def specificity(TN, FP):
    print("Specificity: ", round((TN / (TN + FP)), 4))

# TP/(TP + FP)
def precision(TP, FP):
    prec = (TP / (TP + FP))
    print("Precision: ", round(prec, 4))
    return prec

# (2 * precision * sensitivityRecall)/(precision + sensitivityRecall)
def f1Measure(prec, sR):
    print("F1 Measure: ", round(((2 * prec * sR) / (prec + sR)), 4))

# Matthews Correlation Coefficient
# ((TP * TN) - (FP * FN))/(sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)))
# if any of the additions in the denominator are zero then make them equal 1
def mCC(TP, TN, FP, FN):
    mCCVal = (((TP * TN) - (FP * FN))/(math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))))
    print("MCC: ", round(mCCVal, 4))

print("Confusion Matrix")

while exit == False:
    TP = int(input("Enter the true positive value: "))
    FP = int(input("Enter the false positive value: ")) 
    FN = int(input("Enter the false negative value: ")) 
    TN = int(input("Enter the true negative value: ")) 
    accuracy(TP, FP, FN, TN)
    sR = sensitivityRecall(TP, FN)
    specificity(TN, FP)
    prec = precision(TP, FP)
    f1Measure(prec, sR)
    mCC(TP, TN, FP, FN)
    answer = input("Would you like to enter another confusion matrix? Please enter Y/N.")
    if answer == "N" or answer == "n":
        exit = True
    