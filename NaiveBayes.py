"""

Nick Smith
CS 445 - Machine Learning
Homework 4
Naive Bayes Spambase Classifier
Due: Thursday February 21, 2019

"""

# LIBRARIES
import pandas as pd
import numpy as np
import math

# ------------------------ #
#          PART 1          #
# ------------------------ #

"""GET DATA INTO PROPER FORMAT FOR EXPERIMENT"""

# DATASET WITH SHUFFLING
data = pd.read_csv("spambase.csv", header=None)
data = data.sample(frac=1).reset_index(drop=True)


# SPLIT SPAM AND NOT SPAM
data_spam = data.loc[data[57] == 1]
data_not = data.loc[data[57] == 0]

# HALF SPAM/ HALF NOT SPAM
# 4601 - total
# 2300 per test/training set
# 920 spam, 1380 not spam

# Training Data
temp1 = [data_spam[0:920], data_not[0:1380]]
training_data = pd.concat(temp1)
training_data = training_data.reset_index(drop=True)

# Split data for spam vs not spam
data_train_spam = training_data.loc[training_data[57] == 1]
data_train_not = training_data.loc[training_data[57] == 0]

# Training data matrix
train_spam_temp = data_train_spam.as_matrix().astype(np.float)
train_not_temp = data_train_not.as_matrix().astype(np.float)

# Spam and Non-Spam mean/std Dev for training
spam_mean_train = np.mean(train_spam_temp[:,0:57], axis=0)
spam_std_train = np.std(train_spam_temp[:,0:57], axis=0)
not_mean_train = np.mean(train_not_temp[:,0:57], axis=0)
not_std_train = np.std(train_not_temp[:,0:57], axis=0)


# Test Data
temp2 = [data_spam[921:1840], data_not[1381:2760]]
test_data = pd.concat(temp2)
test_data = test_data.reset_index(drop=True)

# Test Data Matrix
X_train = training_data.as_matrix().astype(np.float)
X_test = test_data.as_matrix().astype(np.float)
X_t_features = X_test[:,:57].copy()
X_t_classifier = X_test[:,57]


# ------------------------ #
#          PART 2          #
# ------------------------ #

"""CREATE PROBABILISTIC MODEL"""
"""
THIS IS THE PRIOR PROBABILITY OF SPAM
VS NOT SPAM. 
"""

spam = 0
not_spam = 0

for row in X_train:
    if row[57] == 1:
        spam += 1
    else:
        not_spam += 1

prob_spam = float(spam/len(X_train))    # .4 \__as expected
prob_not = not_spam/len(X_train)        # .6 /

# ------------------------ #
#          PART 3          #
# ------------------------ #

"""GAUSSIAN NAIVE BAYES"""

def naive_bayes(x, mean, std_dev):

    # As per assignment, do not divide by 0
    if std_dev == 0.0000:
        std_dev = 0.0001


    # Variable supplements for actual algorithm
    expon = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std_dev, 2))))
    denom = math.sqrt(2 * math.pi) * std_dev

    # Error checking
    if expon == 0.0:
        return expon - math.log(denom)
    else:
        return math.log(expon) - math.log(denom) * expon


"""TO SPAM, OR NOT TO SPAM"""

def predict():

    # Class prediction for test data
    predictions = []
    for row in range(len(X_t_features)):
        spam_probs = []
        not_probs = []

        for col in range(len(X_t_features[row])):
            # Spam probabilities for each entry
            log_of_spam_prob = naive_bayes(X_t_features[row,col], spam_mean_train[col], spam_std_train[col])
            spam_probs.append(log_of_spam_prob)
            # Not Spam probabilities
            log_of_not_prob = naive_bayes(X_t_features[row,col], not_mean_train[col], not_std_train[col])
            not_probs.append(log_of_not_prob)

        # predictions for each
        spam_prediction = math.log(prob_spam) + sum(spam_probs)
        not_prediction = math.log(prob_not) + sum(not_probs)

        # Compare results and sort answers (argmax)
        if spam_prediction < not_prediction:
            predictions.append(0.0)  # not spam
        else:
            predictions.append(1.0)  # spam

    return predictions

def data_report():

    # Generate confusion matrix
    # Also calculate accuracy, precision, and recall
    # experiment write up

    predictions = predict()
    correct = 0
    conf_matrix = np.zeros((2, 2))

    """
    CONFUSION MATRIX 
    
    0,0 = True Spam
    0,1 = False Spam
    1,0 = False Not Spam
    1,1 = True Not Spam
     ____ ____
    |_TS_|_FN_|
    |_FS_|_TN_|
    
    """

    for i,j in zip(X_t_classifier, predictions):
        # For accuracy purposes in report

        # True Spam / Not Spam cases
        if i == j:
            correct += 1
            if i == 1:
                conf_matrix[1,1] += 1  # increment not spam
            elif i == 0:
                conf_matrix[0,0] += 1  # increment spam


        if i == 0 and j == 1:  # False Spam
            conf_matrix[0, 1] += 1

        if i == 1 and j == 0:  # False Not Spam
            conf_matrix[1, 0] += 1

    # Calculate the Test Accuracy
    accuracy = correct / len(X_t_features)

    # Calculate the Precision
    # Precision = True Spam / (True Spam + False Spam)
    precision = (conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0]))

    # Calculate the Recall
    # Recall = True Spam / (True Spam + False Not Spam)
    recall = (conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0]))

    print("Recall:")
    print(("{0:.3f}%").format(recall * 100))
    print("Precision:")
    print(("{0:.3f}%").format(precision * 100))
    print("Accuracy:")
    print(("{0:.3f}%").format(accuracy * 100))
    print("Confusion Matrix:")
    print(""" ___________ ________________         
|_True_Spam_|_False Not Spam_|
|_FalseSpam_|__True_Not_Spam_|
    """)
    print(conf_matrix)
    
# Run Program
data_report()
