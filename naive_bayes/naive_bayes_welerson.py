#!/usr/bin/python

#Project Author: Welerson Melo

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels: 
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

t0 = time()

clf.fit(features_train, labels_train)

print ("tempo de treinamento:", round(time()-t0), "s")

t0 = time()

labels_predict = clf.predict(features_test)

print ("tempo de previsao:", round(time()-t0), "s")

accuracy_value = accuracy_score(labels_test, labels_predict)

print (accuracy_value)

#########################################################
