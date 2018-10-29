#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
### your code goes here ###

import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(C = 10000.0, kernel = "rbf")

t0 = time()

clf.fit(features_train, labels_train)

print ("tempo de treinamento:", round(time()-t0), "s")

t0 = time()

labels_predict = clf.predict(features_test)

cont = 0
for c in labels_predict:
	if c == 1:
		cont = cont + 1

print "Quant emails de cris: ", cont

print("tempo de previsao:", round(time()-t0), "s")

accuracy_value = accuracy_score(labels_test, labels_predict)

print(accuracy_value)


#########################################################


