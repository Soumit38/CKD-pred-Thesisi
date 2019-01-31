#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: SoumitPc
"""

# DATA PREPROCESSING

# no ckd = 0, ckd = 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

dataset = pd.read_csv('ckd-dataset.csv')
X = dataset.iloc[:, 0:24].values
y = dataset.iloc[:, 24].values

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

for i in range(0,399):
    if y[i] == 'ckd':
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)

for a in range(0, 399):
    if X[a][5] == 'normal':
        X[a][5] = 0
    if X[a][5] == 'abnormal':
        X[a][5] = 1
        
for a in range(0, 399):
    if X[a][6] == 'normal':
        X[a][6] = 0
    if X[a][6] == 'abnormal':
        X[a][6] = 1
        
for a in range(0, 399):
    if X[a][7] == 'notpresent':
        X[a][7] = 0
    if X[a][7] == 'present':
        X[a][7] = 1
        
for a in range(0, 399):
    if X[a][8] == 'notpresent':
        X[a][8] = 0
    if X[a][8] == 'present':
        X[a][8] = 1
        
for a in range(0, 399):
    for b in range(18, 24):
        if X[a][b] == 'yes' or X[a][b] == 'good':
            X[a][b] = 0
        if X[a][b] == 'no' or X[a][b] == 'poor':
            X[a][b] = 1
    
for a in range(0,399):
    for b in range(0, 24):
        if(isinstance(X[a][b], int)):
            X[a][b] = float(X[a][b])
        elif(isinstance(X[a][b], str)):
            if(is_float(X[a][b])):
                X[a][b] = float(X[a][b])
                
totals = [0] * 24
added = [0] * 24           
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float)):
            totals[b] += X[a][b]
            added[b] += 1
            
averages = [0] * 24          
for a in range(0, 24):
    averages[a] = totals[a] / added[a]
 
c = 0
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float) == 0):
            X[a][b] = averages[b]
            c += 1
    
X = X.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

NB_y_pred = NB_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
NB_cm = confusion_matrix(y_test, NB_y_pred)
print("Naive Bayes Classifier : ")
print("==============================")
print("Confusion matrix : \n", NB_cm)

from sklearn.metrics import accuracy_score
NB_r = accuracy_score(y_test, NB_y_pred)
print("Accuracy : ", NB_r)

from sklearn.metrics import f1_score
NB_ff =f1_score(y_test, NB_y_pred, average='binary')
print("F1 score : ", NB_ff)


# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion='entropy')
DT_classifier.fit(X_train, y_train)
 
DT_y_pred = DT_classifier.predict(X_test)
 
from sklearn.metrics import confusion_matrix
DT_cm = confusion_matrix(y_test, DT_y_pred)
print("\nDecision Tree classifier : ")
print("==============================")
print("Confusion matrix : \n", DT_cm)
 
from sklearn.metrics import accuracy_score
DT_r = accuracy_score(y_test, DT_y_pred)
print("Accuracy : ", DT_r)
 
from sklearn.metrics import f1_score
DT_ff =f1_score(y_test, DT_y_pred, average='binary')
print("F1 score : ", DT_ff)

# Random Forest classifier

from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
RF_classifier.fit(X_train, y_train)

RF_y_pred = RF_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
RF_cm = confusion_matrix(y_test, RF_y_pred)
print("\nRandom Forest classifier : ")
print("==============================")
print("Confusion matrix : \n", RF_cm)

from sklearn.metrics import accuracy_score
RF_r = accuracy_score(y_test, RF_y_pred)
print("Accuracy : ", RF_r)

from sklearn.metrics import f1_score
RF_ff =f1_score(y_test, RF_y_pred, average='binary')
print("F1 score : ", RF_ff)

# SVM classifier

from sklearn.svm import SVC
SVM_classifier = SVC(kernel='linear')
SVM_classifier.fit(X_train, y_train)

SVM_y_pred = SVM_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
SVM_cm = confusion_matrix(y_test, SVM_y_pred)
print("\nSVM classifier : ")
print("==============================")
print("Confusion matrix : \n", SVM_cm)

from sklearn.metrics import accuracy_score
SVM_r = accuracy_score(y_test, SVM_y_pred)
print("Accuracy : ", SVM_r)

from sklearn.metrics import f1_score
SVM_ff =f1_score(y_test, SVM_y_pred, average='binary')
print("F1 score : ", SVM_ff)

















