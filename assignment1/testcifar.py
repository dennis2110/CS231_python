# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:29:44 2018

@author: 606C
NearestNeighbor classifier
"""
import random


import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

#from __future__ import print_function

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_temp = X_train
y_train_temp = y_train
X_train_folds = np.array_split(X_train_temp, num_folds)
y_train_folds = np.array_split(y_train_temp, num_folds)
print(X_train_folds[4])

k_to_accuracies = {}

num_test = X_train_folds[0].shape[0]
for j in range(len(k_choices)):
    k = k_choices[j]
    for i in range(1,num_folds+1):
        X_train_temp = np.concatenate((X_train_folds[num_folds-i],X_train_folds[num_folds-i-1],X_train_folds[num_folds-i-2],X_train_folds[num_folds-i-3]),axis = 0)
        y_train_temp = np.concatenate((y_train_folds[num_folds-i],y_train_folds[num_folds-i-1],y_train_folds[num_folds-i-2],y_train_folds[num_folds-i-3]))
        X_test_temp = X_train_folds[num_folds-i-4]
        y_test_temp = y_train_folds[num_folds-i-4]
        classifier.train(X_train_temp, y_train_temp)
        y_test_pred = classifier.predict(X_test_temp, k=k)
        num_correct = np.sum(y_test_pred == y_test_temp)
        accuracy = float(num_correct) / num_test
        k_to_accuracies.setdefault(k,[]).append(accuracy)

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean,yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()