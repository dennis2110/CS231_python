# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:26:27 2018

@author: 606C
"""

import random

import cv2
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#num_classes = len(classes)
#samples_per_class = 8
#for y, cls in enumerate(classes):
#    idxs = np.flatnonzero(y_train == y)
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx].astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()

img = X_train[100].astype('uint8')
cv2.imshow('img',img)

cv2.waitKey(0)
img = X_train[1000].astype('uint8')
cv2.imshow('img',img)

cv2.waitKey(0)
img = X_train[10].astype('uint8')
cv2.imshow('img',img)

cv2.waitKey(0)
img = X_train[1010].astype('uint8')
cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()