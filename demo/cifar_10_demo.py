# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:00:09 2018

@author: 606C
CIFAR-10
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle('C:\\Users\\606C\\Desktop\\CS231_python\\cifar-10-batches-py\\data_batch_1')
num = 456
print('123:%d' % num)
print('123:{0}'.format(num))