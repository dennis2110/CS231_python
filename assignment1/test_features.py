# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:30:39 2018

@author: 606C
"""
from cs231n.features import *
import matplotlib.pyplot as plt

img = plt.imread('images.png')
print(img.shape)
dst_ravel, dst  = hog_feature(img)
 
print(dst_ravel)
print(dst)