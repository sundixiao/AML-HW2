
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import requests
import urllib.request as urllib2



a = np.array([[1,3,5],[2,4,6],[1,1,1]])
mean = np.mean(a, axis=0)
mean = mean.reshape((1,3))
a_mean = a-mean

a_new = np.concatenate((a,mean),axis=0) 

batch = np.array([1,2,3,4,5,6])
batch = batch.reshape((1,6))
a = np.array([1,2,3,4,5,6])
a = a.reshape((6,1))
c = np.dot(batch,a)