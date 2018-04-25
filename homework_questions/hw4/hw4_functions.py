######
######  This file includes different functions used in HW3
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    subgradient = 0
    if order==0:
        value = 0
        for i in range(n):
            value += max(0,1-np.asscalar(labels[i]*w.T*features[i].T))
        value = value/n
        return value
    elif order==1:
        value = math.inf
        for i in range(n):
            subgradient += 0 if np.asscalar(labels[i])*np.asscalar(np.dot(w.T,features[i].T)) > 1 else -labels[i]*features[i]
        subgradient = subgradient/n
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    subgradient = 0
    if order==0:
        value = 0
        for i in range(n):
            value += max(0,1-np.asscalar(labels[i]*w.T*features[i].T))
        value = value/n
        return value
    elif order==1:
        value = math.inf
        for iter in range(minibatch_size):
            i = random.randint(0,n-1)
            subgradient += 0 if np.asscalar(labels[i])*np.asscalar(np.dot(w.T,features[i].T)) > 1 else -labels[i]*features[i]
        subgradient = subgradient/minibatch_size
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
