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
    if order==0:
        # value = ( TODO: value )
        return value
    elif order==1:
        # value = ( TODO: value )
        # subgradient = ( TODO: sungradient )
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
    
def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        return value
    elif order==1:
        # value = ( TODO: value )
        # subgradient = ( TODO: sungradient )
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
