import numpy as np
import time

def function_on_line(func, x, direction, order):
    if order==0:
        value = func(x, order)
        return value
    elif order==1:
        value, gradient = func(x, order)
        return (value, gradient.T*direction)
    elif order==2:
        value, gradient, hessian = func(x, order)
        return (value, gradient.T*direction, direction.T*hessian*direction)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
