import numpy as np
import time
from algorithm import bisection
from algorithm.function_on_line import function_on_line

def exact_line_search( func, x, direction, eps=1e-9, maximum_iterations=65536 ):
    """
    'Exact' linesearch (using bisection method)
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """

    x = np.matrix( x )
    direction = np.matrix( direction )

    value_0 = func( x , 0 )

    # setting an upper bound on the optimum.
    MIN_eta = 0
    MAX_eta = 1
    iterations = 0

    value = func( x + MAX_eta * direction, 0 )
    value = np.double( value )

    # look for a stepsize that gives a function value greater than in the current point
    while value<value_0 :

        MAX_eta *= 2

        value = func( x + MAX_eta * direction, 0 )

        iterations += 1

        if iterations >= maximum_iterations/2:
            break

    #construct new function equal to f on the line
    func_on_line = lambda eta, order: function_on_line( func, x + eta * direction, direction, order )

    # bisection search in the interval (MIN_t, MAX_t)
    return bisection(func_on_line, MIN_eta, MAX_eta, eps, maximum_iterations/2)
