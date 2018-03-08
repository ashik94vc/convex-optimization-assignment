import numpy as np
import time
from algorithm import exact_line_search

def gradient_descent( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=exact_line_search, *linesearch_args ):
    """
    Gradient Descent
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point, should be a float
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    # gradient updates
    while True:

        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # direction= (TODO)
        direction = -gradient

        # if (TODO: TERMINATION CRITERION): break
        if np.linalg.norm(gradient)**2 < eps:
            break

        t = linesearch( func, x, direction, *linesearch_args )

        # x= (TODO: UPDATE x)
        x = x + t*direction

        iterations += 1
        if iterations >= maximum_iterations:
            break

    return (x, values, runtimes, xs)
