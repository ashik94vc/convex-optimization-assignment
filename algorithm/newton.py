import numpy as np
import time

def newton( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """
    Newton's Method
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix( initial_x )

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    # newton's method updates
    while True:

        value, gradient, hessian = func( x , 2 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        hessian = np.matrix( hessian )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # direction = (TODO)
        direction = -hessian.I * gradient

        # if (TODO: TERMINATION CRITERION): break
        if gradient.T*hessian.I*gradient < eps:
            break

        t = linesearch( func, x, direction )

        # x = (TODO: UPDATE x)
        x = x + t*direction

        iterations += 1
        if iterations >= maximum_iterations:
            break

    return (x, values, runtimes, xs)
