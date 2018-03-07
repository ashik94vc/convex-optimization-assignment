import numpy as np
import time

def bisection( one_d_fun, MIN, MAX, eps=1e-5, maximum_iterations=65536 ):

  # counting the number of iterations
  iterations = 0

  if eps <= 0:
      raise ValueError("Epsilon must be positive")

  while True:

    MID = ( MAX + MIN ) / 2

    # Oracle access to the function value and derivative
    value, derivative = one_d_fun( MID, 1 )

    # if (TODO: TERMINATION CRITERION): break
    if (MAX - MIN) <= eps:
        break
    if derivative < 0:
        MIN = MID
    # if derivative... (TODO: LINE SEARCH)
    elif derivative > 0:
        MAX = MID
    else:
        break
    iterations += 1
    if iterations>=maximum_iterations:
        break

  return MID
