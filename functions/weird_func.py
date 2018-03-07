from functions.imported_packages import *

def weird_func( x, order=0 ):

  # f(x) = x^4 + 6x^2 + 12(x-4)e^(x-1)
  value = pow(x, 4) + 6 * pow(x, 2) + 12 * (x - 4) * exp(x - 1)

  if order==0:
      return value
  elif order==1:
      # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
      gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

      return (value, gradient)
  elif order==2:
      # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
      gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

      # f''(x)= 12 (1 + e^(-1 + x) (-2 + x) + x^2)
      hessian = 12 * (1 + (x-2) * exp(x-1) + pow(x,2))

      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
