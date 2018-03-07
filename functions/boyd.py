from functions.imported_packages import *

def boyd_example_func(x, order=0):
  a=np.matrix('1  3')
  b=np.matrix('1  -3')
  c=np.matrix('-1  0')
  x=np.asmatrix(x)

  value = exp(a*x-0.1)+exp(b*x-0.1)+exp(c*x-0.1)
  if order==0:
      return value
  elif order==1:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      return (value, gradient)
  elif order==2:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      hessian = a.T*a*exp(a*x-0.1)+b.T*b*exp(b*x-0.1)+c.T*c*exp(c*x-0.1)
      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
