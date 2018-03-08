from functions.imported_packages import *

def circle_function(x, order=0):
    r=10.0
    x=np.asmatrix(x)

    value = np.power(r,2) - x.T*x
    if order==0:
        return value
    elif order==1:
        gradient = 2*x
        return (value, gradient)
    elif order==2:
        gradient = 2*x
        hessian = 2*np.identity(2)
        return (value, gradient, hessian)
    else:
          raise ValueError("The argument \"order\" should be 0, 1 or 2")
