from functions.imported_packages import *

def quadratic( H, b, x, order=0 ):
    """
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    value = 0.5 * x.T * H * x + b.T * x

    if order == 0:
        return value
    elif order == 1:
        gradient = H * x + b
        return (value, gradient)
    elif order == 2:
        gradient = H * x + b
        hessian = H
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


#Lecture 4 converting elliptical to circular contour
def transformed_quadratic(H, b, x, order=0):
    """
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    x1 = np.power(H,0.5)*x
    value = 0.5*(np.linalg.norm(x1,ord=2)**2)+b.T*np.power(H,-0.5)*x1

    if order == 0:
        return value
    elif order == 1:
        gradient = x1 + b.T*np.power(H,-0.5)
        return (value, gradient)
    elif order == 2:
        gradient = x1 + b.T*np.power(H,-0.5)
        hessian = np.identity(2)
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
