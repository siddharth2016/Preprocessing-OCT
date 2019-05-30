# Newton Divided Difference code

import numpy as np
import matplotlib.pyplot as plt

def coef(x, y):
    x.astype(float)
    y.astype(float)
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

    return np.array(a) # return an array of coefficient

def Eval(a, x, r):

    '''
    a : array returned by function coef()
    x : array of data points
    r : the node to interpolate at
    '''
    x.astype(float)
    n = len( a ) - 1
    temp = a[n]
    for i in range( n - 1, -1, -1 ):
        temp = temp * ( r - x[i] ) + a[i]
    return temp # return the y_value interpolation

x = np.array([4, 2, 7, 1, 8, 16, 10])
y = np.array([2, 5, 8, 10, 13, 16, 20])

a = coef(x, y)
print(a)

e = Eval(a, x, 10)
print(e)
