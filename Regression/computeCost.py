from __future__ import division
import numpy as np

# Compute cost for linear regression
def computeCost(X, y, theta):
    m = len(y)  #number of tranning examples
    J = 0
    J = 1/(2*m)*np.dot((X*theta - y).T,(X*theta - y))
    return J