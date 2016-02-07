import numpy as np
import matplotlib.pyplot as plt
from getData import *
from GradientD import *

def linearRegreSin(url):
    [a,b] = getData(url)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
    plt.figure(1)
    plt.plot(a,b,"*")
    y=a*theta
    plt.plot(a,y)
    print(pow(sum((y-b)**2),1/2)) #print MSE
    plt.show()

def linearRegreMul(url):
    [a,b] = getData(url)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
    y=np.dot(a,theta)
    print(pow(sum((y-b)**2),1/2)) #print MSE

if __name__ == "__main__":
    #linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set1.dat")
    #linearRegreMul("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    [m,n] = getData("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set1.dat")
    iterations = 1500
    alpha = 0.01
    theta = np.matrix(np.zeros(2)).T #initialize fitting parameters
    gradientD(m,n,theta,alpha,iterations)