import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from getData import *
from GradientD import *

def linearRegreSin(url,degree):
    [a,b] = getData(url)
    trainA = a[0:139]
    trainB = b[0:139]
    testA = a[140:]
    testB = b[140:]

    poly = PolynomialFeatures(degree)
    trainA = np.float64(poly.fit_transform(trainA))
    testA = np.float64(poly.fit_transform(testA))
    theta = np.dot(np.dot(np.linalg.inv(np.dot(trainA.T,trainA)),trainA.T),trainB)
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('data')
    plt.plot(trainA[:,1],trainB,"r*")
    y=np.dot(trainA, theta)
    print(pow(sum((y-trainB)**2),1/2)/140) #print MSE

    y=np.dot(testA, theta)
    #plt.plot(testA[:,1], testB, "r.")
    plt.plot(testA[:,1],y,"k*")
    print(pow(sum((y-testB)**2),1/2)/60) #print MSE
    plt.show()
    print(theta)

def linearRegreMul(url):
    [a,b] = getData(url)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
    y=np.dot(a,theta)
    print(pow(sum((y-b)**2),1/2)) #print MSE

if __name__ == "__main__":
    #linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set1.dat",1)
    for i in range(1,10):
        linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set2.dat",i)
        linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set3.dat",i)
        linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set4.dat",i)
    #linearRegreMul("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    [m,n] = getData("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set4.dat")
    k = np.ones((n.size,1))
    m = np.hstack([k,m])
    theta = np.dot(np.dot(np.linalg.inv(np.dot(m.T,m)),m.T),n)
    print(theta)
    iterations = 1500
    alpha = 0.005
    theta = np.matrix(np.zeros(m.shape[1])).T #initialize fitting parameters
    [theta, J_history] = gradientD(m,n,theta,alpha,iterations)
    y = np.array(np.dot(m,theta))
    print(J_history)
    #print(y)
    print(pow(sum((y-n)**2),1/2)/(n.size)) #print MSE
    # plt.figure(1)
    # plt.plot(m,n,"*")
    # plt.plot(m,y,"*")
    # plt.show()
    print(theta)
