import numpy as np
import requests
import matplotlib.pyplot as plt

def getdata(url):
    response = requests.get(url)
    rData = response.text.strip() #get the row data
    n = 0
    point1 = []
    point2 = []
    flag = 0
    for i in range(len(rData)):   #get the comments
        if(rData[i] == '#' and flag == 0):
            point1.append(i)
            flag = 1-flag
        if(rData[i] == '\n' and flag == 1):
            point2.append(i)
            flag = 1-flag
    dataSp = rData.split('\n')[len(point1):] #remove comments
    dataSpRow = dataSp[0].strip().split(' ') #set a observer as a rowlist
    metricLen = len(dataSp)
    metricWid = len(dataSpRow)-1
    x = np.zeros((metricLen,metricWid))
    y = np.zeros((metricLen,1))
    for i in range(metricLen):
        dataSpRow = dataSp[i].strip().split(' ')
        for j in range(metricWid):
            x[i][j] = float(dataSpRow[j])
            y[i][0] = float(dataSpRow[-1])
    return x,y

def linearRegreSin(url):
    [a,b] = getdata(url)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
    plt.figure(1)
    plt.plot(a,b,"*")
    y=a*theta
    plt.plot(a,y)
    print(pow(sum((y-b)**2),1/2)) #print MSE
    plt.show()


def linearRegreMul(url):
    [a,b] = getdata(url)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
    y=np.dot(a,theta)
    print(pow(sum((y-b)**2),1/2)) #print MSE

if __name__ == "__main__":
    linearRegreSin("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set3.dat")
    linearRegreMul("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    #http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set3.dat
