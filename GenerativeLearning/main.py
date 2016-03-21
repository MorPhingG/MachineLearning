import numpy as np
from sklearn.lda import LDA
from getData import *

from sklearn import datasets

# Q1
# [x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
# x = x[0:100,0]
# y = y[0:100]
# clf = LDA()
# clf.fit(x,y)
# LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
#   store_covariance=False, tol=0.0001)
# for i in range(100):
#     y_pre = clf.predict([x[i]])
#     print(y_pre)

# Q2
[x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
x = x[0:100]
y = y[0:100]
sum = np.zeros((4,1))
for i in range(4):
    for j in range(50):
        sum[i] = sum[i] + x[j][i]
sum = sum/50 #calculate mean
std = np.std(x[0:50],axis=0)

x = x[50:100]
sum1 = np.zeros((4,1))
for i in range(4):
    for j in range(50):
        sum1[i] = sum1[i] + x[j][i]
sum1 = sum1/50 #calculate mean
std1 = np.std(x,axis=0)

[x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
x = x[0:100]
y = y[0:100]
clf = LDA()
clf.fit(x,y)
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)
y_pre = np.zeros((100,1))
for i in range(100):
    y_pre[i] = clf.predict([x[i]])

# Q3
[x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
x = x[100:150]
sum2 = np.zeros((4,1))
for i in range(4):
    for j in range(50):
        sum2[i] = sum2[i] + x[j][i]
sum2 = sum2/50 #calculate mean
std2 = np.std(x,axis=0)

[x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
clf = LDA()
clf.fit(x,y)
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)
y_pre = np.zeros((150,1))
for i in range(150):
    y_pre[i] = clf.predict([x[i]])

[x,y]=getData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
x = x[0:120]
y = y[0:120]
clf = LDA()
clf.fit(x,y)
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)
y_pre = np.zeros((120,1))
for i in range(120):
    y_pre[i] = clf.predict([x[i]])

# Q4
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" (iris.data.shape[0],(iris.target != y_pred).sum()))