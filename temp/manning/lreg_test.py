__author__ = 'lanx'

import regression
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def plotBestFit(wei,data, label):
    theta = wei
    dataArr = np.array(data)
    n = np.shape(dataArr)[0]

    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(label[i]) == 0:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-theta[0] - theta[1] * x) / theta[2] # z = w0 + w1x1 + w2x2

    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifier(theta, x1, x2):
    z = theta[0] + theta[1]*x1 + theta[2] * x2
    value = regression.sigmoid(z)
    return value

dataArr, labelArr = loadDataSet()
theta = regression.logisticRegression(dataArr, labelArr)
theta2 = regression.sga1(np.array(dataArr), labelArr)


print theta
"""
plotBestFit(theta2, dataArr, labelArr)

n = np.shape(dataArr)[0]
for i in range(n):
    output = classifier(theta, dataArr[i][1], dataArr[i][2])
    if(output >= 0.5):
        group = 1
    else:
        group = 0
    print dataArr[i][1], dataArr[i][2], output, "-->", group
"""

#-0.017612	14.053064	0
