__author__ = 'lanx'
import regression
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat=[]; labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
#        print "curLine:" , curLine
        for i in range(numFeat):
#            print "curLine[%d]" % i, (curLine[i])
            lineArr.append(float(curLine[i]))
#        print "lineArr: ", lineArr
#        print "curLine[-1]", curLine[2]
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

x, y = loadDataSet('ex0.txt')

ws = regression.standRegres(x, y)
ws2 = regression.gradRegres(x, y)
#ws3 = regression.gradRegressMatrix(x, y)

print ws
print ws2
#print ws3

xArr = np.asarray(x)
yArr = np.asarray(y)
yHat = xArr * ws

print np.corrcoef(yHat.T, yArr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xArr[:,1], yArr)

xCopy = xArr.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1],yHat)


plt.show()

