__author__ = 'lanx'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ml_utils

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def logisticRegression(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    m,n = np.shape(dataMat)
    alpha = 0.001
    numIteration = 500

    theta = np.ones((n,1))
    for k in range(numIteration):
        h = sigmoid(dataMat * theta)
        loss = (labelMat - h)

        gradient = dataMat.transpose() * loss
        theta = theta + alpha * gradient
    return theta

def sigmoid(inX):
    return 1.0 / (1+ np.exp(-inX))

def classifier(theta, x1, x2):
    z = theta[0] + theta[1]*x1 + theta[2] * x2
    value = sigmoid(z)
    return value

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

data = np.genfromtxt('data/logistic_data1.txt', delimiter=',')
X = data[:,[0,1]]
y = data[:,[2]]
m,n = np.shape(X)

X, mu, sigma = ml_utils.featureNormalize(X)
print X, mu, sigma

X = np.column_stack((np.ones(m),X))
theta = np.zeros(n+1)

#Y = np.zeros((m,1))
#Y[:,0] = data[:,2]


#dataArr, labelArr = loadDataSet()
theta = logisticRegression(X,y)

print "Logistic Regression(Original)"
print theta

W = tf.Variable(tf.ones([1,3]))
x_input = tf.placeholder(tf.float32, shape=(len(X),3))
y_input = tf.placeholder(tf.float32)
init_op = tf.initialize_all_variables()

activation = tf.sigmoid(tf.matmul(W,x_input, transpose_b=True))
loss =  - tf.log(np.prod(tf.pow(activation,y_input)*tf.pow(1-activation,(1-y_input))))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(init_op)
    #sess.run(y_hat, feed_dict={x_input: x})
    for step in xrange(10000):
        sess.run(train, feed_dict={x_input:X, y_input:y})
        if step % 10000 == 0:
            print(step, sess.run(W))
    theta2 = W.eval().transpose()

print "Logistic Regression(TensorFlow)", theta2

n = np.shape(X)[0]
tot = 0
error = 0
for i in range(n):
    output = classifier(theta2, X[i][1], X[i][2])
    if(output >= 0.5):
        group = 1
    else:
        group = 0
    print X[i][1], "\t", X[i][2], "\t\t", output, "\t-->", group, y[i]
    tot = tot + 1
    if(group != y[i]):
        error = error + 1

print "Total: %d, Error: %d Accuracy: %f" % (tot, error, (tot-error)/float(tot)*100)

#plotBestFit(theta2, dataArr, labelArr)


#print theta[0][0], theta[0][1], theta[0][2]
#print classifier(theta, 0.3, 0.7)