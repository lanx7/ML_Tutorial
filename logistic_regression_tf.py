__author__ = 'lanx'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def sigmoid(inX):
    return 1.0 / (1+ np.exp(-inX))

def predict(theta, X):
    m,n = np.shape(X)
    p = np.zeros(m)

    h = sigmoid(np.dot(X, theta))
    for i in range(np.size(h)):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p

def logisticRegression(X, Y, theta, alpha, numIterations):
    m,n = np.shape(X) # X: m-by-n matrix
    theta = tf.Variable(tf.zeros([n,1]))

    x_input = tf.placeholder(tf.float32, shape=(m,n))
    y_input = tf.placeholder(tf.float32, shape=(m,1))

    init_op = tf.initialize_all_variables()

    h = tf.sigmoid(tf.matmul(x_input, theta))

    y1_term = -y_input * tf.log(h)
    y0_term = (1.0-y_input) * tf.log(1.0-h)
    cost = tf.reduce_sum(y1_term - y0_term) / m

    #print h
    #print y1_term
    #print y0_term
    #print cost

    #cost = - tf.log(tf.reduce_prod(tf.pow(activation,y_input)*tf.pow(1-activation,(1-y_input))))
    #print cost

    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(init_op)
        #sess.run(y_hat, feed_dict={x_input: x})
        for step in xrange(numIterations):
            result = sess.run([train,cost], feed_dict={x_input:X, y_input:Y})
            if step % 10000 == 0:
                print(step, sess.run(theta), result[1])
        theta2 = theta.eval()

    print "Logistic Regression(TensorFlow)", theta2
    return theta2

def logisticRegression_ref(X, Y, theta, alpha, numIterations):
    m,n = np.shape(X) # X: m-by-n matrix
    x_input = tf.placeholder(tf.float32, [None,n])
    y_input = tf.placeholder(tf.float32, [None])

    theta = tf.Variable(tf.zeros([n,1]), tf.float32)
    b = tf.Variable(tf.zeros([1], tf.float32))
    outputs = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(x_input, theta), [1]) + b))

    init_op = tf.initialize_all_variables()

    loss_tmp = - tf.to_double(y_input) * tf.log(outputs) - (1.0 - tf.to_double(y_input)) * tf.log(1.0 - outputs)
    #loss = tf.reduce_sum(tf.to_float(loss_tmp))
    loss = tf.reduce_mean(tf.to_float(loss_tmp))

    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(init_op)
        #sess.run(y_hat, feed_dict={x_input: x})
        for step in xrange(numIterations):
            result = sess.run([train], feed_dict={x_input:X, y_input:Y})
            if step % 10000 == 0:
                print(step, sess.run(theta))
        theta2 = theta.eval()

    print "Logistic Regression(TensorFlow)", theta2
    return theta2



def logisticRegression_ref2(X, Y, theta, alpha, numIterations):
    W = tf.Variable(tf.ones([1,3]))
    x_input = tf.placeholder(tf.float32, shape=(len(X),3))
    y_input = tf.placeholder(tf.float32)
    init_op = tf.initialize_all_variables()

    activation = tf.sigmoid(tf.matmul(W,x_input, transpose_b=True))
    loss =  - tf.log(np.prod(tf.pow(activation,y_input)*tf.pow(1-activation,(1-y_input))))
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(init_op)
        #sess.run(y_hat, feed_dict={x_input: x})
        for step in xrange(numIterations):
            sess.run(train, feed_dict={x_input:X, y_input:y})
            if step % 10000 == 0:
                print(step, sess.run(W))
        theta2 = W.eval().transpose()

    print "Logistic Regression(TensorFlow)", theta2
    return theta2
