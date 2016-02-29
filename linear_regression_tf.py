__author__ = 'lanx'

import numpy as np
import tensorflow as tf


def linearRegression(X, Y, theta, alpha, numIterations):
    m,n = np.shape(X)
    print m,n

    theta_tf = tf.Variable(tf.zeros([n,1]))

    x_input = tf.placeholder(tf.float32, shape=(m,n))
    y_input = tf.placeholder(tf.float32, shape=(m,1))
    init_op = tf.initialize_all_variables()

    hypothesis = tf.matmul(x_input, theta_tf)
    cost = tf.reduce_mean(tf.square(hypothesis - y_input))
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)

    print theta_tf
    print x_input
    print y_input
    print hypothesis
    print cost

    with tf.Session() as sess:
        sess.run(init_op)
        for step in xrange(numIterations):
            result = sess.run([train,theta_tf,cost], feed_dict={x_input:X, y_input:Y})

            if step % 1000 == 0:
                print(step, result[1], "cost is %f" % result[2])
        theta = theta_tf.eval()

    return theta
