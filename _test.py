import numpy as np
import ml_utils
import tensorflow as tf
X = np.array([[1,2,3],[4,5,3],[7,8,9]])
index = [1,2]
print X[index]

index_col = [1,2]
print X[:,index]

index2 = np.where(X[:,2] == 3)
print X[:,index2[0]]

print X[1,2]
print X[1]

print range(1,6)

u = np.linspace(-1,1.5,50)
print u
print np.size(u)
print ml_utils.mapFeature(1,3)

y = np.array([0,5,2])

ground_truth = ml_utils.onehot_transform(y)
print ground_truth

A = np.array([[1,2],[3,4]])
B = np.array([[2,2],[3,3]])

print A
print B
print A*B

def sigmoid(inX):
    return 1.0 / (1+ np.exp(-inX))

with tf.Session() as sess:
    theta_tf = tf.ones([1,3])*128

    print theta_tf.eval()
    print tf.sigmoid(theta_tf).eval()
    print sigmoid(np.ones(3)*128)
    print tf.reduce_sum(theta_tf).eval()
    print tf.zeros([1]).eval()