from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

X = loadMNISTImages('data/mnist/train-images-idx3-ubyte')
print np.shape(X)
print X[:,1]

image = np.reshape(X[:,2],(28,28))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(image, cmap=mpl.cm.Greys)
plt.show()
