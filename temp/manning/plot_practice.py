__author__ = 'lanx'
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 2], [3, 4]])

m = np.mat(x)
x[0,0] = 5


print m