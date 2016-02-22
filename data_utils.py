import struct
import numpy as np
import array
import matplotlib.pyplot as plt
import matplotlib as mpl

def loadMNISTImages(filename):
    print ('Loading MNIST Image Training data.......')
    image_file = open(filename,'rb')

    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    magic = struct.unpack('>I', head1)[0]
    num_examples = struct.unpack('>I', head2)[0]
    num_rows     = struct.unpack('>I', head3)[0]
    num_cols     = struct.unpack('>I', head4)[0]

    print num_examples, num_rows, num_cols

    """ Initialize dataset as array of zeros """

    dataset = np.zeros((num_rows*num_cols, num_examples))  # 28-by-28 * 60000

    """ Read the actual image data """

    images_raw  = array.array('B', image_file.read())
    image_file.close()

    """ Arrange the data in columns """
    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)
        dataset[:, i] = images_raw[limit1 : limit2]

    """ Normalize and return the dataset """
    return dataset / 255

def loadMNISTLabels(filename):
    print ('Loading MNIST Label Training data.......')
    label_file = open(filename,'rb')
    head1 = label_file.read(4)
    head2 = label_file.read(4)

    num_examples = struct.unpack('>I',head2)[0]

    """ Initialize data labels as array of zeros """
    labels = np.zeros(num_examples, dtype = np.int)

    """ Read the label data """
    labels_raw = array.array('b', label_file.read())
    label_file.close()

    """ Copy and return the label data """
    labels[:] = labels_raw[:]

    return labels

def showImage(image,label):
    image = np.reshape(image,(28,28)) * 255
    fig = plt.figure()
    plt.imshow(image, cmap=mpl.cm.Greys)
    plt.xlabel('y=%d' % label)
    plt.show()
