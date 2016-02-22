import struct
import numpy as np
import array

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

    print magic, num_examples, num_rows, num_cols

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