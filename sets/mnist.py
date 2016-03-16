import struct
import array
import gzip
import numpy as np
from dataset.dataset import Dataset, Parser


class Mnist(Parser):
    """
    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples. It is a
    subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image. It is a good database
    for people who want to try learning techniques and pattern recognition
    methods on real-world data while spending minimal efforts on preprocessing
    and formatting. (From: http://yann.lecun.com/exdb/mnist/)
    """

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    def parse(self, train_x, train_y, test_x, test_y):
        train = list(self.read(train_x, train_y))
        test = list(self.read(test_x, test_y))
        train_data, train_target = [x[0] for x in train], [x[1] for x in train]
        test_data, test_target = [x[0] for x in test], [x[1] for x in test]
        return train_data, train_target, test_data, test_target

    @classmethod
    def read(cls, data_filename, target_filename):
        data_array, data_size, rows, cols = cls._read_data(data_filename)
        target_array, target_size = cls._read_target(target_filename)
        assert data_size == target_size
        for i in range(data_size):
            data = data_array[i * rows * cols:(i + 1) * rows * cols]
            data = np.array(data).reshape(rows * cols) / 255
            target = np.zeros(10)
            target[target_array[i]] = 1
            yield data, target

    @staticmethod
    def _read_data(filename):
        with gzip.open(filename, 'rb') as file_:
            _, size, rows, cols = struct.unpack('>IIII', file_.read(16))
            target = array.array('B', file_.read())
            assert len(target) == size * rows * cols
            return target, size, rows, cols

    @staticmethod
    def _read_target(filename):
        with gzip.open(filename, 'rb') as file_:
            _, size = struct.unpack('>II', file_.read(8))
            target = array.array('B', file_.read())
            assert len(target) == size
            return target, size
