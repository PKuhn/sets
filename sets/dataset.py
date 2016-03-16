import errno
import os
import random
import shutil
from urllib.request import urlopen
import numpy as np


class Dataset:
    """
    A list of data and target pairs that can be saved and loaded.
    """

    def __init__(self, name, folder, data=None, target=None):
        self._name = name
        self._folder = folder
        self._data = data
        self._target = target

    def __len__(self):
        assert len(self.data) == len(self.target)
        return len(self.data)

    def __iter__(self):
        yield from zip(self.data, self.target)

    @property
    def data(self):
        assert self._data is not None
        return self._data

    @property
    def target(self):
        assert self._target is not None
        return self._target

    @data.setter
    def data(self, value):
        assert self._data is None
        if value is None:
            return
        self._data = np.array(value)

    @target.setter
    def target(self, value):
        assert self._target is None
        if value is None:
            return
        self._target = np.array(value)

    @property
    def cached(self):
        filenames = [self._filename(x) for x in ('data', 'target')]
        return all(os.path.isfile(x) for x in filenames)

    @property
    def loaded(self):
        return self._data is not None and self._target is not None

    def load(self):
        assert self.cached
        self._data = np.load(self._filename('data'))
        self._target = np.load(self._filename('target'))

    def save(self):
        assert self.loaded
        np.save(self._filename('data'), self._data)
        np.save(self._filename('target'), self._target)

    def random_batch(self, size):
        assert self.loaded
        indices = random.sample(range(len(self)), size)
        return self.data[indices], self.target[indices]

    def _filename(self, attribute):
        filename = '{}-{}.npy'.format(self._name, attribute)
        return os.path.join(self._folder, filename)


class Parser:
    """
    Download files and parse them to provide training and testing datasets.
    """

    urls = []

    def __init__(self):
        self._train = Dataset(self._folder(), 'train')
        self._test = Dataset(self._folder(), 'test')
        self._ensure_downloads()
        self._ensure_datasets()
        assert self._train.loaded and self._test.loaded

    def parse(self, *files):
        """
        Parse the downloaded files given their filenames passed as arguments.
        Return four Numpy arrays holding the training data, training targets,
        testing data and testing targets.
        """
        raise NotImplementedError

    @staticmethod
    def show(data, target):
        """
        Visualize a single example of the dataset.
        """
        raise NotImplementedError

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    def _ensure_downloads(self):
        for url in type(self).urls:
            filename = self._filename(url)
            if os.path.isfile(filename):
                continue
            print('Download', filename)
            self._download(url, filename)

    def _ensure_datasets(self):
        if self._train.cached and self._test.cached:
            print('Load cached dataset')
            self._train.load()
            self._test.load()
        else:
            filenames = [self._filename(x) for x in type(self).urls]
            print('Parse dataset')
            results = self.parse(*filenames)
            self._train.data, self._train.target = results[:2]
            self._test.data, self._test.target = results[2:]
        self._train.save()
        self._test.save()

    @classmethod
    def _filename(cls, url):
        _, filename = os.path.split(url)
        return os.path.join(cls._folder(), filename)

    @classmethod
    def _folder(cls, prefix='~/.dataset'):
        name = cls.__name__.lower()
        path = os.path.expanduser(os.path.join(prefix, name))
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
        return path

    @staticmethod
    def _download(url, filename):
        with urlopen(url) as response, open(filename, 'wb') as file_:
            shutil.copyfileobj(response, file_)
        return filename
