import re
import tarfile
import yaml
import sets.utility


class StreamMeta(dict):

    def __init__(self, container):
        self._container = container
        self.update(self._defaults())
        if 'stream.yaml' in self._container.getnames():
            file_ = self._container.extractfile('stream.yaml')
            self.update(yaml.load(file_))

    def __setitem__(self, key, value):
        self[key] = value
        file_ = yaml.save(self)
        self._container.addfile('stream.yaml', file_)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def _defaults():
        length = 0
        inner_shape = ()
        return locals()


class Stream:
    """
    Stream of Numpy chunks, synched to disk.
    """

    CHUNK_NAME = re.compile(r'./chunk-[0-9]+.npy')

    def __init__(self, file_=None, filename=None):
        self._container = tarfile.open(filename, mode='w')
        self._meta = StreamMeta(self._container)

    @property
    def shape(self):
        return (self._meta.length,) + self._meta.inner_shape

    def __len__(self):
        return self._meta.length

    def chunks(self):
        """
        Iterate over Numpy chunks of the dataset.
        """
        pass

    def rows(self):
        """
        Iterate over individual rows of the dataset.
        """
        pass

    def rename(self, filename):
        pass

    def copy(self, filename):
        pass

    def append(self, data):
        """
        Add a Numpy chunk to the stream. If you want equally sized chunk files
        internally, call reorganize_chunks() afterwards.
        """
        if not self._meta.inner_shape:
            self._meta.inner_shape = data.shape[1:]
        pass

    def __getitem__(self, slice_):
        """
        Return the selected rows as a Numpy array. Throw a MemoryError if the
        data doesn't fit into memory.
        """
        pass

    def __setitem__(self, slice_, data):
        """
        Replace the selected range with a Numpy chunk.
        """
        pass

    def map(self, function):
        """
        Modify the stream inplace by applying the provided map function on the
        Numpy chunks of the stream. The function must return Numpy chunks of
        the same length.
        """
        pass

    def filter(self, function):
        """
        Modify the stream inplace my applying the provided filter function on
        the Numpy chunks of the stream. The function must return a 1D boolean
        numpy array.
        """
        pass

    def shuffle(self, seed=0):
        """
        Shuffle the order of chunks and the rows within each chunk.
        """
        pass

    def to_numpy(self):
        """
        Load the whole stream into a Numpy array. Throw a MemoryError if the
        data doesn't fit into memory.
        """
        pass

    def reorganize_chunks(self, size):
        pass

    def _chunk_names(self):
        filenames = self._container.getmembers()
        filenames = [x for x in filenames if type(self).CHUNK_NAME.match(x)]
        filenames = utility.natural_sort(filenames)
        return filenames
