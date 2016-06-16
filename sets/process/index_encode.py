import numpy as np
from sets.core import MapStep


class IndexEncode(MapStep):
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            vocabulary = []
        self.lookup = {}
        self.vocabulary = vocabulary
        for idx, x in enumerate(self.vocabulary):
            self.lookup[x] = idx

    def _get_shape(self, ds):
        return (ds.shape[0],)

    def apply(self, batch):
        encoded = np.empty((len(batch)), dtype=np.int32)
        for idx, x in enumerate(batch):
            if x not in self.vocabulary:
                print(len(self.vocabulary))
                self.lookup[x] = len(self.vocabulary)
                self.vocabulary.append(x)
            encoded[idx] = self.lookup[x]
        return encoded
