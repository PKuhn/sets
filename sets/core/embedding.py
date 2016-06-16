import warnings
import numpy as np
from sets.core import MapStep


class Embedding(MapStep):
    """
    Replace string words by numeric vectors using a lookup table. The default
    fallback for unknown words is the average embedding vector and a zero
    vector for falsy words.
    """

    def __init__(self, words, embeddings):
        """
        Words is a list of words to embedd. Embeddings is a numpy array of same
        length.
        """
        self._index = {self.key(k): i for i, k in enumerate(words)}
        if len(self._index) != len(words):
            warnings.warn('the keys of some words override each other')
        self._embeddings = np.array(embeddings)
        self._shape = self._embeddings.shape[1:]
        self._average = self._embeddings.mean(axis=0)
        self._zeros = np.zeros(self.shape)

    @property
    def shape(self):
        return self._shape

    def _get_shape(self, ds):
        return ds.shape + self.shape

    def _get_datatype(self, ds):
        return float

    def __contains__(self, word):
        return self.key(word) in self._index

    def __getitem__(self, word):
        index = self._index[self.key(word)]
        embedding = self._embeddings[index]
        return embedding

    def apply(self, array):
        shape = array.shape + self.shape
        embedded = np.empty(shape)
        for index in np.ndindex(array.shape):
            embedded[index] = self._lookup(array[index])
        return embedded

    def key(self, word):
        # pylint: disable=no-self-use
        if isinstance(word, np.ndarray):
            return word.tostring()
        return word

    def fallback(self, word):
        return self._average

    def _lookup(self, word):
        if word in self:
            return self[word]
        if self._is_null(word):
            return self._zeros
        else:
            return self._average

    @staticmethod
    def _is_null(word):
        if isinstance(word, np.ndarray):
            return not word.size
        return not word
