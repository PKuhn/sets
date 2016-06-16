import re
import nltk
import numpy as np
import h5py
from sets.core import MapStep


class Tokenize(MapStep):

    _regex_tag = re.compile(r'<[^>]+>')

    def _get_datatype(self, column):
        return h5py.special_dtype(vlen=str)

    def _get_shape(self, ds):
        max_len = 0
        for sent in ds:
            tokenized = list(self._tokenize(sent))
            max_len = max(max_len, len(tokenized))
        self.shape = (ds.shape[0], max_len)
        return self.shape

    def apply(self, sentences):
        shape = (sentences.shape[0], self.shape[1])
        tokenized = np.empty(shape, dtype=object)
        for idx, sentence in enumerate(sentences):
            tokenized_sentence = list(self._tokenize(sentence))
            padding = ['' for _ in range(
                self.shape[1] - len(tokenized_sentence))]
            tokenized[idx] = tokenized_sentence + padding
        return tokenized

    @classmethod
    def _tokenize(cls, sentence):
        """
        Split a sentence while preserving tags.
        """
        while True:
            match = cls._regex_tag.search(sentence)
            if not match:
                yield from cls._split(sentence)
                return
            chunk = sentence[:match.start()]
            yield from cls._split(chunk)
            tag = match.group(0)
            yield tag
            sentence = sentence[(len(chunk) + len(tag)):]

    @staticmethod
    def _split(sentence):
        tokens = nltk.word_tokenize(sentence)
        tokens = [x.lower() for x in tokens]
        return tokens
