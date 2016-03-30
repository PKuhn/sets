import nltk
from sets.core import Step, Dataset


class Tokenize(Step):

    def __call__(self, dataset):
        data = self._tokenize(dataset.data)
        data = self._pad(data)
        return Dataset(data, dataset.target)

    @staticmethod
    def _tokenize(data):
        return [nltk.word_tokenize(x) for x in data]

    @staticmethod
    def _pad(tokenized):
        width = max(len(x) for x in tokenized)
        padded = np.empty((len(tokenized), width))
        for index, tokens in enumerate(tokenized):
            padded[index][:len(tokens)] = tokens
        return padded
