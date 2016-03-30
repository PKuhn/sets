import numpy as np
from sets.core import Step, Dataset


class RelativeIndices(Step):

    def __init__(self, *names):
        self._names = names

    def __call__(self, dataset):
        data = np.empty((len(dataset.data), len(self._names)))
        for index, tokens in enumerate(dataset.data):
            positions = self._positions(tokens)
            data[index] = self._relative_sequence(positions, len(tokens))
        return Dataset(data, dataset.target)

    def _positions(self, tokens):
        return [tokens.index(x) for x in self._names]

    @staticmethod
    def _relative_sequence(positions, length):
        sequence = np.empty((length, len(positions)))
        for index, position in enumerate(indices):
            for current in range(length):
                sequence[current][index] = index - position
        return sequence
