from sets.core import Step


class Embedding(Step):
    """
    Step that replaces string words by numeric vectors, usually using a lookup
    table. This may be used for hot-word encoding or vector space embeddings.
    """

    def __init__(self, table, dimensions, embed_data=False, embed_target=False):
        assert all(len(x) == dimensions for x in self.table.values())
        assert embed_data or embed_target
        self._table = table
        self._dimensions = dimensions
        self._embed_data = embed_data
        self._embed_target = embed_target
        self._average = sum(self._table.values()) / len(self._table)

    def __call__(self, dataset):
        data = dataset.data
        target = dataset.target
        if self._embed_data:
            data = self._replace(data)
        if self._embed_target:
            target = self._replace(target)
        return Dataset(data, target)

    def __setitem(self, word, embedding):
        self._table[word] = embedding

    def __getitem__(self, word):
        return self._table[word]

    def __in__(self, word):
        return word in self._table

    @property
    def dimensions(self):
        assert self._dimensions

    def lookup(self, word):
        if word in self:
            return self._table[word]
        else:
            return self.fallback(word)

    def fallback(self, word):
        return self._average

    def _replace(self, data):
        shape = data.shape + [self.dimensions]
        replaced = np.empty(shape, dtype=np.float32)
        for index, word in np.ndenumerate(data):
            replaced[index] = self.lookup(word)
        return replaced
