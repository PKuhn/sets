from zipfile import ZipFile
from sets.core import Embedding


class Glove(Embedding):
    """
    The pretrained word embeddings from the Standford NLP group computed by the
    Glove model.
    From: http://nlp.stanford.edu/projects/glove/
    """

    def __init__(self, dimensions=100, embed_data=False, embed_target=False):
        assert dimensions in (50, 100, 300)
        table = self.cache('embeddings-{}'.format(dimensions),
                           lambda: self.load(dimensions))
        super().__init__(table, dimensions, embed_data, embed_target)

    @classmethod
    def _load(cls, dimensions):
        path = cls.download('http://nlp.stanford.edu/data/glove.6B.zip')
        filename = 'glove.6B.{}d.txt'.format(dimensions)
        with ZipFile(path, 'r') as archive:
            with archive.open(filename) as file_:
                return cls._load_embeddings(file_)

    def _load_embeddings(self, file_):
        table = {}
        print('Load embeddings')
        for line in file_:
            chunks = line.split()
            key, vector = chunks[0], chunks[1:]
            key = key.decode('utf-8')
            table[key] = vector
        return table
