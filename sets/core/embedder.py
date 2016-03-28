from collections import defaultdict
from sets.core import Step
from zipfile import ZipFile


class Embedder(Step):
    """
    This class is a base class for steps which involve converting words
    into vector embeddings. It is responsible for the download and parsing
    of embeddings.
    The pretrained word embeddings from the Standfordi NLP group created by
    the glove model, which is described http://nlp.stanford.edu/projects/glove/
    """
    def __init__(self):
        self.possible_dimensions = [50, 100, 300]

    def _get_embeddings(self, dimension=300):
        if dimension not in self.possible_dimensions:
            raise ValueError('Not a correct dimension')

        file_path = self.download('http://nlp.stanford.edu/data/glove.6B.zip')
        file_name = 'glove.6B.{}d.txt'.format(dimension)
        with ZipFile(file_path, 'r') as embedding_zip:
            with embedding_zip.open(file_name) as embedding_file:
                return self._load_embeddings(embedding_file, dimension)

    def _load_embeddings(self, embedding_file, dimension):
        embeddings = Embeddings(dimension)
        print("Starting to load embeddings")
        for line in embedding_file:
            splitted = line.split()
            embeddings[splitted[0]] = splitted[1:]
        print("Finished loading embeddings")
        print(embeddings.embeddings.keys())
        return embeddings


class Embeddings():
    def __init__(self, dimension):
        self.embeddings = {}
        self.dimension = dimension

    def __setitem__(self, key, val):
        if len(val) != self.dimension:
            raise ValueError('embedding has not the right length')
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        self.embeddings[key] = val

    def __getitem__(self, key):
        if key in self.embeddings:
            return self.embeddings[key]
        return self._get_example_embedding()

    def _get_example_embedding(self):
        return [0 for _ in range(self.dimension)]
