from time import gmtime, strftime
from itertools import groupby
from zipfile import ZipFile
import re
import numpy as np
from nltk import word_tokenize
from sets.core import Step, Dataset, Embedder


class SemEvalRelation(Step):
    """
    The SemEvalRelation dataset corresponds to the SemEval 2010 task
    of classifying relations.
    """

    def __call__(self):
        return self.cache('train', self._parse_train)

    def _url(self):
        timestring = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        timestring = timestring.replace(' ', '%20')
        return 'http://semeval2.fbk.eu/get.php?id=0&time={}'.format(timestring)

    def _parse_train(self):
        train_path = self.download(self._url(), 'SemEvalRelation.zip')
        with ZipFile(train_path, 'r') as semeval_zip:
            file_name = 'SemEval2010_task8_data_release/TRAIN_FILE.TXT'
            with semeval_zip.open(file_name) as train_file:
                return SemEvalRelation.parse_sem_eval(train_file)

    @staticmethod
    def parse_sem_eval(f):
        print('Parsing sem eval')
        examples = [list(g) for k, g in groupby(f, lambda x: x != b'\r\n') if k]
        train = [SemEvalRelation.preprocess_sentence(example[0].strip())
                 for example in examples]
        target = [example[1].strip() for example in examples]
        return Dataset(train, target)

    @staticmethod
    def preprocess_sentence(line):
        # Make free space after E1 and E2 so that it can be seperated
        # bla<E1>asd</E1> => bla E1
        line = str(line)
        line = re.sub(r"<e1>(.*)</e1>", " E1 ", line)
        line = re.sub(r"<e2>(.*)</e2>", " E2 ", line)
        return line


class SemEvalEmbedder(Embedder):
    def __call__(self, dataset):
        return self.cache('embedded', self.convert_dataset, dataset)

    def convert_dataset(self, dataset):
        embeddings = self._get_embeddings()
        matrices = SemEvalEmbedder.extract_as_matrices(dataset, embeddings)
        return Dataset(matrices, dataset.target)

    @staticmethod
    def extract_as_matrices(dataset, embeddings):
        max_sent_length = max([len(word_tokenize(sent)) for sent
                              in dataset.data])
        matrices = []
        for idx, sent in enumerate(dataset.data):
            if idx % 100 == 0:
                print("Processed {} of 8000 relations".format(idx))
            matrices.append(SemEvalEmbedder.get_matrix_for_relation(sent,
                            max_sent_length, embeddings))
        print('Finished converting relations')
        return matrices

    @staticmethod
    def get_matrix_for_relation(sent, max_length, embeddings):
        tokenized_sent = word_tokenize(sent)
        word_arrays = [
            SemEvalEmbedder._get_array_for_word(tokenized_sent, word, idx,
            embeddings) for idx, word in enumerate(tokenized_sent)]
        padding = [np.array([0 for _ in range(embeddings.dimension + 2)]) for _
                   in range(max_length - len(word_arrays))]
        return np.vstack(word_arrays + padding)

    @staticmethod
    def _get_array_for_word(tokenized_sent, word, word_pos, embeddings):
        first_pos = tokenized_sent.index('E1')
        second_pos = tokenized_sent.index('E2')
        first_relative = [word_pos - first_pos]
        second_relative = [word_pos - second_pos]
        embedding = embeddings[word]
        return np.array(first_relative + second_relative + embedding)
