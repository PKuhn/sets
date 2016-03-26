from itertools import groupby
from zipfile import ZipFile
import re
import numpy as np
from nltk import word_tokenize
from core import Step, Dataset


class SemEvalRelation(Step):
    """
    The SemEvalRelation dataset corresponds to the SemEval 2010 task
    of classifying relations .
    """

    def __init__(self, provider='http://semeval2.fbk.eu'):
        self._provider = provider

    def __call__(self):
        return self.cache('train', self._parse_train)

    def parse(self, train_data, test_data):
        train_data, train_target = self.parse_sem_eval(train_data)
        test_data, test_target = self.parse_sem_eval(test_data)
        return train_data, train_target, test_data, test_target

    def _url(self, ressource):
        return self._provider + '/' + ressource

    def _parse_train(self):
        train_path = self.download(self._url(
            'get.php?id=0&time=2016-03-26%2011:29:56'), 'SemEvalRelation.zip')
        print(train_path)
        with ZipFile(train_path, 'r') as semeval_zip:
            with semeval_zip.open('TRAIN.txt') as train_file:
                return SemEvalRelation.parse_sem_eval(train_file)

    @staticmethod
    def parse_sem_eval(f):
        examples = [list(g) for k, g in groupby(f, lambda x: x != '\n') if k]
        train = [SemEvalRelation.preprocess_sentence(example[0].strip())
                 for example in examples]
        target = [example[1].strip() for example in examples]
        return Dataset(train, target)

    @staticmethod
    def preprocess_sentence(line):
        # Make free space after E1 and E2 so that it can be seperated
        # bla<E1>asd</E1> => bla E1
        line = re.sub(r"<e1>(.*)</e1>", " E1 ", line)
        line = re.sub(r"<e2>(.*)</e2>", " E2 ", line)
        return line

    @staticmethod
    def extract_as_matrices(dataset, embeddings):
        max_sent_length = max([len(word_tokenize(sent)) for sent
                              in dataset.train])
        matrices = []
        for idx, sent in enumerate(dataset.train):
            if idx % 100 == 0:
                print("Processed {} of 8000 relations".format(idx))
            matrices.append(SemEvalRelation.get_matrix_for_relation(sent,
                            max_sent_length, embeddings))
        return matrices

    @staticmethod
    def get_matrix_for_relation(sent, max_length, embeddings):
        tokenized_sent = word_tokenize(sent)
        word_arrays = [
            SemEvalRelation._get_array_for_word(tokenized_sent, word, idx,
            embeddings) for idx, word in enumerate(tokenized_sent)]
        padding = [np.array([0 for _ in range(embeddings.width() + 2)]) for _
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

if __name__ == '__main__':
    data = SemEvalRelation()()
    print(data)
