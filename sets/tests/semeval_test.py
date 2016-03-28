from nltk import word_tokenize
from unittest.mock import MagicMock
from sets.semeval import SemEvalRelation, SemEvalEmbedder
import numpy as np


def test_parsing():
    m = MagicMock()
    m.__iter__.return_value = get_example_lines()
    relations = SemEvalRelation.parse_sem_eval(m)
    assert(len(relations._data) == 2)
    assert((relations._target == np.array(['WHOLE-PART', 'OWNERSHIP'])).all())
    assert('The  E1  is made of  E2 ' == relations._data[0])


def test_sentence_embedding():
    m = MagicMock()
    m.__getitem__.return_value = [1, 2, 3]
    sent = 'The E2 belongs to the E1'
    rel = SemEvalEmbedder._get_array_for_word(word_tokenize(sent), 'the', 4, m)
    assert((np.array([-1, 3, 1, 2, 3]) == rel).all())


def test_matrix_creation():
    sent = 'The E2 belongs to the E1'
    max_length = 8
    embeddings = build_mock_embeddings()
    result = SemEvalEmbedder.get_matrix_for_relation(sent, max_length,
                                                     embeddings)
    assert(result.shape == (8, 5))
    assert((result[7] == np.array([0, 0, 0, 0, 0])).all())
    assert((result[0] == np.array([-5, -1, 1, 2, 3])).all())


def build_mock_embeddings():
    m = MagicMock()
    m.__getitem__.side_effect = access_embeddings
    m.width.return_value = 3
    return m


def access_embeddings(key):
    items = {'The': [1, 2, 3], 'the': [1, 1, 1], 'E2': [1, 1, 1],
             'E1': [4, 2, 1], 'belongs': [2, 2, 3], 'to': [1, 5, 3]}
    return items[key]


def get_example_lines():
    lines = ['The <e1>car</e1> is made of <e2>parts</e2>',
             'WHOLE-PART', 'Some explaining comment', '',
             'The <e2>car</e2> belongs to the <e1>man</e1>', 'OWNERSHIP',
             'Comment', '']
    return [line + '\n' for line in lines]
