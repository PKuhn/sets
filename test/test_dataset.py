import os
import sets
import pytest

def test_semeval(tmpdir):
    semeval = str(tmpdir.join('semeval.hdf5'))
    dataset = sets.SemEvalRelation()(semeval, str(tmpdir))
    assert dataset['sentences'].shape == (8000,)
    semeval_sents = os.path.join(semeval, 'sentences')
    semeval_tokenized = os.path.join(semeval, 'tokenized')
    dataset = sets.Tokenize()(semeval_sents, semeval_tokenized)
    assert dataset['tokenized'].shape == (8000, 97)
    semeval_labels = os.path.join(semeval, 'target')
    index_encoded = os.path.join(semeval, 'encoded')
    sets.IndexEncode()(semeval_labels, index_encoded)
    semeval_embedded = os.path.join(semeval, 'embedded')
    embedding_dim = 100
    dataset = sets.Glove(embedding_dim)(semeval_tokenized, semeval_embedded)
    assert dataset['embedded'].shape == (8000, 97, embedding_dim)

@pytest.mark.skip
def test_ocr():
    pass
    # dataset = sets.Ocr()
    # dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])


@pytest.mark.skip
def test_wikipedia():
    url = 'https://dumps.wikimedia.org/enwiki/20160501/' \
          'enwiki-20160501-pages-meta-current1.xml-p000000010p000030303.bz2'
   #  dataset = sets.Wikipedia(url, 100)
