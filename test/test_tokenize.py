import os
import numpy as np
import pytest
import h5py
from sets.process import Tokenize


class TestTokenize:
    @pytest.fixture
    def hdf5(self, tmpdir):
        file = h5py.File(str(tmpdir) + 'test.hdf5')
        string_dt = h5py.special_dtype(vlen=str)
        sents = ['This is a sentence.', 'Another sentence']
        # TODO find out how to create string dataset properly
        ds = file.create_dataset('sents', (2,), dtype=string_dt)
        ds[:] = sents
        return file

    def test_tokenize(self, hdf5, tmpdir):
        source_path = os.path.join(hdf5.filename, 'sents')
        target_path = os.path.join(hdf5.filename, 'tokenized')
        tokenized = Tokenize()(source_path, target_path)['tokenized']
        sent1 = np.array(['this', 'is', 'a', 'sentence', '.'])
        sent2 = np.array(['another', 'sentence', '', '', ''])
        expected = (np.ndarray(shape=(2, 5), dtype=object))
        expected[:] = [sent1, sent2]
        assert (expected == tokenized).all()
