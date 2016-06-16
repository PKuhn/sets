import os
import h5py
import pytest
import numpy as np

from sets.core import Embedding


class TestEmbedding:
    @pytest.fixture
    def hdf5(self, tmpdir):
        file = h5py.File(str(tmpdir) + 'test.hdf5')
        string_dt = h5py.special_dtype(vlen=str)
        # TODO find out how to create string dataset properly
        words = ['the', 'man', 'sees', 'the', 'black', 'cat']
        ds = file.create_dataset('words', (6,), dtype=string_dt)
        ds[:] = words
        return file

    @pytest.fixture
    def embeddings(self):
        words = ['the', 'man', 'sees', 'the', 'cat']
        embeddings = np.array([np.random.rand(10) for _ in range(len(words))])
        return words, embeddings

    def test_embedding(self, hdf5, tmpdir, embeddings):
        source_path = os.path.join(hdf5.filename, 'words')
        target_path = str(tmpdir.join('test.hdf5/embedded'))
        words, vector_repr = embeddings
        embedding = Embedding(words, vector_repr)
        embedded = embedding(source_path, target_path)
        dataset = embedded.get('embedded')
        assert dataset.shape == (6, 10)

    def test_average(self, hdf5, embeddings, tmpdir):
        source_path = os.path.join(hdf5.filename, 'words')
        target_path = str(tmpdir.join('test.hdf5/embedded'))
        words, vector_repr = embeddings
        embedding = Embedding(words, vector_repr)
        embedded = embedding(source_path, target_path)
        dataset = embedded['embedded']
        averaged_word = dataset[4]
        assert (averaged_word == np.mean(vector_repr, axis=0)).all()
