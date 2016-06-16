import os
import pytest
import h5py
import numpy as np
from sets.core import MapStep


class TestMapStep:

    @pytest.fixture
    def hdf5(self, tmpdir):
        file = h5py.File(str(tmpdir.join('test.hdf5')))
        numbers = np.array(range(4))
        string_dt = h5py.special_dtype(vlen=str)
        words = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet']
        file.create_dataset('numbers', data=numbers, dtype=float)
        # TODO find out how to create string dataset properly
        ds = file.create_dataset('words', (5,), dtype=string_dt)
        ds[:] = words
        larger_chunk = np.random.rand(10000, 5, 5)
        file.create_dataset('large', (10000, 5, 5), data=larger_chunk)
        return file

    def test_mapping(self, hdf5, tmpdir):
        source_path = os.path.join(hdf5.filename, 'numbers')
        target_path = os.path.join(hdf5.filename, 'squared')
        mapping_step = QuadraticStep()
        quadratic = mapping_step(source_path, target_path)
        assert (quadratic['squared'][:] == [x**2 for x in range(4)]).all()

    def test_batching(self, hdf5, tmpdir):
        source_path = os.path.join(hdf5.filename, 'large')
        target_path = os.path.join(str(tmpdir), 'test.hdf5/squared')
        quadratic = QuadraticStep()(source_path, target_path)
        assert quadratic['squared'].shape == (10000, 5, 5)

    def test_map_new_file(self, hdf5, tmpdir):
        source_path = os.path.join(hdf5.filename, 'numbers')
        target_path = os.path.join(str(tmpdir), 'new.hdf5/squared', 'squared')
        QuadraticStep()(source_path, target_path)
        assert os.path.exists(os.path.join(str(tmpdir), 'new.hdf5'))


class QuadraticStep(MapStep):

    def apply(self, data):
        return np.array([np.square(x) for x in data])
