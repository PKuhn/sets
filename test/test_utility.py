import os
import h5py
import numpy as np
import pytest
import sets


@pytest.fixture
def dataset():
    data = [[1, 3], [0, -1.5], [0, 0]]
    target = [0, 0.5, 1]
    return sets.Dataset(data=data, target=target)


@pytest.fixture
def hdf5(tmpdir):
    file = h5py.File(str(tmpdir) + 'test.hdf5')
    numbers = [1, 2, 3, 4, 2]
    file.create_dataset('numbers', data=numbers, dtype=float)
    return file


def test_concat(dataset):
    dataset['other'] = [[1], [2], [3]]
    result = sets.Concat(1, 'data')(dataset, columns=('data', 'other'))
    assert 'other' not in result.columns
    assert (result.target == dataset.target).all()
    assert len(result) == len(dataset)
    assert result.data.shape[1] == dataset.data.shape[1] + 1
    assert (result.data[:, :-1] == dataset.data).all()


def test_index_encoding(hdf5):
    source_path = os.path.join(hdf5.filename, 'numbers')
    target_path = os.path.join(hdf5.filename, 'encoded')
    result = sets.IndexEncode()(source_path, target_path)
    expected = np.array([0, 1, 2, 3, 1], dtype=np.int32)
    assert (result['encoded'][:] == expected).all()


def test_split(dataset):
    one, two = sets.Split(0.5)(dataset)
    assert len(one) + len(two) == len(dataset)
    data = np.concatenate((one.data, two.data))
    target = np.concatenate((one.target, two.target))
    assert (data == dataset.data).all()
    assert (target == dataset.target).all()


def test_normalize(dataset):
    width = dataset.data.shape[1]
    normalize = sets.Normalize(dataset)
    result = normalize(dataset)
    assert np.allclose(result.data.mean(axis=0), np.zeros(width))
    assert np.allclose(result.data.std(axis=0), np.ones(width))
    assert np.allclose(result.target.mean(axis=0), np.zeros(width))
    assert np.allclose(result.target.std(axis=0), np.ones(width))
    shifted = dataset.data + 1
    other = normalize(sets.Dataset(data=shifted, target=dataset.target))
    assert not np.allclose(other.data.mean(axis=0), np.zeros(width))
    assert np.allclose(other.data.std(axis=0), np.ones(width))
    assert np.allclose(result.target.mean(axis=0), np.zeros(width))
    assert np.allclose(result.target.std(axis=0), np.ones(width))
