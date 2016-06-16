import numpy as np
import itertools
import h5py
import os
from sets.core import Step


class MapStep(Step):
    """
    Read dataset in batches apply some mapping function.
    """

    def __call__(self, source_path, target_path,
                 overwrite=False, batch_size=1000):
        """
        Args:
            source_path (str): path to input data of format
                /path/source_file.hdf5/source_dataset
            target_path (str): path to write data to of format
                /path/target_file.hdf5/target_dataset Note that source_file and
                target_file can be identical
            overwrite (bool): states whether the result should be
                recomputed when the target dataset is already present
            batch_size (int): number of elements from the incoming dataset
                are processed per batch.

        Returns:
            hdf5 file where the data is written to.
        """
        self._validate_input(source_path)
        source_filepath, source_dataset = self._split_dataset_path(source_path)
        source_file = h5py.File(source_filepath)
        target_filepath, target_dataset = self._split_dataset_path(target_path)
        if source_filepath == target_filepath:
            target_file = source_file
        else:
            target_file = h5py.File(target_filepath)
        if target_dataset in target_file.keys() and not overwrite:
            print("Skip existing dataset")
            return target_file
        if target_dataset not in target_file.keys():
            target_dataset = self._create_target(
                target_dataset, target_file, source_file[source_dataset])

        self._map_dataset(source_file[source_dataset],
                          target_dataset, batch_size)
        return target_file

    def apply(self, batch):
        """
        Args:
            batch (np.ndarray): array with the observations in the 0 axis
        Returns:
            np.ndarray with mapped observations in the 0 axis
        """
        raise NotImplemented

    def _create_target(self, name, file, source_dataset):
        """
        Args:
            name (str): name of the dataset to create
            file (hdf5 file object): where new dataset is created
            source_dataset (h5py dataset): needed to determine shape
                of the dataset to create
        Return:
            Hdf5 dataset object of newly created dataset
        """
        shape = self._get_shape(source_dataset)
        dtype = self._get_datatype(source_dataset)
        return file.create_dataset(name, shape, dtype=dtype)

    def _validate_input(self, path):
        """ Checks whether source file exists and contains specified dataset
        Args:
            path (str): formatted path/target_file.hdf5/target_dataset
        Raises:
            ValueError: if either file or dataset not present
        """
        filepath, dataset = self._split_dataset_path(path)
        if not os.path.exists(filepath):
            raise ValueError("Specified input file does not exist")
        file = h5py.File(filepath)
        if dataset not in file.keys():
            raise ValueError("Input file has no data set {}".format(dataset))

    def _get_shape(self, dataset):
        """
        The shape of the data after it was mapped to the
        new dataset. As default the previous shape of the dataset
        is kept.
        """
        return dataset.shape

    def _get_datatype(self, source_dataset):
        """ Get datatype of the dataset the step writes into """
        return source_dataset.dtype

    def _map_dataset(self, incoming, target_dataset, batch_size):
        """
        Iterates over incoming in steps of batch_size and writes
        output of self.apply to target_dataset.
        """
        start = 0
        for batch in self._get_batches(batch_size, incoming):
            target_dataset[start: start + len(batch)] = self.apply(batch)
            start += len(batch)

    def _get_batches(self, batch_size, iterable):
        """
        Generates batches of batchsize from iterable. Last batch will
        be of the size of remaining elements.
        """
        it = iter(iterable)
        while True:
            batch = np.array(list(itertools.islice(it, batch_size)))
            if batch.shape[0] == 0:
                return
            yield batch
