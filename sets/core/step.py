import re
import os
from sets import utility
from urllib.request import urlretrieve


class Step:
    """
    A cached step for processing datasets. Base class for parsing and altering
    datasets.
    """

    @classmethod
    def disk_cache(cls, basename, function, *args, method=True, **kwargs):
        """
        Cache the return value in the correct cache directory. Set 'method' to
        false for static methods.
        """
        @utility.disk_cache(basename, cls.directory(), method=method)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper(*args, **kwargs)

    @classmethod
    def directory(cls, prefix=None):
        """
        Path that should be used for caching. Different for all subclasses.
        """
        prefix = prefix or utility.read_config().directory
        name = cls.__name__.lower()
        directory = os.path.expanduser(os.path.join(prefix, name))
        utility.ensure_directory(directory)
        return directory

    @classmethod
    def _split_dataset_path(cls, path):
        """ Split hdf5 path into file and dataset name
        Args:
            path: string of format /my/filepath/file.hdf5/dataset_name
        Returns
            Tuple (/my/filepath/file.hdf5, dataset_name)
        """
        filepattern = '(.+?.hdf5)/(.*)'
        pattern = re.compile(filepattern)
        match = pattern.match(path)
        if not match:
            raise ValueError("Invalid input path {}".format(path))
        file = match.group(1)
        dataset = match.group(2)
        return file, dataset
