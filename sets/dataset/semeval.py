import h5py
import tempfile
import os
import itertools
from zipfile import ZipFile
import re
import requests
from sets.core import Step
from sets.utility import download


class SemEvalRelation(Step):
    """
    Task 8 from the SemEval 2010 conference, named 'Multi-Way Classification of
    Semantic Relations Between Pairs of Nominals'. Only the training set is
    returned since we believe targets are not available for the test set.
    From: http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
    """

    DOWNLOAD_PAGE = \
        'http://semeval2.fbk.eu/semeval2.php?' \
        'location=download&task_id=11&datatype=test'
    FILENAME = \
        'SemEval2010_task8_all_data/' \
        'SemEval2010_task8_training/TRAIN_FILE.TXT'

    _regex_line = re.compile(r'^[0-9]+\t"(.*)"$')
    _regex_e1 = re.compile(r'<e1>.*</e1>')
    _regex_e2 = re.compile(r'<e2>.*</e2>')

    def __call__(self, path, directory=None):
        """
        Args:
            path (str): Location where hdf5 file will be saved
        """
        assert os.path.splitext(path)[1] == '.hdf5'
        file = h5py.File(path)
        filepath = self._download_task(directory)
        self._parse_train(filepath, file)
        return file

    @classmethod
    def _parse_train(cls, filepath, hdf5):
        """
        Args:
            filepath(str): path where the semeval zip was downloaded to
            hdf5(h5py.File): file object in which the data should be saved
        """
        with ZipFile(filepath, 'r') as archive:
            with archive.open(cls.FILENAME) as file_:
                return cls._parse(file_, hdf5)

    @classmethod
    def _download_task(cls, directory=None):
        if not directory:
            directory = tempfile.TemporaryDirectory().name
        download_path = os.path.join(directory, "task8.zip")
        if os.path.exists(download_path):
            return download_path
        response = requests.get(cls.DOWNLOAD_PAGE)
        assert response.status_code == 200
        url = re.search(r'get.php?[^"]*', response.text).group(0)
        url = 'http://semeval2.fbk.eu/' + url.replace(' ', '%20')
        return download(url, download_path)

    @classmethod
    def _parse(cls, file_, hdf5):
        paragraphs = itertools.groupby(file_, lambda x: x != b'\r\n')
        paragraphs = [list(g) for k, g in paragraphs if k]
        data = [cls._process_data(x[0]) for x in paragraphs]
        target = [cls._process_target(x[1]) for x in paragraphs]
        string_dt = h5py.special_dtype(vlen=str)
        sentences = hdf5.create_dataset(
            'sentences', (len(data),), dtype=string_dt)
        sentences[:] = data
        target = hdf5.create_dataset(
            'target', (len(target),), dtype=string_dt)
        target[:] = target
        return hdf5

    @classmethod
    def _process_data(cls, line):
        line = line.decode('ascii').strip()
        line = cls._regex_line.search(line).group(1)
        line = cls._regex_e1.sub('<e1>', line)
        line = cls._regex_e2.sub('<e2>', line)
        return line

    @staticmethod
    def _process_target(line):
        line = line.decode('ascii').strip()
        return line
