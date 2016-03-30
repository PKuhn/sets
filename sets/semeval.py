import itertools
from zipfile import ZipFile
import re
import numpy as np
from sets.core import Step, Dataset


class SemEvalRelation(Step):
    """
    Task 8 from the SemEval 2010 conference, named "Multi-Way Classification of
    Semantic Relations Between Pairs of Nominals". Only the training set is
    returned since we believe targets are not available for the test set.
    From: http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
    """

    _regex_e1 = re.compile(r'\s?<e1>.*</e1>\s?')
    _regex_e2 = re.compile(r'\s?<e2>.*</e2>\s?')

    def __call__(self):
        return self.cache('train', self._parse_train)

    def _parse_train(self):
        filepath = self._download()
        with ZipFile(filepath, 'r') as archive:
            filename = 'SemEval2010_task8_all_data/' \
                       'SemEval2010_task8_training/TRAIN_FILE.TXT'
            with archive.open(filename) as file_:
                return self._parse(file_)

    @classmethod
    def _download(cls):
        filename = 'task8.zip'
        filepath = os.path.join(cls.folder(), filename)
        if os.path.isfile(filepath):
            return filepath
        download_page = 'http://semeval2.fbk.eu/semeval2.php?' \
                        'location=download&task_id=11&datatype=test'
        response = requests.get(download_page)
        assert response.status_code == 200
        url = re.search(r'get.php?[^"]*', response.text).group(0)
        return self.download(url, filename)

    @classmethod
    def _parse(cls, file_):
        paragraphs = itertoos.groupby(file_, lambda x: x != b'\r\n')
        paragraphs = [list(g) for k, g in paragraphs if k]
        data = [cls._process_data(x[0]) for x in paragraphs]
        target = [cls._process_target(example[1]) for x in paragraphs]
        return Dataset(data, target)

    @classmethod
    def _process_data(cls, line):
        line = line.strip()
        line = cls._regex_e1.sub(' <e1/> ', line)
        line = cls._regex_e2.sub(' <e2/> ', line)
        return line

    @staticmethod
    def _process_target(line):
        return line.strip()
