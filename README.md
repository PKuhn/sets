Sets
====

Read datasets in a standard way.

Example
-------

```python
from sets import Mnist

# Download, parse and cache the dataset.
train, test = Mnist()

# Sample random batches.
data, target = train.sample(50)

# Iterate over all examples.
for data, target in test:
    pass
```

Datasets
--------

| Dataset | Description | Format | Size |
| ------- | ----------- | ------ | ---- |
| `Mnist` | Standard dataset of handwritten digits. | Data is normalized to 0-1 range. Targets are one-hot encoded. | 60k/10k |
| `SemEvalRelation` | Relation classification from the SemEval 2010 conference. | String sentences with entity tags `<e1>` and `<e2>`. | 8k |
| `Ocr` | Handwritten letter sequences. | Binary images of 16x8 pixels. | 6877 |

Processes
---------

| Utility | Description |
| ------- | ----------- |
| `Concat` | Concatenate the specified columns of a dataset. |
| `Glove` | Replace words by pre-trained vectors from the Glovel mode. |
| `Normalize` | Fit mean and std to a dataset and then normalize any dataset by that. |
| `OneHot` | Replace words by their index in a specified list. |
| `Split` | Split a dataset according to one or more ratios. |
| `Tokenize` | Split and padd sentences using NLTK. Preserve tags in angle brackets. |
| `WordDistance` | Add a column of offsets to the provided words. |

Interface
---------

The `Dataset` class holds data columns that are immutable Numpy arrays, equal
in length. Strings index columns and integers index rows. Supports indexing by
value, slice and list.

| Attribute | Description |
| --------- | ----------- |
| `dataset.columns` | Sorted list of columns. |
| `dataset.column`, `dataset['column']` | Get a copy of this column's Numpy array. |
| `del dataset['column']` | Drop one or more columns. |
| `len(dataset)` | Number of rows. Each column will be of that length. |
| `for row in dataset` | Iterate over all rows as tuples. Tuples are sorted by column names. |
| `dataset.sample(size)` | Return new dataset of `size` randomly sampled rows. |
| `dataset.copy()` | Perform a deep copy. |

The `Step` class is used for producing and processing datasets. All steps have
a `__call__()` function that returns one or more dataset objects. For example,
a parser may return the training set and the test set.

```python
train, test = sets.Mnist()
```

An embedding class may take as parameters to `__call__()` a dataset with string
values and return a version of this dataset with the words replaced by their
embeddings.

```python
tokenize = sets.Tokenize()
glove = sets.Glove()
dataset = tokenize(dataset, columns=['data'])
dataset = glove(dataset, columns=['data'])
```

Caching
-------

By default, datasets will be cached inside `~/.dataset/sets/`. To save even
more time, use the `@sets.disk_cache(basename, directory, method=False)`
decorator and apply it to your whole pipeline. It hashes function arguments in
order to determine if a cache is valid.

Contributions
-------------

Parsers for new datasets are welcome.

License
-------

Released under the MIT license.
