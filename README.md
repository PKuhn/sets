Sets
====

Read datasets in a standard way.

Example
-------

```python
from sets import Mnist

# Download, parse and cache the dataset.
train, test = Mnist()()

# Sample random batches.
data, target = train.random_batch(50)

# Iterate over all examples.
for data, target in test:
    pass
```

Datasets
--------

| Dataset | Description | Format | Size |
| ------- | ----------- | ------ | ---- |
| `Mnist` | Standard dataset of handwritten digits. | Data is normalized to 0-1 range. Targets are one-hot encoded. | 60k/10k |
| `SemEvalRelation` | Relation classification from the SemEval 2010 conference. | String sentences with entities represented as `E1` and `E2`. | 8k |

Utilities
---------

| Utility | Description |
| ------- | ----------- |
| `Tokenize` | Split sentences using the NLTK tokenizer and pass the tokens with empty strings. |
| `Glove` | Replace string words by pre-computed vector embeddings from the Glovel mode. |
| `OneHot` | Replace values by their index in a list of provided words. |
| `Concat` | Concatenate the data rows of the provided dataests. |
| `RelativeIndices` | Return a new dataset of the relative indices of provided words in each sequence. |

Interface
---------

The dataset class is used to hold an immutable array of data and their targets.

| Attribute | Description |
| --------- | ----------- |
| `data` | Numpy array holding the data of all examples. Elements are float32 or string types or vectors thereof. |
| `target` | Numpy array holding the targets of all examples. |
| `__len__()` | Number of examples. |
| `__iter__()` | Iterate over all pairs of data and targets. |
| `random_batch(size)` | Return two lists of randomly sampled data and corresponding targets. |

The step class is used for producing and processing datasets. All steps have a `__call__(self)` function that returns one or more dataset objects. For example, a parser may return the training set and the test set.

```python
mnist = sets.Mnist()
train, test = mnist()
```

An embedding class may take as parameters to `__call__(self)` a dataset with string values and return a version of this dataset with the words replaced by their embeddings.

```python
tokenize = sets.Tokenize()
glove = sets.Glove()
dataset = sets.Tokenize(dataset)
dataset = sets.Glove(dataset)
```

Caching
-------

By default, datasets will be cached inside `~/.dataset/`.

Contributions
-------------

Parsers for new datasets are welcome.

License
-------

Released under the MIT license.
