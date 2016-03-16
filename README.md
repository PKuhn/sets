Sets
====

Read datasets in a standard way.

Example
-------

```python
from sets import Mnist

# Download, parse and cache a dataset.
dataset = Mnist()

# Sample random batches.
data, target = dataset.train.random_batch(50)

# Iterate over all examples.
for data, target in dataset.test:
    pass
```

Parsers
-------

| Dataset | Description | Format | Size |
| ------- | ----------- | ------ | ---- |
| `Mnist` | Standard dataset of handwritten digits. | Data is normalized to 0-1 range. Targets are one-hot encoded. | 60k/10k |

Interface
---------

The parser has two properties, `train` and `test`, that both support the same
properties and methods.

| Attribute | Description |
| --------- | ----------- |
| `random_batch(size)` | Return two lists of randomly sampled data and corresponding targets. |
| `__len__()` | Number of examples. |
| `__iter__()` | Iterate over all pairs of data and targets. |
| `data` | Numpy array holding the data of all examples. Elements are float32 or string types or vectors thereof. |
| `target` | Numpy array holdingthe targets of all examples. |

Caching
-------

By default, datasets will be cached in `~/.dataset/`.

Contributions
-------------

Parsers for new datasets are welcome.

License
-------

Released under MIT license.
