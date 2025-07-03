import pytest

from atrax import Dataset

def test_dataset_mean_basic():
    ds = Dataset([
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 4, 'b': 5, 'c': 6},
        {'a': 7, 'b': 8, 'c': 9}
    ])

    r1 = ds.mean(axis=0)
    r2 = ds.mean(axis=1)

    assert r1 == {'a': 4.0, 'b': 5.0, 'c': 6.0}
    assert r2 == [2.0, 5.0, 8.0]