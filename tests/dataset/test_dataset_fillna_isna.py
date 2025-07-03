import pytest

from atrax import Dataset


def test_isna():
    ds = Dataset([
        {'a': 1, 'b': None},
        {'a': None, 'b': 2},
        {'a': None, 'b': None},
    ])
    result = ds.isna()
    assert result.data == [
        {'a': False, 'b': True},
        {'a': True, 'b': False},
        {'a': True, 'b': True},
    ]

def test_fillna_scalar():
    ds = Dataset([
        {'a': None, 'b': 1},
        {'a': 2, 'b': None},
    ])
    result = ds.fillna(0)
    assert result.data == [
        {'a': 0, 'b': 1},
        {'a': 2, 'b': 0},
    ]

def test_fillna_dict():
    ds = Dataset([
        {'a': None, 'b': None, 'c': 1},
        {'a': 1, 'b': None, 'c': None},
    ])
    result = ds.fillna({'a': -1, 'b': -2, 'c': 0})
    assert result.data == [
        {'a': -1, 'b': -2, 'c': 1},
        {'a': 1, 'b': -2, 'c': 0},
    ]

def test_fillna_inplace():
    ds = Dataset([{'a': None, 'b': 2}])
    ds.fillna(99, inplace=True)
    assert ds.data == [{'a': 99, 'b': 2}]
