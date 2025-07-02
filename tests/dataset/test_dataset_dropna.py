import pytest

from atrax import Dataset

def test_dropna_any_default():
    ds = Dataset([
        {'a': 1, 'b': 2},
        {'a': None, 'b': 2},
        {'a': 3, 'b': None},
        {'a': None, 'b': None},
    ])
    result = ds.dropna()
    assert len(result) == 1
    assert result.data == [{'a': 1, 'b': 2}]

def test_dropna_all():
    ds = Dataset([
        {'a': 1, 'b': None},
        {'a': None, 'b': 2},
        {'a': None, 'b': None},
    ])
    result = ds.dropna(how='all')
    assert len(result) == 2
    assert all(row != {'a': None, 'b': None} for row in result.data)

def test_dropna_subset():
    ds = Dataset([
        {'a': None, 'b': 1, 'c': 1},
        {'a': None, 'b': None, 'c': 1},
        {'a': 3, 'b': 3, 'c': None},
    ])
    result = ds.dropna(subset=['a', 'b'])
    assert len(result) == 1
    assert result.data == [{'a': 3, 'b': 3, 'c': None}]  

def test_dropna_thresh():
    ds = Dataset([
        {'a': 1, 'b': None, 'c': None},
        {'a': 1, 'b': 2, 'c': None},
        {'a': None, 'b': None, 'c': None},
    ])
    result = ds.dropna(thresh=2)
    assert len(result) == 1
    assert result.data == [{'a': 1, 'b': 2, 'c': None}]

def test_dropna_inplace():
    ds = Dataset([
        {'a': 1, 'b': 2},
        {'a': None, 'b': 3},
        {'a': 2, 'b': None}
    ])
    ds.dropna(inplace=True)
    assert len(ds.data) == 1
    assert ds.data == [{'a': 1, 'b': 2}]

def test_dropna_invalid_how():
    ds = Dataset([{'a': None}])
    with pytest.raises(ValueError):
        ds.dropna(how='bad_option')      