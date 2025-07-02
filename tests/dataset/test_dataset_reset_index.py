import pytest

from atrax import Dataset

def test_reset_index_inplace():
    ds = Dataset([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4},
    ])
    ds._index = [10, 11]
    ds._index_name = 'custom_index'

    ds.reset_index(inplace=True)
    assert ds._index is None
    assert ds._index_name is None

def test_reset_index_not_inplace():
    ds = Dataset([
        {'a': 5, 'b': 6},
        {'a': 7, 'b': 8},
    ])
    ds._index = [100, 200]
    ds._index_name = 'store_id'

    new_ds = ds.reset_index(inplace=False)
    assert new_ds._index is None
    assert new_ds._index_name is None

    # Original dataset remains unchanged
    assert ds._index == [100, 200]
    assert ds._index_name == 'store_id'

def test_reset_index_default_behavior():
    ds = Dataset([{'x': 1}, {'x': 2}])
    assert ds.reset_index() is None  # default is inplace=True
    assert ds._index is None
    assert ds._index_name is None
