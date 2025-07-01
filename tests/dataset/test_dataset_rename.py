import pytest
from atrax import Dataset

def test_rename_basic():
    ds = Dataset([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    renamed = ds.rename(columns={'a': 'x', 'b': 'y'})
    assert renamed.data == [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]

def test_rename_inplace():
    ds = Dataset([{"foo": 10, "bar": 20}])
    ds.rename(columns={"foo": "FOO"}, inplace=True)
    assert ds.data == [{"FOO": 10, "bar": 20}]
    assert ds.columns == ["FOO", "bar"] 

def test_rename_no_columns_passed():
    ds = Dataset([{"a": 1}])
    renamed = ds.rename()
    assert renamed.data == ds.data  # no changes
    assert renamed is ds  # should return self       

def test_rename_invalid_columns_type():
    ds = Dataset([{"a": 1}])
    try:
        ds.rename(columns=["a", "b"])
    except TypeError as e:
        assert str(e) == "`columns` must be a dictionary mapping old column names to new names"
    else:
        assert False, "Expected TypeError for non-dict columns"   

def test_rename_partial():
    ds = Dataset([{"a": 5, "b": 6}])
    renamed = ds.rename(columns={"a": "alpha"})
    assert renamed.data == [{"alpha": 5, "b": 6}]
    assert renamed.columns == ["alpha", "b"]         