from atrax import Dataset, Series
import pytest
from test_data import data

def make_dataset():
    return Dataset([
        {'name': 'Apple', 'type': 'fruit',  'price': 1.2},
        {'name': 'Broccoli', 'type': 'veggie', 'price': 2.5},
        {'name': 'Carrot', 'type': 'veggie', 'price': 1.0},
    ])

def test_get_column_as_series():
    ds = make_dataset()
    s = ds['type']
    assert isinstance(s, Series)
    assert s.data == ['fruit', 'veggie', 'veggie']
    assert s.name == 'type'

def test_column_subset_as_dataset():
    ds = make_dataset()
    subset = ds[['name', 'price']]
    assert isinstance(subset, Dataset)
    assert subset.data == [
        {'name': 'Apple', 'price': 1.2},
        {'name': 'Broccoli', 'price': 2.5},
        {'name': 'Carrot', 'price': 1.0},
    ]    

def test_boolean_series_filter():
    ds = make_dataset()
    mask = Series([True, False, True])
    filtered = ds[mask]
    assert isinstance(filtered, Dataset)
    assert len(filtered.data) == 2
    assert filtered.data == [
        {'name': 'Apple', 'type': 'fruit', 'price': 1.2},
        {'name': 'Carrot', 'type': 'veggie', 'price': 1.0}
    ]    


def test_boolean_series_mismatched_length():
    ds = make_dataset()
    mask = Series([True, False])  # too short!
    try:
        ds[mask]
        assert False, "Expected ValueError for mismatched mask length"
    except ValueError as e:
        assert "Boolean Series must match the length of the dataset" in str(e)

# def test_invalid_key_type_raises():
#     ds = make_dataset()
#     try:
#         ds[42]
#         assert False, "Expected TypeError"
#     except TypeError as e:
#         assert "Key must be a string" in str(e)     


def test_loc_boolean_series_filter():
    data = [
        {"name": "Apple", "price": 1.2},
        {"name": "Steak", "price": 10.5},
        {"name": "Carrot", "price": 0.9},
    ]
    index = ['r1', 'r2', 'r3']
    ds = Dataset(data, index=index)

    # Boolean Series mask (keep Apple and Carrot)
    mask = Series([True, False, True], index=index)

    result = ds.loc[mask]

    assert isinstance(result, Dataset)
    assert result.data == [
        {"name": "Apple", "price": 1.2},
        {"name": "Carrot", "price": 0.9}
    ]
    assert result._index == ['r1', 'r3']  

def test_loc_fallback_no_filter():
    data = [
        {"name": "Apple", "price": 1.2},
        {"name": "Steak", "price": 10.5},
    ]
    index = ['a', 'b']
    ds = Dataset(data, index=index)

    result = ds.loc[:]  # slice means: "no filtering"

    assert result.data == data
    assert result._index == index             